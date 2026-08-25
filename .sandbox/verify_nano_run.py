#!/usr/bin/env python3
"""Validate the retained Nano image, manifest, backend log, and step trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageStat

TRACE_PATTERN = re.compile(
    r"(?:Cosmos3 mixed-precision trace: strategy=fp8, steps=|"
    r"COSMOS3_MIXED_PRECISION_TRACE strategy=fp8 steps=)([A-Z0-9,]+)"
)
CACHE_PATTERN = re.compile(
    r"Cosmos3 mixed precision ready: cache=(\w+), cached_linears=(\d+), "
    r"device_cache_gib=([0-9.]+), host_cache_gib=([0-9.]+), "
    r"device_cache_bytes=(\d+), host_cache_bytes=(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--first-steps", type=int, required=True)
    parser.add_argument("--last-steps", type=int, required=True)
    parser.add_argument("--reasoner-policy", required=True)
    parser.add_argument(
        "--w8a16-cache",
        choices=("none", "generation", "all", "gpu_block", "cpu_block"),
        required=True,
    )
    return parser.parse_args()


def expected_trace(first_steps: int, last_steps: int, num_steps: int) -> list[str]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return ["W8A8"]
    return ["W8A16" if index < first_steps or index >= num_steps - last_steps else "W8A8" for index in range(num_steps)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.image, args.manifest, args.log, args.request):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty inference artifact: {path}")

    request = json.loads(args.request.read_text())
    manifest = json.loads(args.manifest.read_text())
    mixed = manifest["mixed_precision"]
    sampling = manifest["sampling"]
    assert mixed["cosmos3_mixed_precision_format"] == "fp8"
    assert mixed["cosmos3_mixed_precision_first_steps"] == args.first_steps
    assert mixed["cosmos3_mixed_precision_last_steps"] == args.last_steps
    assert mixed["cosmos3_mixed_precision_reasoner_policy"] == args.reasoner_policy
    assert mixed["cosmos3_mixed_precision_w8a16_cache"] == args.w8a16_cache
    assert sampling == request["sampling"]
    assert request["negative_prompt"] is None

    asset_root = args.asset_root.resolve()
    prompt_file = (asset_root / request["prompt_file"]).resolve()
    try:
        prompt_file.relative_to(asset_root)
    except ValueError as exc:
        raise RuntimeError(f"T2I prompt path escapes asset root: {prompt_file}") from exc
    if not prompt_file.is_file() or prompt_file.stat().st_size == 0:
        raise RuntimeError(f"missing or empty canonical T2I prompt: {prompt_file}")
    prompt_sha256 = sha256_file(prompt_file)
    assert request["source_sha256"]["prompt_file"] == prompt_sha256
    assert manifest["assets"]["prompt_file"]["sha256"] == prompt_sha256
    assert manifest["source"] == request["source"]

    with Image.open(args.image) as image:
        image.load()
        assert image.size == (sampling["width"], sampling["height"])
        channel_stddev = ImageStat.Stat(image.convert("RGB")).stddev
        if max(channel_stddev) < 1.0:
            raise RuntimeError(f"generated image is effectively uniform: stddev={channel_stddev}")

    log_text = args.log.read_text(errors="replace")
    required_messages = (
        "Using CUTLASS FP8 linear kernels",
        "Cosmos3 mixed precision installed",
    )
    for message in required_messages:
        if message not in log_text:
            raise RuntimeError(f"inference log does not prove required runtime state: {message!r}")

    cache_matches = CACHE_PATTERN.findall(log_text)
    matching = [match for match in cache_matches if match[0] == args.w8a16_cache]
    if not matching:
        raise RuntimeError(
            "inference log does not contain exact mixed-precision cache accounting "
            f"for cache {args.w8a16_cache!r}: observed={cache_matches}"
        )
    _, cached_linears, device_gib, host_gib, device_bytes, host_bytes = matching[-1]
    cached_linears = int(cached_linears)
    device_gib = float(device_gib)
    host_gib = float(host_gib)
    device_bytes = int(device_bytes)
    host_bytes = int(host_bytes)
    if args.w8a16_cache == "none":
        assert cached_linears == device_bytes == host_bytes == 0
    elif args.w8a16_cache in ("generation", "all"):
        assert cached_linears > 0 and device_gib > 10.0
        assert device_bytes > 0 and host_gib == 0.0 and host_bytes == 0
    elif args.w8a16_cache == "gpu_block":
        assert cached_linears > 0 and 0.0 < device_gib < 1.0
        assert device_bytes > 0 and host_gib == 0.0 and host_bytes == 0
    else:
        assert args.w8a16_cache == "cpu_block"
        assert cached_linears > 0 and 0.0 < device_gib < 1.0 and host_gib > 10.0
        assert device_bytes > 0 and host_bytes > 0

    traces = [match.group(1).split(",") for match in TRACE_PATTERN.finditer(log_text)]
    expected = expected_trace(
        args.first_steps,
        args.last_steps,
        sampling["num_inference_steps"],
    )
    if expected not in traces:
        raise RuntimeError(f"expected mixed trace not found: expected={expected}, observed={traces}")

    print(
        "validated Nano inference: "
        f"image={args.image} size={sampling['width']}x{sampling['height']} "
        f"sampling={sampling['num_inference_steps']}/{sampling['guidance_scale']:g}/"
        f"{sampling['flow_shift']:g} cfg={sampling['guidance_interval']} "
        f"trace={','.join(expected)}"
    )


if __name__ == "__main__":
    main()
