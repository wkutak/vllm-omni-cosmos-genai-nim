#!/usr/bin/env python3
"""Validate the retained Nano I2V video, manifest, source image, backend, and step trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

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
    parser.add_argument("--video", type=Path, required=True)
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
    parser.add_argument(
        "--linear-backend",
        choices=("auto", "cutlass", "marlin"),
        default="cutlass",
    )
    parser.add_argument(
        "--quantization-mode",
        choices=("checkpoint", "online_fp8"),
        default="checkpoint",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_trace(first_steps: int, last_steps: int, num_steps: int) -> list[str]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_steps == 1:
        return ["W8A8"]
    return ["W8A16" if index < first_steps or index >= num_steps - last_steps else "W8A8" for index in range(num_steps)]


def direct_resize_input(path: Path, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        source_size = image.size
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image), source_size


def decode_video(path: Path) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    return np.stack(frames, axis=0), rate


def main() -> None:
    args = parse_args()
    for path in (args.video, args.manifest, args.log, args.request):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty I2V artifact: {path}")

    request = json.loads(args.request.read_text())
    manifest = json.loads(args.manifest.read_text())
    mixed = manifest["mixed_precision"]
    sampling = manifest["sampling"]
    online_fp8 = args.quantization_mode == "online_fp8"
    expected_format = "none" if online_fp8 or args.linear_backend == "marlin" else "fp8"
    assert mixed["cosmos3_mixed_precision_format"] == expected_format
    assert mixed["cosmos3_mixed_precision_first_steps"] == args.first_steps
    assert mixed["cosmos3_mixed_precision_last_steps"] == args.last_steps
    assert mixed["cosmos3_mixed_precision_reasoner_policy"] == args.reasoner_policy
    assert mixed["cosmos3_mixed_precision_w8a16_cache"] == args.w8a16_cache
    assert manifest["quantization_mode"] == args.quantization_mode
    assert manifest["linear_backend"] == args.linear_backend
    assert sampling == request["sampling"]
    source_assets = {
        name: (args.asset_root / request[name]).resolve()
        for name in ("prompt_file", "negative_prompt_file", "input_image")
    }
    for name, path in source_assets.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty canonical I2V asset: {path}")
        observed_sha256 = sha256_file(path)
        assert request["source_sha256"][name] == observed_sha256
        assert manifest["assets"][name]["sha256"] == observed_sha256

    frames, encoded_fps = decode_video(args.video)
    expected_shape = (
        sampling["num_frames"],
        sampling["height"],
        sampling["width"],
        3,
    )
    if frames.shape != expected_shape:
        raise RuntimeError(f"unexpected I2V video shape: expected={expected_shape}, observed={frames.shape}")
    if abs(encoded_fps - sampling["fps"]) > 0.01:
        raise RuntimeError(f"unexpected encoded FPS: expected={sampling['fps']}, observed={encoded_fps}")
    sampled = frames[[0, len(frames) // 2, -1]].astype(np.float32)
    if float(sampled.std(axis=(1, 2, 3)).max()) < 1.0:
        raise RuntimeError("generated video is effectively uniform")
    motion_mae = float(np.abs(sampled[-1] - sampled[0]).mean())
    if motion_mae < 0.5:
        raise RuntimeError(f"generated video is effectively static: first/last MAE={motion_mae:.4f}")

    reference, source_size = direct_resize_input(source_assets["input_image"], sampling["width"], sampling["height"])
    conditioning = manifest["conditioning_image"]
    assert conditioning["source_size"] == list(source_size)
    assert conditioning["request_size"] == [sampling["width"], sampling["height"]]
    assert conditioning["preprocessing"] == "direct_resize_lanczos_to_request_size"
    first_frame_mae = float(np.abs(frames[0].astype(np.float32) - reference.astype(np.float32)).mean())
    if first_frame_mae > 45.0:
        raise RuntimeError(f"generated frame 0 does not resemble the conditioning image: MAE={first_frame_mae:.3f}")

    log_text = args.log.read_text(errors="replace")
    required_messages = (
        (
            "Building quantization config: fp8",
            "for Fp8PerTensorOnlineLinearMethod",
        )
        if online_fp8
        else ("Selected MarlinFP8ScaledMMLinearKernel",)
        if args.linear_backend == "marlin"
        else (
            "Using CUTLASS FP8 linear kernels",
            "Cosmos3 mixed precision installed",
        )
    )
    for message in required_messages:
        if message not in log_text:
            raise RuntimeError(f"I2V log does not prove required runtime state: {message!r}")

    if not online_fp8 and args.linear_backend != "marlin":
        cache_matches = CACHE_PATTERN.findall(log_text)
        matching = [match for match in cache_matches if match[0] == args.w8a16_cache]
        if not matching:
            raise RuntimeError(
                "I2V log does not contain exact mixed-precision cache accounting "
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
    if online_fp8:
        forbidden = (
            "Detected ModelOpt fp8 checkpoint",
            "Using CUTLASS FP8 linear kernels",
            "Cosmos3 mixed precision installed",
        )
        for message in forbidden:
            if message in log_text:
                raise RuntimeError(f"online FP8 run unexpectedly entered another path: {message!r}")
        if traces:
            raise RuntimeError("online FP8 run unexpectedly installed the mixed dispatcher")
        expected = ["W8A8"] * sampling["num_inference_steps"]
    elif args.linear_backend == "marlin":
        if "Using CUTLASS FP8 linear kernels" in log_text:
            raise RuntimeError("Marlin run unexpectedly selected CUTLASS")
        if "Cosmos3 mixed precision installed" in log_text or traces:
            raise RuntimeError("Marlin run unexpectedly installed the mixed dispatcher")
        expected = ["W8A16"] * sampling["num_inference_steps"]
    else:
        expected = expected_trace(
            args.first_steps,
            args.last_steps,
            sampling["num_inference_steps"],
        )
        if expected not in traces:
            raise RuntimeError(f"expected mixed trace not found: expected={expected}, observed={traces}")

    print(
        "validated Nano I2V inference: "
        f"video={args.video} shape={tuple(frames.shape)} fps={encoded_fps:g} "
        f"sampling={sampling['num_inference_steps']}/{sampling['guidance_scale']:g}/"
        f"{sampling['flow_shift']:g} cfg={sampling['guidance_interval']} "
        f"first_frame_mae={first_frame_mae:.3f} motion_mae={motion_mae:.3f} "
        f"quantization={args.quantization_mode} backend={args.linear_backend} "
        f"trace={','.join(expected)}"
    )


if __name__ == "__main__":
    main()
