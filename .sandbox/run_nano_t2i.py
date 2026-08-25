#!/usr/bin/env python3
"""Run one Cosmos3-Nano FP8 mixed-step T2I request and retain evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request",
        type=Path,
        default=Path(__file__).with_name("nano_t2i_request.json"),
    )
    parser.add_argument("--first-steps", type=int, default=3)
    parser.add_argument("--last-steps", type=int, default=3)
    parser.add_argument(
        "--reasoner-policy",
        choices=("high_precision", "base_precision"),
        default="high_precision",
    )
    parser.add_argument(
        "--w8a16-cache",
        choices=("none", "generation", "all", "gpu_block", "cpu_block"),
        default="gpu_block",
    )
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--guidance-interval-start", type=float)
    parser.add_argument("--guidance-interval-end", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--torch-compile", action="store_true")
    default_group = parser.add_mutually_exclusive_group()
    default_group.add_argument(
        "--use-default-mixed-precision",
        action="store_true",
        help="Omit mixed-precision and CUTLASS overrides to exercise checkpoint defaults.",
    )
    default_group.add_argument(
        "--force-cutlass-fp8-false",
        action="store_true",
        help="Exercise automatic mixed precision with native FP8 kernel selection.",
    )
    return parser.parse_args()


def select(value: Any, payload: dict[str, Any], name: str) -> Any:
    return payload[name] if value is None else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_asset(asset_root: Path, relative_path: str) -> Path:
    root = asset_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"asset path escapes --asset-root: {relative_path!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical T2I asset: {path}")
    return path


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def write_image(image: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(output, image)
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(output)


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text())
    sampling = request["sampling"]
    width = int(select(args.width, sampling, "width"))
    height = int(select(args.height, sampling, "height"))
    num_steps = int(select(args.num_inference_steps, sampling, "num_inference_steps"))
    guidance = float(select(args.guidance_scale, sampling, "guidance_scale"))
    flow_shift = float(select(args.flow_shift, sampling, "flow_shift"))
    request_guidance_interval = sampling["guidance_interval"]
    guidance_interval = (
        float(request_guidance_interval[0] if args.guidance_interval_start is None else args.guidance_interval_start),
        float(request_guidance_interval[1] if args.guidance_interval_end is None else args.guidance_interval_end),
    )
    if guidance_interval[0] > guidance_interval[1]:
        raise ValueError(f"invalid guidance interval: {guidance_interval}")
    seed = int(select(args.seed, sampling, "seed"))

    prompt_file = resolve_asset(args.asset_root, request["prompt_file"])
    prompt_sha256 = sha256_file(prompt_file)
    expected_prompt_sha256 = request["source_sha256"]["prompt_file"]
    if prompt_sha256 != expected_prompt_sha256:
        raise ValueError(f"canonical T2I prompt changed: expected={expected_prompt_sha256}, observed={prompt_sha256}")

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    mixed_policy = {
        "cosmos3_mixed_precision_format": "fp8",
        "cosmos3_mixed_precision_first_steps": args.first_steps,
        "cosmos3_mixed_precision_last_steps": args.last_steps,
        "cosmos3_mixed_precision_reasoner_policy": args.reasoner_policy,
        "cosmos3_mixed_precision_dense_weight_cache": args.w8a16_cache,
    }
    if args.force_cutlass_fp8_false:
        mixed_config = {}
        effective_mixed_config = mixed_policy
        configuration_source = "force_cutlass_false"
    elif args.use_default_mixed_precision:
        mixed_config = {}
        effective_mixed_config = mixed_policy
        configuration_source = "automatic"
    else:
        mixed_config = mixed_policy
        effective_mixed_config = mixed_policy
        configuration_source = "explicit"
    prompt = compact_json_file(prompt_file)

    print(
        "Cosmos3 Nano mixed inference: "
        f"model={args.model} shape={width}x{height} steps={num_steps} "
        f"guidance={guidance:g} flow_shift={flow_shift:g} "
        f"guidance_interval={guidance_interval} "
        f"first={args.first_steps} last={args.last_steps} "
        f"reasoner={args.reasoner_policy} cache={args.w8a16_cache} "
        f"configuration={configuration_source} "
        f"seed={seed}"
    )
    load_start = time.time()
    omni_kwargs = {
        "model": args.model,
        "model_class_name": "Cosmos3OmniDiffusersPipeline",
        "trust_remote_code": True,
        "enforce_eager": not args.torch_compile,
        "tensor_parallel_size": 1,
        "ulysses_degree": 1,
        "cfg_parallel_size": 1,
        "max_sequence_length": 4096,
        "model_config": {"guardrails": False},
    }
    if args.force_cutlass_fp8_false:
        omni_kwargs["force_cutlass_fp8"] = False
    elif not args.use_default_mixed_precision:
        omni_kwargs.update(
            additional_config=mixed_config,
            force_cutlass_fp8=True,
        )
    omni = Omni(**omni_kwargs)
    load_seconds = time.time() - load_start
    print(f"Omni engine ready in {load_seconds:.1f}s")

    prompt_payload = {
        "prompt": prompt,
        "negative_prompt": request.get("negative_prompt") or None,
        "modalities": ["image"],
    }
    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        seed=seed,
        max_sequence_length=4096,
        extra_args={
            "flow_shift": flow_shift,
            "guidance_interval": guidance_interval,
            "max_sequence_length": 4096,
            "use_resolution_template": False,
            "use_system_prompt": False,
            "guardrails": False,
        },
    )

    generation_start = time.time()
    outputs = omni.generate(prompt_payload, params, use_tqdm=True)
    generation_seconds = time.time() - generation_start
    if not outputs:
        raise RuntimeError("Omni returned no outputs")
    first_output = outputs[0]
    if not first_output.images:
        raise RuntimeError("Omni returned no generated images")
    image = first_output.images[0]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8:
        float_image = image.astype(np.float32)
        if float_image.min() < 0:
            float_image = (float_image + 1.0) / 2.0
        multiplier = 255.0 if float_image.max() <= 1.0 else 1.0
        image = np.clip(float_image * multiplier, 0, 255).astype(np.uint8)

    write_image(image, args.output)
    manifest = {
        "model": args.model,
        "output": str(args.output),
        "request": str(args.request),
        "source": request["source"],
        "assets": {
            "prompt_file": {
                "path": str(prompt_file),
                "sha256": prompt_sha256,
            }
        },
        "image_shape": list(image.shape),
        "mixed_precision": effective_mixed_config,
        "mixed_precision_configuration": configuration_source,
        "sampling": {
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "flow_shift": flow_shift,
            "guidance_interval": list(guidance_interval),
            "seed": seed,
        },
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} shape={tuple(image.shape)} in {generation_seconds:.1f}s; manifest={manifest_path}")
    omni.close()


if __name__ == "__main__":
    main()
