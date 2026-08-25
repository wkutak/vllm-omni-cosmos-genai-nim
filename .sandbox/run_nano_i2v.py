#!/usr/bin/env python3
"""Run the canonical Cosmos3-Nano I2V request with mixed FP8 activation precision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        default=Path(__file__).with_name("nano_i2v_request.json"),
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
    parser.add_argument(
        "--linear-backend",
        choices=("auto", "cutlass", "marlin"),
        default="cutlass",
        help="Use CUTLASS mixed FP8 or standalone all-W8A16 FP8 Marlin.",
    )
    parser.add_argument(
        "--quantization-mode",
        choices=("checkpoint", "online_fp8"),
        default="checkpoint",
        help="Load checkpoint-declared quantization or quantize BF16 weights online.",
    )
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--guidance-interval-start", type=float)
    parser.add_argument("--guidance-interval-end", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--torch-compile", action="store_true")
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
        raise FileNotFoundError(f"missing canonical I2V asset: {path}")
    return path


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def validate_source_hashes(
    request: dict[str, Any],
    assets: dict[str, Path],
) -> dict[str, str]:
    expected = request.get("source_sha256", {})
    observed = {name: sha256_file(path) for name, path in assets.items()}
    mismatches = {
        name: {"expected": expected.get(name), "observed": digest}
        for name, digest in observed.items()
        if expected.get(name) != digest
    }
    if mismatches:
        raise ValueError(f"canonical I2V assets changed: {mismatches}")
    return observed


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_video_frames(payload: Any) -> np.ndarray:
    if isinstance(payload, (list, tuple)):
        if not payload:
            raise RuntimeError("Omni returned an empty video payload")
        if len(payload) == 1:
            payload = payload[0]
        else:
            payload = np.stack([to_numpy(frame) for frame in payload], axis=0)

    frames = to_numpy(payload)
    while frames.ndim > 4 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"expected a rank-4 video payload, got shape {frames.shape}")

    if frames.shape[-1] in (3, 4):
        pass
    elif frames.shape[1] in (3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    elif frames.shape[0] in (3, 4):
        frames = np.transpose(frames, (1, 2, 3, 0))
    else:
        raise ValueError(f"could not identify the RGB channel in video shape {frames.shape}")
    frames = frames[..., :3]

    if np.issubdtype(frames.dtype, np.floating):
        if not np.isfinite(frames).all():
            raise ValueError("generated video contains non-finite pixels")
        frames = frames.astype(np.float32)
        if float(frames.min()) < 0.0:
            frames = (frames + 1.0) / 2.0
        if float(frames.max()) <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0.0, 255.0).round().astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


def main() -> None:
    args = parse_args()
    request = json.loads(args.request.read_text())
    sampling = request["sampling"]
    width = int(select(args.width, sampling, "width"))
    height = int(select(args.height, sampling, "height"))
    num_frames = int(select(args.num_frames, sampling, "num_frames"))
    fps = int(select(args.fps, sampling, "fps"))
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
    online_fp8 = args.quantization_mode == "online_fp8"
    if online_fp8:
        if args.linear_backend != "auto":
            raise ValueError(
                "online_fp8 must use --linear-backend auto so vLLM selects the supported online FP8 kernel"
            )
        if args.first_steps != 0 or args.last_steps != 0:
            raise ValueError("online_fp8 is an all-W8A8 endpoint; use --first-steps 0 --last-steps 0")
        if args.w8a16_cache != "none":
            raise ValueError("online_fp8 does not use the W8A16 cache; use --w8a16-cache none")
        model_path = Path(args.model)
        checkpoint_quantization = None
        if model_path.is_dir():
            checkpoint_quantization = json.loads((model_path / "config.json").read_text()).get("quantization_config")
        if model_path.is_dir() and (
            (model_path / "hf_quant_config.json").exists() or checkpoint_quantization is not None
        ):
            raise ValueError("online_fp8 requires a BF16/FP16 checkpoint without declared checkpoint quantization")
    if args.linear_backend == "marlin":
        if online_fp8:
            raise ValueError("Marlin benchmark mode requires checkpoint quantization")
        if args.first_steps < num_steps or args.last_steps != 0:
            raise ValueError(
                "Marlin is supported here only as the standalone all-W8A16 "
                f"endpoint; use --first-steps {num_steps} --last-steps 0"
            )
        force_marlin = os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "")
        if force_marlin.strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("Marlin on SM89+ requires VLLM_TEST_FORCE_FP8_MARLIN=1")

    assets = {
        "prompt_file": resolve_asset(args.asset_root, request["prompt_file"]),
        "negative_prompt_file": resolve_asset(args.asset_root, request["negative_prompt_file"]),
        "input_image": resolve_asset(args.asset_root, request["input_image"]),
    }
    source_sha256 = validate_source_hashes(request, assets)

    from PIL import Image

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    with Image.open(assets["input_image"]) as source_image:
        source_image = source_image.convert("RGB")
        source_image_size = source_image.size
        target_size = (width, height)
        input_image = (
            source_image.resize(target_size, Image.Resampling.LANCZOS)
            if source_image.size != target_size
            else source_image.copy()
        )
    prompt = compact_json_file(assets["prompt_file"])
    negative_prompt = compact_json_file(assets["negative_prompt_file"])
    mixed_config = {
        # Online FP8 owns its load-time conversion, while Marlin owns a packed
        # W8A16 layout. Neither can use the ModelOpt mixed dispatcher.
        "cosmos3_mixed_precision_format": ("none" if online_fp8 or args.linear_backend == "marlin" else "fp8"),
        "cosmos3_mixed_precision_first_steps": args.first_steps,
        "cosmos3_mixed_precision_last_steps": args.last_steps,
        "cosmos3_mixed_precision_reasoner_policy": args.reasoner_policy,
        "cosmos3_mixed_precision_dense_weight_cache": args.w8a16_cache,
    }

    print(
        "Cosmos3 Nano mixed I2V inference: "
        f"model={args.model} input={assets['input_image']} shape={width}x{height} "
        f"frames={num_frames} fps={fps} steps={num_steps} "
        f"guidance={guidance:g} flow_shift={flow_shift:g} "
        f"guidance_interval={guidance_interval} "
        f"first={args.first_steps} last={args.last_steps} "
        f"reasoner={args.reasoner_policy} cache={args.w8a16_cache} "
        f"quantization={args.quantization_mode} backend={args.linear_backend} seed={seed}"
    )
    load_start = time.time()
    omni_kwargs = dict(
        model=args.model,
        model_class_name="Cosmos3OmniDiffusersPipeline",
        trust_remote_code=True,
        enforce_eager=not args.torch_compile,
        tensor_parallel_size=1,
        ulysses_degree=1,
        cfg_parallel_size=1,
        max_sequence_length=4096,
        model_config={"guardrails": False},
        additional_config=mixed_config,
        force_cutlass_fp8=(args.quantization_mode == "checkpoint" and args.linear_backend == "cutlass"),
    )
    if online_fp8:
        omni_kwargs["quantization"] = "fp8"
    omni = Omni(**omni_kwargs)
    load_seconds = time.time() - load_start
    print(f"Omni engine ready in {load_seconds:.1f}s")

    prompt_payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "modalities": ["video"],
        "multi_modal_data": {"image": input_image},
    }
    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        frame_rate=float(fps),
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        seed=seed,
        max_sequence_length=4096,
        extra_args={
            "flow_shift": flow_shift,
            "guidance_interval": guidance_interval,
            "max_sequence_length": 4096,
            "use_resolution_template": False,
            "use_duration_template": False,
            "use_system_prompt": False,
            "guardrails": False,
        },
    )

    try:
        generation_start = time.time()
        outputs = omni.generate(prompt_payload, params, use_tqdm=True)
        generation_seconds = time.time() - generation_start
        if not outputs:
            raise RuntimeError("Omni returned no outputs")
        images = getattr(outputs[0], "images", None)
        if images is None:
            raise RuntimeError("Omni returned no generated video frames")
        frames = normalize_video_frames(images)
        expected_shape = (num_frames, height, width, 3)
        if frames.shape != expected_shape:
            raise ValueError(f"generated video shape {frames.shape} does not match {expected_shape}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(mux_video_audio_bytes(frames, None, fps=float(fps)))
        manifest = {
            "model": args.model,
            "output": str(args.output),
            "request": str(args.request),
            "source": request["source"],
            "assets": {name: {"path": str(path), "sha256": source_sha256[name]} for name, path in assets.items()},
            "conditioning_image": {
                "source_size": list(source_image_size),
                "request_size": list(input_image.size),
                "preprocessing": "direct_resize_lanczos_to_request_size",
            },
            "frame_shape": list(frames.shape),
            "mixed_precision": mixed_config,
            "quantization_mode": args.quantization_mode,
            "linear_backend": args.linear_backend,
            "sampling": {
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
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
        print(f"wrote {args.output} shape={tuple(frames.shape)} in {generation_seconds:.1f}s; manifest={manifest_path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
