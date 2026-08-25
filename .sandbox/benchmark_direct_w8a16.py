#!/usr/bin/env python3
"""Benchmark the direct Cosmos3 W8A16 kernel against dense BF16 execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import torch

from vllm_omni.diffusion.models.cosmos3.mixed_precision.direct_w8a16 import (
    direct_w8a16_linear,
)

PROJECTION_FAMILIES = (
    (4096, 4096),
    (4096, 1024),
    (4096, 12288),
    (12288, 4096),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, action="append", dest="m_values")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def elapsed_ms(function: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def benchmark_shape(
    m_size: int,
    k_size: int,
    n_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, int | float | bool | list[int]]:
    x = torch.randn(m_size, k_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(k_size, n_size, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.03125], device="cuda", dtype=torch.float32)
    bias = torch.randn(n_size, device="cuda", dtype=torch.bfloat16)
    dense_weight = weight.to(torch.bfloat16)
    dense_weight.mul_(scale.to(torch.bfloat16))

    direct = lambda: direct_w8a16_linear(  # noqa: E731
        x, weight, scale, bias, k_size, n_size
    )
    dense_cached = lambda: torch.nn.functional.linear(  # noqa: E731
        x, dense_weight.t(), bias
    )

    direct_output = direct()
    dense_output = dense_cached()
    difference = (direct_output.float() - dense_output.float()).abs()
    direct_ms = elapsed_ms(direct, warmup, repeats)
    dense_cached_ms = elapsed_ms(dense_cached, warmup, repeats)

    # The uncached path includes conversion, scale, transpose, and GEMM.
    def dense_uncached() -> torch.Tensor:
        materialized = weight.to(torch.bfloat16)
        materialized.mul_(scale.to(torch.bfloat16))
        return torch.nn.functional.linear(x, materialized.t(), bias)

    dense_uncached_ms = elapsed_ms(dense_uncached, warmup, repeats)
    return {
        "shape": [m_size, k_size, n_size],
        "direct_ms": direct_ms,
        "dense_cached_ms": dense_cached_ms,
        "dense_uncached_ms": dense_uncached_ms,
        "direct_vs_cached_ratio": direct_ms / dense_cached_ms,
        "direct_vs_uncached_ratio": direct_ms / dense_uncached_ms,
        "dense_sidecar_bytes": dense_weight.numel() * dense_weight.element_size(),
        "direct_sidecar_bytes": 0,
        "bit_identical": torch.equal(direct_output, dense_output),
        "different_elements": int(torch.count_nonzero(difference)),
        "total_elements": difference.numel(),
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be non-negative and --repeats positive")
    m_values = args.m_values or [256, 21600]
    if any(value <= 0 for value in m_values):
        raise ValueError("--m values must be positive")

    torch.manual_seed(args.seed)
    rows = [
        benchmark_shape(m_size, k_size, n_size, args.warmup, args.repeats)
        for m_size in m_values
        for k_size, n_size in PROJECTION_FAMILIES
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": rows,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
