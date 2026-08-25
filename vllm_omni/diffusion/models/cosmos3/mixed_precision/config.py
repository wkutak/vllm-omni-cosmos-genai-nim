# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration parsing and denoising-step policy for mixed precision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .registry import MIXED_PRECISION_FORMATS

MixedPrecisionFormat = str
ReasonerPolicy = Literal["high_precision", "base_precision"]
PrecisionPath = Literal["reasoner", "generation"]
DenseWeightCacheMode = Literal[
    "none",
    "generation",
    "all",
    "cpu_block",
    "gpu_block",
]
W8A16CacheMode = DenseWeightCacheMode

REASONER_POLICIES = frozenset({"high_precision", "base_precision"})
DENSE_WEIGHT_CACHE_MODES = frozenset(
    {
        "none",
        "generation",
        "all",
        "cpu_block",
        "gpu_block",
    }
)
W8A16_CACHE_MODES = DENSE_WEIGHT_CACHE_MODES


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Validated precision policy shared by every quantization strategy."""

    format: MixedPrecisionFormat = "none"
    first_steps: int = 3
    last_steps: int = 3
    reasoner_policy: ReasonerPolicy = "high_precision"
    # Retain the original field and additional_config spelling for API
    # compatibility. Strategies should use dense_weight_cache below.
    w8a16_cache: DenseWeightCacheMode = "gpu_block"

    @classmethod
    def from_additional_config(
        cls,
        additional_config: Mapping[str, Any] | None,
    ) -> Cosmos3MixedPrecisionConfig:
        """Parse Cosmos3 fields from vLLM-Omni's additional configuration."""
        values = additional_config or {}
        precision_format = str(values.get("cosmos3_mixed_precision_format", "none")).lower()
        if precision_format not in MIXED_PRECISION_FORMATS:
            raise ValueError(
                "cosmos3_mixed_precision_format must be one of "
                f"{sorted(MIXED_PRECISION_FORMATS)}, got {precision_format!r}"
            )

        first_steps = _non_negative_int(
            values.get("cosmos3_mixed_precision_first_steps", 3),
            "cosmos3_mixed_precision_first_steps",
        )
        last_steps = _non_negative_int(
            values.get("cosmos3_mixed_precision_last_steps", 3),
            "cosmos3_mixed_precision_last_steps",
        )

        reasoner_policy = str(
            values.get(
                "cosmos3_mixed_precision_reasoner_policy",
                "high_precision",
            )
        ).lower()
        if reasoner_policy not in REASONER_POLICIES:
            raise ValueError(
                "cosmos3_mixed_precision_reasoner_policy must be one of "
                f"{sorted(REASONER_POLICIES)}, got {reasoner_policy!r}"
            )

        w8a16_cache = str(
            values.get(
                "cosmos3_mixed_precision_w8a16_cache",
                "gpu_block",
            )
        ).lower()
        if w8a16_cache not in W8A16_CACHE_MODES:
            raise ValueError(
                f"cosmos3_mixed_precision_w8a16_cache must be one of {sorted(W8A16_CACHE_MODES)}, got {w8a16_cache!r}"
            )

        return cls(
            format=precision_format,
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
            w8a16_cache=w8a16_cache,  # type: ignore[arg-type]
        )

    @property
    def dense_weight_cache(self) -> DenseWeightCacheMode:
        """Return the format-neutral dense-weight cache selection."""
        return self.w8a16_cache

    @property
    def enabled(self) -> bool:
        """Return whether the transformer should install mixed precision."""
        return self.format != "none"

    def use_high_precision(self, step_index: int, num_steps: int) -> bool:
        """Select the high path for an actual scheduler-step index."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        # TODO: Distinguish the engine's one-step initialization request from a real
        # one-step user request so the configured precision policy can be honored.
        if num_steps == 1:
            return False
        return step_index < self.first_steps or step_index >= num_steps - self.last_steps


def _non_negative_int(value: object, name: str) -> int:
    """Validate an integer configuration field without accepting booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value
