# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Format strategy interface shared by the Cosmos3 mixed runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from .block_cache import Cosmos3PrecisionLayerState
from .config import PrecisionPath

if TYPE_CHECKING:
    from .runtime import Cosmos3MixedPrecisionRuntime


class Cosmos3PrecisionStrategy(ABC):
    """Format-specific validation and arithmetic behind the common runtime."""

    base_label: str
    high_label: str

    @abstractmethod
    def validate_quant_config(self, quant_config: object | None) -> None:
        """Validate the checkpoint-level quantization contract."""

    @abstractmethod
    def accepts(self, method: object | None) -> bool:
        """Return whether this strategy owns a linear's quantization method."""

    @abstractmethod
    def validate_before_processing(
        self,
        method: LinearMethodBase,
        module_name: str,
    ) -> None:
        """Reject destructive backends before their post-load transform."""

    @abstractmethod
    def bind(
        self,
        layer: torch.nn.Module,
        *,
        module_name: str,
        path: PrecisionPath,
        block_index: int = 0,
        linear_index: int = 0,
    ) -> Cosmos3PrecisionLayerState:
        """Validate and bind one linear after the base post-load transform."""

    def apply_base(
        self,
        method: LinearMethodBase,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Delegate base precision to the checkpoint's original method."""
        return method.apply(layer, x, bias)

    def install_runtime(
        self,
        transformer: torch.nn.Module,
        runtime: Cosmos3MixedPrecisionRuntime,
    ) -> None:
        """Install optional strategy-owned block lifecycle hooks."""

    def finalize(self) -> None:
        """Finalize strategy resources after all linears are bound."""

    def prepare_generation(self, high_precision: bool) -> None:
        """Prepare strategy resources for one scheduler-step selection."""

    def reset(self) -> None:
        """Return asynchronous resources to a request-safe state."""

    def cache_stats(self) -> tuple[int, int, int]:
        """Return cached linear count, device bytes, and host bytes."""
        return 0, 0, 0

    @abstractmethod
    def apply_high(
        self,
        state: Cosmos3PrecisionLayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the strategy's high-precision activation path."""
