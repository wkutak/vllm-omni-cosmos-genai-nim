# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quantization-specific arithmetic and weight ownership.

The common runtime selects base or high precision; strategies validate the
checkpoint representation and implement those paths. The FP8 strategy retains
the original ModelOpt method for W8A8 and supplies dense W8A16 weights through
an uncached path, full caches, or a block provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)

from .block_cache import (
    Cosmos3PrecisionLayerState,
    CpuW8A16BlockWeightProvider,
    GpuW8A16BlockWeightProvider,
    W8A16BlockWeightProvider,
)
from .config import (
    W8A16_CACHE_MODES,
    Cosmos3MixedPrecisionConfig,
    PrecisionPath,
    W8A16CacheMode,
)

if TYPE_CHECKING:
    from .runtime import Cosmos3MixedPrecisionRuntime

_W8A16_WEIGHT_BUFFER = "_cosmos3_w8a16_weight"


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


class Fp8W8A8W8A16Strategy(Cosmos3PrecisionStrategy):
    """Use ModelOpt FP8 weights with W8A8 base and W8A16 high paths."""

    base_label = "W8A8"
    high_label = "W8A16"

    def __init__(
        self,
        *,
        activation_dtype: torch.dtype = torch.bfloat16,
        cache_mode: W8A16CacheMode = "gpu_block",
    ) -> None:
        """Configure activation dtype and W8A16 dense-weight residency."""
        if activation_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"Cosmos3 W8A16 requires a 16-bit floating activation dtype; got {activation_dtype}")
        if cache_mode not in W8A16_CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {sorted(W8A16_CACHE_MODES)}, got {cache_mode!r}")

        self.activation_dtype = activation_dtype
        self.cache_mode = cache_mode
        self._full_cached_count = 0
        self._full_cached_bytes = 0
        self.block_provider = self._create_block_provider(cache_mode)

    def _create_block_provider(
        self,
        cache_mode: W8A16CacheMode,
    ) -> W8A16BlockWeightProvider | None:
        """Construct the selected bounded-cache provider, if any."""
        if cache_mode == "gpu_block":
            return GpuW8A16BlockWeightProvider(self.activation_dtype)
        if cache_mode == "cpu_block":
            return CpuW8A16BlockWeightProvider(self.activation_dtype)
        return None

    def install_runtime(
        self,
        transformer: torch.nn.Module,
        runtime: Cosmos3MixedPrecisionRuntime,
    ) -> None:
        """Attach bounded-cache hooks to dynamically discovered GEN blocks."""
        if self.block_provider is not None:
            self.block_provider.install(
                list(transformer.gen_layers),
                lambda: runtime.use_high_precision("generation"),
            )

    def finalize(self) -> None:
        """Allocate and populate bounded-cache resources after weight loading."""
        if self.block_provider is not None:
            self.block_provider.initialize()

    def prepare_generation(self, high_precision: bool) -> None:
        """Preload block zero when a scheduler step selects W8A16."""
        if high_precision and self.block_provider is not None:
            self.block_provider.preload_first()

    def reset(self) -> None:
        """Synchronize and release request-scoped provider state."""
        if self.block_provider is not None:
            self.block_provider.reset()

    def cache_stats(self) -> tuple[int, int, int]:
        """Report full-cache or bounded-provider memory ownership."""
        if self.block_provider is None:
            return self._full_cached_count, self._full_cached_bytes, 0
        return (
            self.block_provider.cached_linear_count,
            self.block_provider.device_bytes,
            self.block_provider.host_bytes,
        )

    def validate_quant_config(self, quant_config: object | None) -> None:
        """Require a serialized tensorwise ModelOpt FP8 checkpoint."""
        if not isinstance(quant_config, ModelOptFp8Config):
            quant_name = type(quant_config).__name__ if quant_config is not None else "no quantization config"
            raise ValueError(
                f"Cosmos3 FP8 mixed precision requires a serialized ModelOpt FP8 checkpoint; got {quant_name}"
            )
        if not quant_config.is_checkpoint_fp8_serialized or quant_config.quant_method != "FP8":
            raise ValueError(
                "Cosmos3 FP8 mixed precision supports only serialized tensorwise "
                f"ModelOpt quant_algo='FP8', got {quant_config.quant_method!r}"
            )

    def accepts(self, method: object | None) -> bool:
        """Own only ModelOpt FP8 linear methods in the initial strategy."""
        return isinstance(method, ModelOptFp8LinearMethod)

    def validate_before_processing(
        self,
        method: LinearMethodBase,
        module_name: str,
    ) -> None:
        """Reject Marlin before it destroys the canonical FP8 matrix."""
        fp8_kernel = getattr(method, "fp8_linear", None)
        if isinstance(fp8_kernel, MarlinFP8ScaledMMLinearKernel):
            raise ValueError(
                f"{module_name} selected Marlin FP8, which repacks layer.weight and does not support W8A8. "
                "Cosmos3 mixed W8A8/W8A16 requires a native FP8 backend that retains the canonical weight; "
                "use CUTLASS on SM89+ (for example, force_cutlass_fp8=True)."
            )

    def bind(
        self,
        layer: torch.nn.Module,
        *,
        module_name: str,
        path: PrecisionPath,
        block_index: int = 0,
        linear_index: int = 0,
    ) -> Cosmos3PrecisionLayerState:
        """Validate canonical tensors and connect the selected weight source."""
        weight, weight_scale, input_size, output_size = self._validate_layer(
            layer,
            module_name,
        )
        state = Cosmos3PrecisionLayerState(
            module_name=module_name,
            path=path,
            input_size=input_size,
            output_size=output_size,
            block_index=block_index,
            linear_index=linear_index,
        )

        if self.block_provider is not None and path == "generation":
            self.block_provider.add(state, layer)
            return state

        if self._should_cache_full_weight(path):
            state.cached_weight_name = self._register_full_weight_cache(
                layer,
                module_name=module_name,
                weight=weight,
                weight_scale=weight_scale,
                input_size=input_size,
                output_size=output_size,
            )
        return state

    def _validate_layer(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Validate the physical post-load tensor and return logical metadata."""
        if hasattr(layer, "pre_quant_scale"):
            raise ValueError(
                f"{module_name} contains pre_quant_scale; SmoothQuant checkpoints are not supported "
                "by Cosmos3 FP8 mixed precision yet."
            )

        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"{module_name} has no tensor weight after FP8 post-load processing")
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"{module_name} must retain a canonical float8_e4m3fn weight; got {weight.dtype}. "
                "The selected FP8 backend may have repacked or hidden it."
            )
        if weight.ndim != 2:
            raise ValueError(
                f"{module_name} must retain a rank-2 canonical FP8 weight; got shape {tuple(weight.shape)}"
            )

        input_size = int(getattr(layer, "input_size_per_partition", -1))
        output_size = int(getattr(layer, "output_size_per_partition", -1))
        if input_size <= 0 or output_size <= 0:
            raise ValueError(f"{module_name} has invalid logical dimensions input={input_size}, output={output_size}")
        if weight.shape[0] < input_size or weight.shape[1] < output_size:
            raise ValueError(
                f"{module_name} canonical FP8 weight shape {tuple(weight.shape)} does not cover "
                f"the logical (K, N)=({input_size}, {output_size}) dimensions"
            )

        weight_scale = getattr(layer, "weight_scale", None)
        if not isinstance(weight_scale, torch.Tensor) or weight_scale.numel() != 1:
            count = None if not isinstance(weight_scale, torch.Tensor) else weight_scale.numel()
            raise ValueError(f"{module_name} requires one per-tensor FP8 weight scale, got {count}")
        scale = weight_scale.detach().float()
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{module_name} has a non-finite or non-positive FP8 weight scale")

        return weight, weight_scale, input_size, output_size

    def _should_cache_full_weight(self, path: PrecisionPath) -> bool:
        """Return whether this path owns a per-linear dense device cache."""
        return self.cache_mode == "all" or (self.cache_mode == "generation" and path == "generation")

    def _register_full_weight_cache(
        self,
        layer: torch.nn.Module,
        *,
        module_name: str,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_size: int,
        output_size: int,
    ) -> str:
        """Materialize and register one non-persistent contiguous dense weight."""
        cached_weight = torch.empty(
            (output_size, input_size),
            dtype=self.activation_dtype,
            device=weight.device,
        )
        cached_weight.copy_(weight[:input_size, :output_size].t())
        cached_weight.mul_(
            weight_scale.reshape(1).to(
                device=weight.device,
                dtype=self.activation_dtype,
            )
        )

        if _W8A16_WEIGHT_BUFFER in layer._buffers:
            layer._buffers[_W8A16_WEIGHT_BUFFER] = cached_weight
        elif hasattr(layer, _W8A16_WEIGHT_BUFFER):
            raise RuntimeError(f"{module_name} already defines reserved attribute {_W8A16_WEIGHT_BUFFER}")
        else:
            layer.register_buffer(
                _W8A16_WEIGHT_BUFFER,
                cached_weight,
                persistent=False,
            )

        self._full_cached_count += 1
        self._full_cached_bytes += cached_weight.numel() * cached_weight.element_size()
        return _W8A16_WEIGHT_BUFFER

    def apply_high(
        self,
        state: Cosmos3PrecisionLayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run W8A16 using staged, fully cached, or reconstructed weights."""
        output_shape = (*x.shape[:-1], state.output_size)
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[1] != state.input_size:
            raise ValueError(f"{state.module_name} expected activation width {state.input_size}, got {x_2d.shape[1]}")

        weight = self._resolve_dense_weight(state, layer, x.dtype)
        output = F.linear(x_2d, weight, bias)
        return output.view(output_shape)

    def _resolve_dense_weight(
        self,
        state: Cosmos3PrecisionLayerState,
        layer: torch.nn.Module,
        activation_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resolve a contiguous `(N, K)` weight for the dense reference GEMM."""
        if state.staged_weight is not None:
            return self._validate_dense_weight(
                state,
                state.staged_weight,
                activation_dtype,
                source="staged",
            )

        if state.cached_weight_name is not None:
            cached_weight = getattr(layer, state.cached_weight_name, None)
            if not isinstance(cached_weight, torch.Tensor):
                raise RuntimeError(f"{state.module_name} is missing its W8A16 weight cache")
            return self._validate_dense_weight(
                state,
                cached_weight,
                activation_dtype,
                source="cache",
            )

        weight = layer.weight[: state.input_size, : state.output_size]
        scale = layer.weight_scale.reshape(1).to(
            device=weight.device,
            dtype=activation_dtype,
        )
        return (weight.to(dtype=activation_dtype) * scale).t()

    @staticmethod
    def _validate_dense_weight(
        state: Cosmos3PrecisionLayerState,
        weight: torch.Tensor,
        activation_dtype: torch.dtype,
        *,
        source: str,
    ) -> torch.Tensor:
        """Validate a staged or cached dense matrix before calling F.linear."""
        expected_shape = (state.output_size, state.input_size)
        if tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"{state.module_name} {source} W8A16 shape {tuple(weight.shape)} does not match {expected_shape}"
            )
        if weight.dtype != activation_dtype:
            raise TypeError(
                f"{state.module_name} {source} W8A16 dtype {weight.dtype} "
                f"does not match activation dtype {activation_dtype}"
            )
        return weight


def create_cosmos3_precision_strategy(
    config: Cosmos3MixedPrecisionConfig,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> Cosmos3PrecisionStrategy:
    """Create the format-specific strategy selected by common configuration."""
    if config.format == "fp8":
        return Fp8W8A8W8A16Strategy(
            activation_dtype=activation_dtype,
            cache_mode=config.w8a16_cache,
        )
    raise ValueError(f"No Cosmos3 precision strategy exists for format {config.format!r}")
