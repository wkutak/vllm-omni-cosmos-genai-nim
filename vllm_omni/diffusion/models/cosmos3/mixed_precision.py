# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-denoising-step activation precision for Cosmos3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

MixedPrecisionFormat = Literal["none", "fp8"]
ReasonerPolicy = Literal["high_precision", "base_precision"]
PrecisionPath = Literal["reasoner", "generation"]
W8A16CacheMode = Literal[
    "none",
    "generation",
    "all",
    "cpu_block",
    "gpu_block",
]

_FORMATS = {"none", "fp8"}
_REASONER_POLICIES = {"high_precision", "base_precision"}
_W8A16_CACHE_MODES = {
    "none",
    "generation",
    "all",
    "cpu_block",
    "gpu_block",
}
_W8A16_WEIGHT_BUFFER = "_cosmos3_w8a16_weight"


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Common precision policy, independent of quantization arithmetic."""

    format: MixedPrecisionFormat = "none"
    first_steps: int = 3
    last_steps: int = 3
    reasoner_policy: ReasonerPolicy = "high_precision"
    w8a16_cache: W8A16CacheMode = "gpu_block"

    @classmethod
    def from_additional_config(
        cls,
        additional_config: Mapping[str, Any] | None,
    ) -> Cosmos3MixedPrecisionConfig:
        values = additional_config or {}
        precision_format = str(values.get("cosmos3_mixed_precision_format", "none")).lower()
        if precision_format not in _FORMATS:
            raise ValueError(
                "cosmos3_mixed_precision_format must be one of "
                f"{sorted(_FORMATS)}, got {precision_format!r}"
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
        if reasoner_policy not in _REASONER_POLICIES:
            raise ValueError(
                "cosmos3_mixed_precision_reasoner_policy must be one of "
                f"{sorted(_REASONER_POLICIES)}, got {reasoner_policy!r}"
            )
        w8a16_cache = str(
            values.get(
                "cosmos3_mixed_precision_w8a16_cache",
                "gpu_block",
            )
        ).lower()
        if w8a16_cache not in _W8A16_CACHE_MODES:
            raise ValueError(
                "cosmos3_mixed_precision_w8a16_cache must be one of "
                f"{sorted(_W8A16_CACHE_MODES)}, got {w8a16_cache!r}"
            )
        return cls(
            format=precision_format,  # type: ignore[arg-type]
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
            w8a16_cache=w8a16_cache,  # type: ignore[arg-type]
        )

    @property
    def enabled(self) -> bool:
        return self.format != "none"

    def use_high_precision(self, step_index: int, num_steps: int) -> bool:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        # TODO: Distinguish the engine's one-step initialization request from a real
        # one-step user request so the configured precision policy can be honored.
        if num_steps == 1:
            return False
        return step_index < self.first_steps or step_index >= num_steps - self.last_steps


@dataclass
class Cosmos3PrecisionLayerState:
    """Validated logical dimensions for one strategy-owned linear."""

    module_name: str
    path: PrecisionPath
    input_size: int
    output_size: int
    block_index: int = 0
    linear_index: int = 0
    cached_weight_name: str | None = None
    staged_weight: torch.Tensor | None = None
    block_offset: int = 0


@dataclass(frozen=True)
class _W8A16BlockEntry:
    state: Cosmos3PrecisionLayerState
    layer: torch.nn.Module


def _dequantize_fp8_weight_into(
    target: torch.Tensor,
    entry: _W8A16BlockEntry,
) -> None:
    """Materialize one logical FP8 matrix into a contiguous 16-bit view."""
    state = entry.state
    layer = entry.layer
    source = layer.weight[: state.input_size, : state.output_size]
    target.copy_(source.t())
    target.mul_(
        layer.weight_scale.reshape(1).to(
            device=target.device,
            dtype=target.dtype,
        )
    )


class _W8A16BlockWeightProvider(ABC):
    """Double-buffered block weights with pluggable staging source."""

    def __init__(self, activation_dtype: torch.dtype) -> None:
        self.activation_dtype = activation_dtype
        self._entries: dict[int, list[_W8A16BlockEntry]] = {}
        self._blocks: list[torch.nn.Module] = []
        self._is_active: Callable[[], bool] | None = None
        self._hook_handles: list[Any] = []
        self._device: torch.device | None = None
        self._block_numels: list[int] = []
        self._buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self._stage_stream: Any | None = None
        self._ready_events: tuple[Any, Any] | None = None
        self._free_events: tuple[Any, Any] | None = None
        self._slot_was_used = [False, False]
        self._loaded_block_for_slot: list[int | None] = [None, None]
        self._initialized = False
        self._used_since_reset = False
        self.device_bytes = 0
        self.host_bytes = 0

    def add(self, state: Cosmos3PrecisionLayerState, layer: torch.nn.Module) -> None:
        self._entries.setdefault(state.block_index, []).append(
            _W8A16BlockEntry(state=state, layer=layer)
        )

    def install(
        self,
        blocks: list[torch.nn.Module],
        is_active: Callable[[], bool],
    ) -> None:
        self._blocks = blocks
        self._is_active = is_active
        for block_index, block in enumerate(blocks):
            pre_handle = block.register_forward_pre_hook(
                self._make_pre_hook(block_index),
                prepend=True,
            )
            post_handle = block.register_forward_hook(
                self._make_post_hook(block_index),
                always_call=True,
            )
            self._hook_handles.extend((pre_handle, post_handle))

    def _make_pre_hook(self, block_index: int):
        def pre_hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            del module, args
            if self._is_active is not None and self._is_active():
                self.prepare_block(block_index)

        return pre_hook

    def _make_post_hook(self, block_index: int):
        def post_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            output: Any,
        ) -> Any:
            del module, args
            if self._is_active is not None and self._is_active():
                self.finish_block(block_index)
            return output

        return post_hook

    def initialize(self) -> None:
        if self._initialized:
            return
        if not self._blocks:
            raise RuntimeError("W8A16 block provider has no installed generation blocks")
        if set(self._entries) != set(range(len(self._blocks))):
            raise RuntimeError(
                "W8A16 block provider inventory mismatch: "
                f"blocks={len(self._blocks)}, populated={sorted(self._entries)}"
            )

        devices: set[torch.device] = set()
        self._block_numels = []
        for block_index in range(len(self._blocks)):
            entries = sorted(
                self._entries[block_index],
                key=lambda entry: entry.state.linear_index,
            )
            indices = [entry.state.linear_index for entry in entries]
            if indices != list(range(len(entries))):
                raise RuntimeError(
                    f"W8A16 block {block_index} has non-contiguous linear indices {indices}"
                )
            self._entries[block_index] = entries
            offset = 0
            for entry in entries:
                entry.state.block_offset = offset
                offset += entry.state.output_size * entry.state.input_size
                devices.add(entry.layer.weight.device)
            self._block_numels.append(offset)

        if len(devices) != 1:
            raise RuntimeError(f"W8A16 block provider spans devices: {sorted(map(str, devices))}")
        self._device = devices.pop()
        if self._device.type != "cuda":
            raise RuntimeError(
                "W8A16 block staging currently requires CUDA-resident FP8 weights; "
                f"got {self._device}"
            )

        self._initialize_source()
        max_numel = max(self._block_numels)
        self._buffers = (
            torch.empty(
                max_numel,
                dtype=self.activation_dtype,
                device=self._device,
            ),
            torch.empty(
                max_numel,
                dtype=self.activation_dtype,
                device=self._device,
            ),
        )
        self.device_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in self._buffers
        )

        for block_index, entries in self._entries.items():
            slot = block_index % 2
            for entry in entries:
                state = entry.state
                numel = state.output_size * state.input_size
                state.staged_weight = self._buffers[slot][
                    state.block_offset : state.block_offset + numel
                ].view(state.output_size, state.input_size)

        self._stage_stream = current_omni_platform.Stream()
        self._ready_events = (
            current_omni_platform.Event(),
            current_omni_platform.Event(),
        )
        self._free_events = (
            current_omni_platform.Event(),
            current_omni_platform.Event(),
        )
        self._initialized = True

    @abstractmethod
    def _initialize_source(self) -> None:
        """Prepare the provider-specific source without persistent HBM growth."""

    @abstractmethod
    def _fill_slot(self, block_index: int, slot: int) -> None:
        """Fill one device slot while the staging stream is current."""

    @torch.compiler.disable
    def _enqueue(self, block_index: int) -> None:
        self.initialize()
        assert self._stage_stream is not None
        assert self._ready_events is not None
        assert self._free_events is not None
        slot = block_index % 2
        compute_stream = current_omni_platform.current_stream()
        self._stage_stream.wait_stream(compute_stream)
        with current_omni_platform.stream(self._stage_stream):
            if self._slot_was_used[slot]:
                self._stage_stream.wait_event(self._free_events[slot])
            self._fill_slot(block_index, slot)
            self._ready_events[slot].record(self._stage_stream)
        self._loaded_block_for_slot[slot] = block_index

    @torch.compiler.disable
    def preload_first(self) -> None:
        self.initialize()
        self._used_since_reset = True
        if self._loaded_block_for_slot[0] != 0:
            self._enqueue(0)

    @torch.compiler.disable
    def prepare_block(self, block_index: int) -> None:
        self.initialize()
        self._used_since_reset = True
        assert self._ready_events is not None
        slot = block_index % 2
        if self._loaded_block_for_slot[slot] != block_index:
            self._enqueue(block_index)
        current_omni_platform.current_stream().wait_event(
            self._ready_events[slot]
        )
        next_block = block_index + 1
        if next_block < len(self._blocks):
            self._enqueue(next_block)

    @torch.compiler.disable
    def finish_block(self, block_index: int) -> None:
        assert self._free_events is not None
        slot = block_index % 2
        self._free_events[slot].record(current_omni_platform.current_stream())
        self._slot_was_used[slot] = True
        # A scheduler step can contain multiple CFG transformer passes. Always
        # prepare block zero after a complete high-precision pass rather than
        # predicting pass count from the next scheduler step.
        if block_index + 1 == len(self._blocks):
            self._enqueue(0)

    @torch.compiler.disable
    def reset(self) -> None:
        if self._initialized:
            # The native post-hook is always_call=True, but synchronize here as
            # a final exception/cancellation boundary before the next request.
            current_omni_platform.synchronize()
        self._used_since_reset = False


class _GpuW8A16BlockWeightProvider(_W8A16BlockWeightProvider):
    """Reconstruct the next BF16 block from resident canonical FP8 weights."""

    def _initialize_source(self) -> None:
        return

    def _fill_slot(self, block_index: int, slot: int) -> None:
        del slot
        with torch.no_grad():
            for entry in self._entries[block_index]:
                assert entry.state.staged_weight is not None
                _dequantize_fp8_weight_into(entry.state.staged_weight, entry)


class _CpuW8A16BlockWeightProvider(_W8A16BlockWeightProvider):
    """Keep BF16 blocks in pinned RAM and stream them through two CUDA slots."""

    def __init__(self, activation_dtype: torch.dtype) -> None:
        super().__init__(activation_dtype)
        self._host_blocks: list[torch.Tensor] = []

    def _initialize_source(self) -> None:
        assert self._device is not None
        max_linear_numel = max(
            entry.state.output_size * entry.state.input_size
            for entries in self._entries.values()
            for entry in entries
        )
        scratch = torch.empty(
            max_linear_numel,
            dtype=self.activation_dtype,
            device=self._device,
        )
        with torch.no_grad():
            for block_index, block_numel in enumerate(self._block_numels):
                host_block = torch.empty(
                    block_numel,
                    dtype=self.activation_dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for entry in self._entries[block_index]:
                    state = entry.state
                    numel = state.output_size * state.input_size
                    device_view = scratch[:numel].view(
                        state.output_size,
                        state.input_size,
                    )
                    _dequantize_fp8_weight_into(device_view, entry)
                    host_block[
                        state.block_offset : state.block_offset + numel
                    ].copy_(device_view.reshape(-1), non_blocking=False)
                self._host_blocks.append(host_block)
        self.host_bytes = sum(
            block.numel() * block.element_size() for block in self._host_blocks
        )

    def _fill_slot(self, block_index: int, slot: int) -> None:
        assert self._buffers is not None
        host_block = self._host_blocks[block_index]
        self._buffers[slot][: host_block.numel()].copy_(
            host_block,
            non_blocking=True,
        )

    @torch.compiler.disable
    def preload_first(self) -> None:
        # reset() drops the request-scoped pinned source so the large async
        # video-output buffer can be pinned without competing with 13 GiB of
        # cached generation weights. Recreate it only for a request that
        # actually selects W8A16; the one-step W8A8 warmup keeps its cache.
        if self._initialized and not self._host_blocks:
            self._initialize_source()
        super().preload_first()

    @torch.compiler.disable
    def reset(self) -> None:
        used_since_reset = self._used_since_reset
        super().reset()
        if used_since_reset:
            self._host_blocks.clear()
            torch.accelerator.empty_host_cache()


class Cosmos3PrecisionStrategy(ABC):
    """Format-specific validation and arithmetic behind the common runtime."""

    base_label: str
    high_label: str

    @abstractmethod
    def validate_quant_config(self, quant_config: object | None) -> None:
        """Validate the checkpoint-level quantization contract."""

    @abstractmethod
    def accepts(self, method: object | None) -> bool:
        """Return whether this strategy owns a linear method."""

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
        """Validate and bind state after the base post-load transform."""

    def apply_base(
        self,
        method: LinearMethodBase,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return method.apply(layer, x, bias)

    def install_runtime(
        self,
        transformer: torch.nn.Module,
        runtime: Cosmos3MixedPrecisionRuntime,
    ) -> None:
        """Install any strategy-owned block lifecycle hooks."""

    def finalize(self) -> None:
        """Finalize strategy state after all linears have been bound."""

    def prepare_generation(self, high_precision: bool) -> None:
        """Prepare resources for the selected generation precision."""

    def reset(self) -> None:
        """Return asynchronous strategy resources to a request-safe state."""

    def cache_stats(self) -> tuple[int, int, int]:
        """Return cached linears, device bytes, and host bytes."""
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
        if activation_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "Cosmos3 W8A16 requires a 16-bit floating activation dtype; "
                f"got {activation_dtype}"
            )
        if cache_mode not in _W8A16_CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {sorted(_W8A16_CACHE_MODES)}, got {cache_mode!r}"
            )
        self.activation_dtype = activation_dtype
        self.cache_mode = cache_mode
        self._full_cached_count = 0
        self._full_cached_bytes = 0
        if cache_mode == "gpu_block":
            self.block_provider: _W8A16BlockWeightProvider | None = (
                _GpuW8A16BlockWeightProvider(activation_dtype)
            )
        elif cache_mode == "cpu_block":
            self.block_provider = _CpuW8A16BlockWeightProvider(
                activation_dtype
            )
        else:
            self.block_provider = None

    def install_runtime(
        self,
        transformer: torch.nn.Module,
        runtime: Cosmos3MixedPrecisionRuntime,
    ) -> None:
        if self.block_provider is not None:
            self.block_provider.install(
                list(transformer.gen_layers),
                lambda: runtime.use_high_precision("generation"),
            )

    def finalize(self) -> None:
        if self.block_provider is not None:
            self.block_provider.initialize()

    def prepare_generation(self, high_precision: bool) -> None:
        if high_precision and self.block_provider is not None:
            self.block_provider.preload_first()

    def reset(self) -> None:
        if self.block_provider is not None:
            self.block_provider.reset()

    def cache_stats(self) -> tuple[int, int, int]:
        if self.block_provider is None:
            return self._full_cached_count, self._full_cached_bytes, 0
        return (
            sum(
                len(entries)
                for entries in self.block_provider._entries.values()
            ),
            self.block_provider.device_bytes,
            self.block_provider.host_bytes,
        )

    def validate_quant_config(self, quant_config: object | None) -> None:
        if not isinstance(quant_config, ModelOptFp8Config):
            raise ValueError(
                "Cosmos3 FP8 mixed precision requires a serialized ModelOpt FP8 checkpoint; "
                f"got {type(quant_config).__name__ if quant_config is not None else 'no quantization config'}"
            )
        if not quant_config.is_checkpoint_fp8_serialized or quant_config.quant_method != "FP8":
            raise ValueError(
                "Cosmos3 FP8 mixed precision supports only serialized tensorwise "
                f"ModelOpt quant_algo='FP8', got {quant_config.quant_method!r}"
            )

    def accepts(self, method: object | None) -> bool:
        return isinstance(method, ModelOptFp8LinearMethod)

    def validate_before_processing(
        self,
        method: LinearMethodBase,
        module_name: str,
    ) -> None:
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
            raise ValueError(
                f"{module_name} has invalid logical dimensions input={input_size}, output={output_size}"
            )
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

        cached_weight_name = None
        should_cache = self.cache_mode == "all" or (
            self.cache_mode == "generation" and path == "generation"
        )
        if should_cache:
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
                raise RuntimeError(
                    f"{module_name} already defines reserved attribute {_W8A16_WEIGHT_BUFFER}"
                )
            else:
                layer.register_buffer(
                    _W8A16_WEIGHT_BUFFER,
                    cached_weight,
                    persistent=False,
                )
            cached_weight_name = _W8A16_WEIGHT_BUFFER
            self._full_cached_count += 1
            self._full_cached_bytes += (
                cached_weight.numel() * cached_weight.element_size()
            )

        state.cached_weight_name = cached_weight_name
        return state

    def apply_high(
        self,
        state: Cosmos3PrecisionLayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output_shape = (*x.shape[:-1], state.output_size)
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[1] != state.input_size:
            raise ValueError(
                f"{state.module_name} expected activation width {state.input_size}, got {x_2d.shape[1]}"
            )
        if state.staged_weight is not None:
            weight = state.staged_weight
            expected_shape = (state.output_size, state.input_size)
            if tuple(weight.shape) != expected_shape:
                raise RuntimeError(
                    f"{state.module_name} staged W8A16 shape {tuple(weight.shape)} "
                    f"does not match {expected_shape}"
                )
            if weight.dtype != x.dtype:
                raise TypeError(
                    f"{state.module_name} staged W8A16 dtype {weight.dtype} "
                    f"does not match activation dtype {x.dtype}"
                )
            output = F.linear(x_2d, weight, bias)
        elif state.cached_weight_name is not None:
            weight = getattr(layer, state.cached_weight_name, None)
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError(f"{state.module_name} is missing its W8A16 weight cache")
            expected_shape = (state.output_size, state.input_size)
            if tuple(weight.shape) != expected_shape:
                raise RuntimeError(
                    f"{state.module_name} W8A16 cache shape {tuple(weight.shape)} "
                    f"does not match {expected_shape}"
                )
            if weight.dtype != x.dtype:
                raise TypeError(
                    f"{state.module_name} W8A16 cache dtype {weight.dtype} "
                    f"does not match activation dtype {x.dtype}"
                )
            output = F.linear(x_2d, weight, bias)
        else:
            weight = layer.weight[: state.input_size, : state.output_size]
            scale = layer.weight_scale.reshape(1).to(device=weight.device, dtype=x.dtype)
            weight = weight.to(dtype=x.dtype) * scale
            output = F.linear(x_2d, weight.t(), bias)
        return output.view(output_shape)


class _Cosmos3MixedPrecisionLinearMethod(LinearMethodBase):
    """Generic dispatcher retaining the checkpoint's original linear method."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        runtime: Cosmos3MixedPrecisionRuntime,
        module_name: str,
        path: PrecisionPath,
        block_index: int = 0,
        linear_index: int = 0,
    ) -> None:
        self.base_method = base_method
        self.runtime = runtime
        self.module_name = module_name
        self.path = path
        self.block_index = block_index
        self.linear_index = linear_index
        self.state: Cosmos3PrecisionLayerState | None = None

    def create_weights(self, *args, **kwargs) -> None:
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.runtime.strategy.validate_before_processing(self.base_method, self.module_name)
        self.base_method.process_weights_after_loading(layer)
        self.state = self.runtime.bind(
            layer,
            module_name=self.module_name,
            path=self.path,
            block_index=self.block_index,
            linear_index=self.linear_index,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.runtime.use_high_precision(self.path):
            return self.runtime.strategy.apply_base(self.base_method, layer, x, bias)
        if self.state is None:
            raise RuntimeError(f"{self.module_name} mixed-precision state was not bound after weight loading")
        return self.runtime.strategy.apply_high(self.state, layer, x, bias)


class Cosmos3MixedPrecisionRuntime:
    """Model-owned precision policy and wrapped Cosmos3 linear inventory."""

    def __init__(
        self,
        config: Cosmos3MixedPrecisionConfig,
        strategy: Cosmos3PrecisionStrategy,
    ) -> None:
        self.config = config
        self.strategy = strategy
        # TODO: Make generation precision request-local before Cosmos3 permits
        # interleaved denoising requests on one transformer instance.
        self._generation_high_precision = False
        self._methods: list[_Cosmos3MixedPrecisionLinearMethod] = []
        self._trace: list[str] = []
        self.last_trace: tuple[str, ...] = ()
        self.installed_counts: dict[PrecisionPath, int] = {
            "reasoner": 0,
            "generation": 0,
        }
        self.bound_counts: dict[PrecisionPath, int] = {
            "reasoner": 0,
            "generation": 0,
        }
        self.cached_weight_count = 0
        self.cached_weight_bytes = 0
        self.host_cached_weight_bytes = 0
        self._ready_logged = False

    def install(self, transformer: torch.nn.Module) -> None:
        components = {
            "reasoner": transformer.language_model.layers,
            "generation": transformer.gen_layers,
        }
        for path, component in components.items():
            for block_index, block in enumerate(component):
                linear_index = 0
                for local_name, layer in block.named_modules():
                    if not isinstance(layer, LinearBase):
                        continue
                    base_method = getattr(layer, "quant_method", None)
                    if not self.strategy.accepts(base_method):
                        continue
                    module_name = getattr(layer, "prefix", None) or (
                        f"{path}.{block_index}.{local_name}"
                    )
                    method = _Cosmos3MixedPrecisionLinearMethod(
                        base_method,
                        self,
                        module_name,
                        path,  # type: ignore[arg-type]
                        block_index,
                        linear_index,
                    )
                    layer.quant_method = method
                    self._methods.append(method)
                    self.installed_counts[path] += 1  # type: ignore[index]
                    linear_index += 1

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                "Cosmos3 mixed precision found no compatible serialized ModelOpt FP8 linears under "
                f"{missing}; discovered counts={self.installed_counts}"
            )

        self.strategy.install_runtime(transformer, self)

        logger.info(
            "Cosmos3 mixed precision installed: strategy=%s, reasoner=%s (%s), "
            "generation=first %d + last %d %s / middle %s, linears=%s",
            self.config.format,
            self.config.reasoner_policy,
            self.strategy.high_label
            if self.config.reasoner_policy == "high_precision"
            else self.strategy.base_label,
            self.config.first_steps,
            self.config.last_steps,
            self.strategy.high_label,
            self.strategy.base_label,
            self.installed_counts,
        )

    def bind(
        self,
        layer: torch.nn.Module,
        *,
        module_name: str,
        path: PrecisionPath,
        block_index: int,
        linear_index: int,
    ) -> Cosmos3PrecisionLayerState:
        state = self.strategy.bind(
            layer,
            module_name=module_name,
            path=path,
            block_index=block_index,
            linear_index=linear_index,
        )
        self.bound_counts[path] += 1
        return state

    def _log_ready_once(self) -> None:
        if self._ready_logged:
            return
        if self.bound_counts != self.installed_counts:
            raise RuntimeError(
                "Cosmos3 mixed precision linears were not all finalized after loading: "
                f"installed={self.installed_counts}, bound={self.bound_counts}"
            )
        self.strategy.finalize()
        (
            self.cached_weight_count,
            self.cached_weight_bytes,
            self.host_cached_weight_bytes,
        ) = self.strategy.cache_stats()
        logger.info(
            "Cosmos3 mixed precision ready: cache=%s, cached_linears=%d, "
            "device_cache_gib=%.3f, host_cache_gib=%.3f, "
            "device_cache_bytes=%d, host_cache_bytes=%d",
            getattr(self.strategy, "cache_mode", "none"),
            self.cached_weight_count,
            self.cached_weight_bytes / (1024**3),
            self.host_cached_weight_bytes / (1024**3),
            self.cached_weight_bytes,
            self.host_cached_weight_bytes,
        )
        self._ready_logged = True

    def finalize(self) -> None:
        """Finalize cache providers after checkpoint post-load processing."""
        self._log_ready_once()

    def use_high_precision(self, path: PrecisionPath) -> bool:
        if path == "reasoner":
            return self.config.reasoner_policy == "high_precision"
        return self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        self._log_ready_once()
        self._generation_high_precision = self.config.use_high_precision(step_index, num_steps)
        self.strategy.prepare_generation(self._generation_high_precision)
        label = self.strategy.high_label if self._generation_high_precision else self.strategy.base_label
        self._trace.append(label)

    def reset(self) -> None:
        if self._trace:
            self.last_trace = tuple(self._trace)
            logger.info(
                "COSMOS3_MIXED_PRECISION_TRACE strategy=%s steps=%s",
                self.config.format,
                ",".join(self.last_trace),
            )
        self._trace.clear()
        self._generation_high_precision = False
        self.strategy.reset()


def create_cosmos3_precision_strategy(
    config: Cosmos3MixedPrecisionConfig,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> Cosmos3PrecisionStrategy:
    if config.format == "fp8":
        return Fp8W8A8W8A16Strategy(
            activation_dtype=activation_dtype,
            cache_mode=config.w8a16_cache,
        )
    raise ValueError(f"No Cosmos3 precision strategy exists for format {config.format!r}")


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value
