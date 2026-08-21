# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Double-buffered W8A16 generation-weight staging.

The provider owns stable per-linear views into two maximum-block device
buffers. Transformer-block hooks make one slot ready for the current block and
fill the other slot for the next block. Subclasses differ only in where the
next dense weight block comes from: resident FP8 tensors or pinned host BF16.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.platforms import current_omni_platform

from .config import PrecisionPath


@dataclass
class Cosmos3PrecisionLayerState:
    """Validated logical dimensions and runtime weight view for one linear."""

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
class W8A16BlockEntry:
    """Pair one validated linear state with its source layer."""

    state: Cosmos3PrecisionLayerState
    layer: torch.nn.Module


def dequantize_fp8_weight_into(
    target: torch.Tensor,
    entry: W8A16BlockEntry,
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


class W8A16BlockWeightProvider(ABC):
    """Stage dense W8A16 weights through two reusable device slots."""

    def __init__(self, activation_dtype: torch.dtype) -> None:
        """Create an empty provider; inventory is populated during weight load."""
        self.activation_dtype = activation_dtype
        self._entries: dict[int, list[W8A16BlockEntry]] = {}
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

    @property
    def cached_linear_count(self) -> int:
        """Return the number of linears backed by the staged block inventory."""
        return sum(len(entries) for entries in self._entries.values())

    def add(self, state: Cosmos3PrecisionLayerState, layer: torch.nn.Module) -> None:
        """Add one generation linear to its dynamically discovered block."""
        self._entries.setdefault(state.block_index, []).append(W8A16BlockEntry(state=state, layer=layer))

    def install(
        self,
        blocks: list[torch.nn.Module],
        is_active: Callable[[], bool],
    ) -> None:
        """Attach precision-aware pre/post hooks to generation blocks."""
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
        """Build a hook that waits for this block's staged weight slot."""

        def pre_hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            del module, args
            if self._is_active is not None and self._is_active():
                self.prepare_block(block_index)

        return pre_hook

    def _make_post_hook(self, block_index: int):
        """Build an exception-safe hook that releases this block's slot."""

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
        """Validate inventory and allocate streams, events, and stable views."""
        if self._initialized:
            return
        if not self._blocks:
            raise RuntimeError("W8A16 block provider has no installed generation blocks")
        if set(self._entries) != set(range(len(self._blocks))):
            raise RuntimeError(
                "W8A16 block provider inventory mismatch: "
                f"blocks={len(self._blocks)}, populated={sorted(self._entries)}"
            )

        devices = self._layout_block_entries()
        if len(devices) != 1:
            raise RuntimeError(f"W8A16 block provider spans devices: {sorted(map(str, devices))}")
        self._device = devices.pop()
        if self._device.type != "cuda":
            raise RuntimeError(f"W8A16 block staging currently requires CUDA-resident FP8 weights; got {self._device}")

        self._initialize_source()
        self._allocate_device_slots()
        self._attach_staged_views()
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

    def _layout_block_entries(self) -> set[torch.device]:
        """Sort linears, assign block offsets, and collect source devices."""
        devices: set[torch.device] = set()
        self._block_numels = []
        for block_index in range(len(self._blocks)):
            entries = sorted(
                self._entries[block_index],
                key=lambda entry: entry.state.linear_index,
            )
            indices = [entry.state.linear_index for entry in entries]
            if indices != list(range(len(entries))):
                raise RuntimeError(f"W8A16 block {block_index} has non-contiguous linear indices {indices}")
            self._entries[block_index] = entries

            offset = 0
            for entry in entries:
                entry.state.block_offset = offset
                offset += entry.state.output_size * entry.state.input_size
                devices.add(entry.layer.weight.device)
            self._block_numels.append(offset)
        return devices

    def _allocate_device_slots(self) -> None:
        """Allocate two slots sized to the largest discovered block."""
        assert self._device is not None
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
        self.device_bytes = sum(buffer.numel() * buffer.element_size() for buffer in self._buffers)

    def _attach_staged_views(self) -> None:
        """Give every linear a stable matrix view into its parity slot."""
        assert self._buffers is not None
        for block_index, entries in self._entries.items():
            slot = block_index % 2
            for entry in entries:
                state = entry.state
                numel = state.output_size * state.input_size
                state.staged_weight = self._buffers[slot][state.block_offset : state.block_offset + numel].view(
                    state.output_size, state.input_size
                )

    @abstractmethod
    def _initialize_source(self) -> None:
        """Prepare the provider-specific source without persistent HBM growth."""

    @abstractmethod
    def _fill_slot(self, block_index: int, slot: int) -> None:
        """Fill one device slot while the staging stream is current."""

    @torch.compiler.disable
    def _enqueue(self, block_index: int) -> None:
        """Fill one slot asynchronously and record its ready event."""
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
        """Make block zero available before a high-precision transformer pass."""
        self.initialize()
        self._used_since_reset = True
        if self._loaded_block_for_slot[0] != 0:
            self._enqueue(0)

    @torch.compiler.disable
    def prepare_block(self, block_index: int) -> None:
        """Wait for the current block and overlap staging of its successor."""
        self.initialize()
        self._used_since_reset = True
        assert self._ready_events is not None

        slot = block_index % 2
        if self._loaded_block_for_slot[slot] != block_index:
            self._enqueue(block_index)
        current_omni_platform.current_stream().wait_event(self._ready_events[slot])

        next_block = block_index + 1
        if next_block < len(self._blocks):
            self._enqueue(next_block)

    @torch.compiler.disable
    def finish_block(self, block_index: int) -> None:
        """Mark a slot reusable and stage block zero after a complete pass."""
        assert self._free_events is not None
        slot = block_index % 2
        self._free_events[slot].record(current_omni_platform.current_stream())
        self._slot_was_used[slot] = True

        # A scheduler step can contain multiple CFG transformer passes. Always
        # prepare block zero after a complete pass rather than predicting the
        # branch count from the next scheduler step.
        if block_index + 1 == len(self._blocks):
            self._enqueue(0)

    @torch.compiler.disable
    def reset(self) -> None:
        """Synchronize staging at the request exception/cancellation boundary."""
        if self._initialized:
            # Post-hooks are always_call=True; this synchronization is the final
            # guard against outstanding work before another request starts.
            current_omni_platform.synchronize()
        self._used_since_reset = False


class GpuW8A16BlockWeightProvider(W8A16BlockWeightProvider):
    """Reconstruct the next dense block from resident canonical FP8 weights."""

    def _initialize_source(self) -> None:
        """Use the existing FP8 tensors directly; no side source is needed."""

    def _fill_slot(self, block_index: int, slot: int) -> None:
        """Convert one block's FP8 matrices into its stable device views."""
        del slot
        with torch.no_grad():
            for entry in self._entries[block_index]:
                assert entry.state.staged_weight is not None
                dequantize_fp8_weight_into(entry.state.staged_weight, entry)


class CpuW8A16BlockWeightProvider(W8A16BlockWeightProvider):
    """Stream dense blocks from request-scoped pinned RAM into two CUDA slots."""

    def __init__(self, activation_dtype: torch.dtype) -> None:
        """Create a provider with a lazily built pinned host source."""
        super().__init__(activation_dtype)
        self._host_blocks: list[torch.Tensor] = []

    def _initialize_source(self) -> None:
        """Build one packed pinned BF16/FP16 tensor per generation block."""
        assert self._device is not None
        max_linear_numel = max(
            entry.state.output_size * entry.state.input_size for entries in self._entries.values() for entry in entries
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
                    dequantize_fp8_weight_into(device_view, entry)
                    host_block[state.block_offset : state.block_offset + numel].copy_(
                        device_view.reshape(-1), non_blocking=False
                    )
                self._host_blocks.append(host_block)

        self.host_bytes = sum(block.numel() * block.element_size() for block in self._host_blocks)

    def _fill_slot(self, block_index: int, slot: int) -> None:
        """Copy one packed pinned block to its device slot asynchronously."""
        assert self._buffers is not None
        host_block = self._host_blocks[block_index]
        self._buffers[slot][: host_block.numel()].copy_(
            host_block,
            non_blocking=True,
        )

    @torch.compiler.disable
    def preload_first(self) -> None:
        """Rebuild an evicted host source only when W8A16 is requested again."""
        # reset() drops the source so decoded-video D2H allocation does not
        # compete with roughly a full generation cache of pinned weights.
        if self._initialized and not self._host_blocks:
            self._initialize_source()
        super().preload_first()

    @torch.compiler.disable
    def reset(self) -> None:
        """Release used pinned weights before the pipeline transfers output."""
        used_since_reset = self._used_since_reset
        super().reset()
        if used_since_reset:
            self._host_blocks.clear()
            torch.accelerator.empty_host_cache()
