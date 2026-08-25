# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime-owned dense A16 cache and double-buffer block staging."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from .strategy import Cosmos3PrecisionStrategy

PrecisionPath = Literal["reasoner", "generation"]


@dataclass
class Cosmos3PrecisionLayerState:
    module_name: str
    path: PrecisionPath
    input_size: int
    output_size: int
    block_index: int
    linear_index: int
    dense_weight: torch.Tensor | None = None
    block_offset: int = 0


@dataclass(frozen=True)
class _BlockEntry:
    state: Cosmos3PrecisionLayerState
    layer: torch.nn.Module
    strategy: Cosmos3PrecisionStrategy


class Cosmos3BlockWeightStager:
    """Stage generation weights through two reusable device buffers."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self._entries: dict[int, list[_BlockEntry]] = {}
        self._blocks: list[torch.nn.Module] = []
        self._is_active: Callable[[], bool] | None = None
        self._hook_handles: list[Any] = []
        self._block_numels: list[int] = []
        self._buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self._stage_stream: Any | None = None
        self._ready_events: tuple[Any, Any] | None = None
        self._free_events: tuple[Any, Any] | None = None
        self._slot_was_used = [False, False]
        self._loaded_block_for_slot: list[int | None] = [None, None]
        self._initialized = False

    def add(
        self,
        state: Cosmos3PrecisionLayerState,
        layer: torch.nn.Module,
        strategy: Cosmos3PrecisionStrategy,
    ) -> None:
        self._entries.setdefault(state.block_index, []).append(
            _BlockEntry(state, layer, strategy)
        )

    def install(
        self,
        blocks: list[torch.nn.Module],
        is_active: Callable[[], bool],
    ) -> None:
        self._blocks = blocks
        self._is_active = is_active
        for block_index, block in enumerate(blocks):
            self._hook_handles.append(
                block.register_forward_pre_hook(self._make_pre_hook(block_index))
            )
            self._hook_handles.append(
                block.register_forward_hook(
                    self._make_post_hook(block_index),
                    always_call=True,
                )
            )

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
        if not self._blocks or set(self._entries) != set(range(len(self._blocks))):
            raise RuntimeError(
                "Cosmos3 block cache requires compatible linears in every generation block"
            )

        devices: set[torch.device] = set()
        for block_index in range(len(self._blocks)):
            entries = sorted(
                self._entries[block_index],
                key=lambda entry: entry.state.linear_index,
            )
            self._entries[block_index] = entries
            offset = 0
            for entry in entries:
                entry.state.dense_weight = None
                entry.state.block_offset = offset
                offset += entry.state.output_size * entry.state.input_size
                devices.add(entry.layer.weight.device)
            self._block_numels.append(offset)

        if len(devices) != 1:
            raise RuntimeError(f"Cosmos3 block cache spans devices: {sorted(map(str, devices))}")
        device = devices.pop()
        if device.type != "cuda":
            raise RuntimeError(f"Cosmos3 block cache requires CUDA weights, got {device}")

        max_numel = max(self._block_numels)
        self._buffers = (
            torch.empty(max_numel, dtype=self.dtype, device=device),
            torch.empty(max_numel, dtype=self.dtype, device=device),
        )
        for block_index, entries in self._entries.items():
            slot = block_index % 2
            for entry in entries:
                state = entry.state
                numel = state.output_size * state.input_size
                offset = state.block_offset
                state.dense_weight = self._buffers[slot][offset : offset + numel].view(
                    state.output_size,
                    state.input_size,
                )

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

    def _fill_slot(self, block_index: int) -> None:
        with torch.no_grad():
            for entry in self._entries[block_index]:
                target = entry.state.dense_weight
                if target is None:
                    raise RuntimeError(f"{entry.state.module_name} has no staged dense weight")
                entry.strategy.materialize_into(target, entry.layer)

    @torch.compiler.disable
    def _enqueue(self, block_index: int) -> None:
        self.initialize()
        if self._stage_stream is None or self._ready_events is None or self._free_events is None:
            raise RuntimeError("Cosmos3 block cache was not initialized")
        slot = block_index % 2
        compute_stream = current_omni_platform.current_stream()
        self._stage_stream.wait_stream(compute_stream)
        with current_omni_platform.stream(self._stage_stream):
            if self._slot_was_used[slot]:
                self._stage_stream.wait_event(self._free_events[slot])
            self._fill_slot(block_index)
            self._ready_events[slot].record(self._stage_stream)
        self._loaded_block_for_slot[slot] = block_index

    @torch.compiler.disable
    def preload_first(self) -> None:
        self.initialize()
        if self._loaded_block_for_slot[0] != 0:
            self._enqueue(0)

    @torch.compiler.disable
    def prepare_block(self, block_index: int) -> None:
        self.initialize()
        if self._ready_events is None:
            raise RuntimeError("Cosmos3 block cache ready events were not initialized")
        slot = block_index % 2
        if self._loaded_block_for_slot[slot] != block_index:
            self._enqueue(block_index)
        current_omni_platform.current_stream().wait_event(self._ready_events[slot])
        if block_index + 1 < len(self._blocks):
            self._enqueue(block_index + 1)

    @torch.compiler.disable
    def finish_block(self, block_index: int) -> None:
        if self._free_events is None:
            raise RuntimeError("Cosmos3 block cache free events were not initialized")
        slot = block_index % 2
        self._free_events[slot].record(current_omni_platform.current_stream())
        self._slot_was_used[slot] = True
        if block_index + 1 == len(self._blocks):
            self._enqueue(0)

    @torch.compiler.disable
    def reset(self) -> None:
        if self._initialized:
            current_omni_platform.synchronize()
