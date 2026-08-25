# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Format-agnostic schedule state, linear discovery, and cache dispatch."""

from __future__ import annotations

from typing import Literal

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

from .cache import (
    Cosmos3BlockWeightStager,
    Cosmos3PrecisionLayerState,
)
from .config import Cosmos3MixedPrecisionConfig
from .strategy import (
    Cosmos3PrecisionStrategy,
    Fp8W8A8W8A16Strategy,
    Nvfp4W4A4W4A16Strategy,
)

PrecisionPath = Literal["reasoner", "generation"]
_STRATEGIES: tuple[Cosmos3PrecisionStrategy, ...] = (
    Fp8W8A8W8A16Strategy(),
    Nvfp4W4A4W4A16Strategy(),
)


def _strategy_for(method: object | None) -> Cosmos3PrecisionStrategy | None:
    return next((strategy for strategy in _STRATEGIES if strategy.accepts(method)), None)


class Cosmos3MixedPrecisionLinearMethod(LinearMethodBase):
    """Dispatch between a checkpoint-native method and dense A16."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        strategy: Cosmos3PrecisionStrategy,
        runtime: Cosmos3MixedPrecisionRuntime,
        module_name: str,
        path: PrecisionPath,
        block_index: int,
        linear_index: int,
    ) -> None:
        self.base_method = base_method
        self.strategy = strategy
        self.runtime = runtime
        self.module_name = module_name
        self.path = path
        self.block_index = block_index
        self.linear_index = linear_index
        self.state: Cosmos3PrecisionLayerState | None = None

    def create_weights(self, *args, **kwargs) -> None:
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.strategy.validate_before_processing(
            self.base_method,
            layer,
            self.module_name,
        )
        self.base_method.process_weights_after_loading(layer)
        self.strategy.validate_after_processing(layer, self.module_name)
        self.state = self.runtime.bind(self, layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.runtime.use_high_precision(self.path):
            return self.base_method.apply(layer, x, bias)
        if self.state is None:
            raise RuntimeError(f"{self.module_name} cache state was not bound after weight loading")
        return self.strategy.apply_high(
            layer,
            x,
            bias,
            weight=self.state.dense_weight,
        )


class Cosmos3MixedPrecisionRuntime:
    """Own one transformer's schedule and optional dense-weight residency."""

    def __init__(
        self,
        config: Cosmos3MixedPrecisionConfig,
        activation_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.activation_dtype = activation_dtype
        self._generation_high_precision = False
        self._methods: list[Cosmos3MixedPrecisionLinearMethod] = []
        self._finalized = False
        self._block_stager = (
            Cosmos3BlockWeightStager(activation_dtype) if config.cache == "block" else None
        )

    def install(self, transformer: torch.nn.Module) -> None:
        components: dict[PrecisionPath, list[torch.nn.Module]] = {
            "generation": list(transformer.gen_layers),
        }
        if self.config.reasoner == "a16":
            components["reasoner"] = list(transformer.language_model.layers)

        for path, blocks in components.items():
            for block_index, block in enumerate(blocks):
                linear_index = 0
                for local_name, layer in block.named_modules():
                    if not isinstance(layer, LinearBase):
                        continue
                    base_method = getattr(layer, "quant_method", None)
                    strategy = _strategy_for(base_method)
                    if strategy is None:
                        continue
                    module_name = getattr(layer, "prefix", None) or f"{path}.{block_index}.{local_name}"
                    method = Cosmos3MixedPrecisionLinearMethod(
                        base_method,
                        strategy,
                        self,
                        module_name,
                        path,
                        block_index,
                        linear_index,
                    )
                    layer.quant_method = method
                    self._methods.append(method)
                    linear_index += 1

        if not self._methods:
            raise ValueError("Cosmos3 mixed precision found no compatible FP8 or NVFP4 ModelOpt linears")
        if self._block_stager is not None:
            self._block_stager.install(
                list(transformer.gen_layers),
                lambda: self.use_high_precision("generation"),
            )

    def bind(
        self,
        method: Cosmos3MixedPrecisionLinearMethod,
        layer: torch.nn.Module,
    ) -> Cosmos3PrecisionLayerState:
        state = Cosmos3PrecisionLayerState(
            module_name=method.module_name,
            path=method.path,
            input_size=int(layer.input_size_per_partition),
            output_size=int(layer.output_size_per_partition),
            block_index=method.block_index,
            linear_index=method.linear_index,
        )
        if self._block_stager is not None and method.path == "generation":
            self._block_stager.add(state, layer, method.strategy)
        return state

    def finalize(self) -> None:
        if self._finalized:
            return
        unbound = [method.module_name for method in self._methods if method.state is None]
        if unbound:
            raise RuntimeError(f"Cosmos3 mixed precision linears were not finalized: {unbound}")
        if self._block_stager is not None:
            self._block_stager.initialize()
        self._finalized = True

    def use_high_precision(self, path: PrecisionPath) -> bool:
        return self.config.reasoner == "a16" if path == "reasoner" else self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        self.finalize()
        self._generation_high_precision = self.config.use_high_precision(step_index, num_steps)
        if self._generation_high_precision and self._block_stager is not None:
            self._block_stager.preload_first()

    def reset(self) -> None:
        self._generation_high_precision = False
        if self._block_stager is not None:
            self._block_stager.reset()
