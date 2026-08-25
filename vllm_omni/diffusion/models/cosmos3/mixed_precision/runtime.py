# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned schedule state, dynamic discovery, and linear dispatch."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

from .block_cache import Cosmos3PrecisionLayerState
from .config import Cosmos3MixedPrecisionConfig, PrecisionPath
from .strategy import Cosmos3PrecisionStrategy

logger = init_logger(__name__)


class Cosmos3MixedPrecisionLinearMethod(LinearMethodBase):
    """Dispatch one linear while retaining its original quantization method."""

    def __init__(
        self,
        base_method: LinearMethodBase,
        runtime: Cosmos3MixedPrecisionRuntime,
        module_name: str,
        path: PrecisionPath,
        block_index: int = 0,
        linear_index: int = 0,
    ) -> None:
        """Record the original method and dynamic inventory coordinates."""
        self.base_method = base_method
        self.runtime = runtime
        self.module_name = module_name
        self.path = path
        self.block_index = block_index
        self.linear_index = linear_index
        self.state: Cosmos3PrecisionLayerState | None = None

    def create_weights(self, *args, **kwargs) -> None:
        """Delegate checkpoint parameter creation to the original method."""
        self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Run safe base post-load processing and bind validated strategy state."""
        self.runtime.strategy.validate_before_processing(
            self.base_method,
            self.module_name,
        )
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
        """Dispatch the call using the runtime's current path selection."""
        if not self.runtime.use_high_precision(self.path):
            return self.runtime.strategy.apply_base(self.base_method, layer, x, bias)
        if self.state is None:
            raise RuntimeError(f"{self.module_name} mixed-precision state was not bound after weight loading")
        return self.runtime.strategy.apply_high(self.state, layer, x, bias)


class Cosmos3MixedPrecisionRuntime:
    """Own one transformer's schedule state and wrapped linear inventory."""

    def __init__(
        self,
        config: Cosmos3MixedPrecisionConfig,
        strategy: Cosmos3PrecisionStrategy,
    ) -> None:
        """Initialize empty inventories and request-lifecycle state."""
        self.config = config
        self.strategy = strategy

        # TODO: Make generation precision request-local before Cosmos3 permits
        # interleaved denoising requests on one transformer instance.
        self._generation_high_precision = False
        self._methods: list[Cosmos3MixedPrecisionLinearMethod] = []
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
        """Discover and wrap accepted linears under both Cosmos3 pathways."""
        components: dict[PrecisionPath, Iterable[torch.nn.Module]] = {
            "reasoner": transformer.language_model.layers,
            "generation": transformer.gen_layers,
        }
        for path, blocks in components.items():
            self._install_component(path, blocks)

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                f"Cosmos3 {self.config.format} mixed precision found no compatible linears under "
                f"{missing}; discovered counts={self.installed_counts}"
            )

        self.strategy.install_runtime(transformer, self)
        self._log_installation()

    def _install_component(
        self,
        path: PrecisionPath,
        blocks: Iterable[torch.nn.Module],
    ) -> None:
        """Wrap dynamically accepted linears in one reasoner or GEN component."""
        for block_index, block in enumerate(blocks):
            linear_index = 0
            for local_name, layer in block.named_modules():
                if not isinstance(layer, LinearBase):
                    continue
                base_method = getattr(layer, "quant_method", None)
                if not self.strategy.accepts(base_method):
                    continue

                module_name = getattr(layer, "prefix", None) or (f"{path}.{block_index}.{local_name}")
                method = Cosmos3MixedPrecisionLinearMethod(
                    base_method,
                    self,
                    module_name,
                    path,
                    block_index,
                    linear_index,
                )
                layer.quant_method = method
                self._methods.append(method)
                self.installed_counts[path] += 1
                linear_index += 1

    def _log_installation(self) -> None:
        """Log the resolved policy and discovered inventory once installed."""
        reasoner_label = (
            self.strategy.high_label if self.config.reasoner_policy == "high_precision" else self.strategy.base_label
        )
        logger.info(
            "Cosmos3 mixed precision installed: strategy=%s, reasoner=%s (%s), "
            "generation=first %d + last %d %s / middle %s, linears=%s",
            self.config.format,
            self.config.reasoner_policy,
            reasoner_label,
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
        """Bind one post-loaded linear and account for completed inventory."""
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
        """Finalize cache resources after proving every wrapper was bound."""
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

    def use_high_precision(self, path: PrecisionPath) -> bool:
        """Resolve reasoner policy or the current generation-step selection."""
        if path == "reasoner":
            return self.config.reasoner_policy == "high_precision"
        return self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        """Select precision once at the scheduler boundary and trace the step."""
        self._log_ready_once()
        self._generation_high_precision = self.config.use_high_precision(
            step_index,
            num_steps,
        )
        self.strategy.prepare_generation(self._generation_high_precision)
        label = self.strategy.high_label if self._generation_high_precision else self.strategy.base_label
        self._trace.append(label)

    def reset(self) -> None:
        """Finish the request trace and return strategy resources to idle state."""
        if self._trace:
            self.last_trace = tuple(self._trace)
            logger.debug(
                "Cosmos3 mixed-precision trace: strategy=%s, steps=%s",
                self.config.format,
                ",".join(self.last_trace),
            )
        self._trace.clear()
        self._generation_high_precision = False
        self.strategy.reset()
