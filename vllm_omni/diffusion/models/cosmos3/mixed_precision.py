# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-denoising-step activation precision for Cosmos3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
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

logger = init_logger(__name__)

MixedPrecisionFormat = Literal["none", "fp8"]
ReasonerPolicy = Literal["high_precision", "base_precision"]
PrecisionPath = Literal["reasoner", "generation"]

_FORMATS = {"none", "fp8"}
_REASONER_POLICIES = {"high_precision", "base_precision"}


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Common precision policy, independent of quantization arithmetic."""

    format: MixedPrecisionFormat = "none"
    first_steps: int = 3
    last_steps: int = 3
    reasoner_policy: ReasonerPolicy = "high_precision"

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
        return cls(
            format=precision_format,  # type: ignore[arg-type]
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
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


@dataclass(frozen=True)
class Cosmos3PrecisionLayerState:
    """Validated logical dimensions for one strategy-owned linear."""

    module_name: str
    path: PrecisionPath
    input_size: int
    output_size: int


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

        return Cosmos3PrecisionLayerState(
            module_name=module_name,
            path=path,
            input_size=input_size,
            output_size=output_size,
        )

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
    ) -> None:
        self.base_method = base_method
        self.runtime = runtime
        self.module_name = module_name
        self.path = path
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

    def install(self, transformer: torch.nn.Module) -> None:
        components = {
            "reasoner": transformer.language_model.layers,
            "generation": transformer.gen_layers,
        }
        for path, component in components.items():
            for local_name, layer in component.named_modules():
                if not isinstance(layer, LinearBase):
                    continue
                base_method = getattr(layer, "quant_method", None)
                if not self.strategy.accepts(base_method):
                    continue
                module_name = getattr(layer, "prefix", None) or f"{path}.{local_name}"
                method = _Cosmos3MixedPrecisionLinearMethod(
                    base_method,
                    self,
                    module_name,
                    path,  # type: ignore[arg-type]
                )
                layer.quant_method = method
                self._methods.append(method)
                self.installed_counts[path] += 1  # type: ignore[index]

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                "Cosmos3 mixed precision found no compatible serialized ModelOpt FP8 linears under "
                f"{missing}; discovered counts={self.installed_counts}"
            )

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
    ) -> Cosmos3PrecisionLayerState:
        return self.strategy.bind(layer, module_name=module_name, path=path)

    def use_high_precision(self, path: PrecisionPath) -> bool:
        if path == "reasoner":
            return self.config.reasoner_policy == "high_precision"
        return self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        self._generation_high_precision = self.config.use_high_precision(step_index, num_steps)
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


def create_cosmos3_precision_strategy(
    config: Cosmos3MixedPrecisionConfig,
) -> Cosmos3PrecisionStrategy:
    if config.format == "fp8":
        return Fp8W8A8W8A16Strategy()
    raise ValueError(f"No Cosmos3 precision strategy exists for format {config.format!r}")


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value
