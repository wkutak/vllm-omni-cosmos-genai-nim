# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit registry for Cosmos3 mixed-precision strategy modules."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from .config import Cosmos3MixedPrecisionConfig
    from .strategy import Cosmos3PrecisionStrategy


class StrategyFactory(Protocol):
    """Construct one format strategy from common runtime configuration."""

    def __call__(
        self,
        config: Cosmos3MixedPrecisionConfig,
        *,
        activation_dtype: torch.dtype,
    ) -> Cosmos3PrecisionStrategy: ...


# Adding a format requires its strategy module plus one auditable registration
# here. Configuration validation and factory dispatch both derive from this map.
_STRATEGY_MODULES = {
    "fp8": ".strategies.fp8",
}

MIXED_PRECISION_FORMATS = frozenset({"none", *_STRATEGY_MODULES})


def create_cosmos3_precision_strategy(
    config: Cosmos3MixedPrecisionConfig,
    *,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> Cosmos3PrecisionStrategy:
    """Load and create the strategy registered for ``config.format``."""
    module_name = _STRATEGY_MODULES.get(config.format)
    if module_name is None:
        raise ValueError(
            f"No Cosmos3 precision strategy exists for format {config.format!r}; "
            f"registered formats={sorted(_STRATEGY_MODULES)}"
        )

    module = importlib.import_module(module_name, package=__package__)
    factory = cast(StrategyFactory | None, getattr(module, "create_strategy", None))
    if factory is None or not callable(factory):
        raise RuntimeError(f"Cosmos3 mixed-precision module {module.__name__!r} must expose a callable create_strategy")
    return factory(config, activation_dtype=activation_dtype)
