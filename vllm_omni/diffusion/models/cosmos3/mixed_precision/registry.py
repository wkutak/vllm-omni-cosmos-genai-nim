# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit registry for Cosmos3 mixed-precision strategy modules."""

from __future__ import annotations

import importlib
import pkgutil
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


def _discover_strategy_modules() -> dict[str, str]:
    """Discover strategy modules by filename without importing them."""
    package = importlib.import_module(".strategies", package=__package__)
    return {
        module.name: f".strategies.{module.name}"
        for module in pkgutil.iter_modules(package.__path__)
        if not module.ispkg and not module.name.startswith("_")
    }


# A strategy is added by placing ``<format>.py`` under ``strategies`` and
# exposing ``create_strategy`` from that module. Common configuration and
# factory dispatch both derive from the discovered filenames.
_STRATEGY_MODULES = _discover_strategy_modules()

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
