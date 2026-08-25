# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed activation precision for Cosmos3 denoising.

High-level flow
---------------

The solution separates *when* precision changes from *how* a quantization
format executes each precision. One model instance owns the policy and runtime
state; format strategies own checkpoint validation, arithmetic, and optional
weight caching.

```text
┌───────────────────────┐
│ 1. Request policy     │  first/last widths, reasoner policy, cache mode
└───────────┬───────────┘
            │
            v
┌───────────────────────┐
│ 2. Strategy selection │  choose arithmetic for the checkpoint format
└───────────┬───────────┘
            │
            v
┌───────────────────────┐
│ 3. Model installation │  discover, wrap, and validate eligible linears
└───────────┬───────────┘
            │
            v
┌───────────────────────┐
│ 4. Scheduler boundary │  choose one precision for this denoising step
└───────────┬───────────┘
            │
            v
┌───────────────────────┐
│ 5. Linear dispatch    │
└───────┬─────────┬─────┘
        │         │
        v         v
┌────────────┐  ┌───────────────┐
│ Base path  │  │ Precise path  │  strategy-specific execution
└──────┬─────┘  └───────┬───────┘
       └─────────┬──────┘
                 v
┌───────────────────────┐
│ 6. Request completion │  record trace and release temporary resources
└───────────────────────┘
```

Flow elements
-------------

1. **Request policy.** Configuration describes the high-precision regions at
   the start and end of denoising, the independent reasoner policy, and the
   weight-cache policy. It contains no quantization arithmetic.
2. **Strategy selection.** One explicit registry selects an implementation
   module for the loaded checkpoint format. Configuration validation and
   factory dispatch derive from the same registry.
3. **Model installation.** The runtime discovers eligible reasoner and
   generation linears from the instantiated model, retains their original
   quantization methods, and asks the strategy to validate their loaded tensor
   representation. Inventories and dimensions are never model-size constants.
4. **Scheduler boundary.** The pipeline sets precision once per denoising step,
   before any transformer calls for that step. Conditional, unconditional, and
   other CFG branches therefore share the same selection.
5. **Linear dispatch.** A lightweight wrapper sends each call to the strategy's
   base or precise path. The runtime does not know whether that means FP8,
   dense BF16, unpacking, or a future specialized kernel.
6. **Request completion.** A ``finally`` boundary records the executed
   precision trace, synchronizes outstanding staging work, and lets the
   strategy release request-scoped resources before output transfer.

Implemented strategy example
----------------------------

``Fp8W8A8W8A16Strategy`` is the first strategy using this architecture. It
keeps one serialized ModelOpt FP8 checkpoint. Middle denoising steps delegate
to the checkpoint's original W8A8 method; configured edge steps keep
activations in 16-bit precision and use the same canonical FP8 weights as
W8A16. The example demonstrates the strategy interface without making FP8
assumptions part of the common scheduler or runtime.

Caching strategies
------------------

``gpu_block`` (default)
    Keep two 16-bit device buffers. While the current generation block runs,
    reconstruct the next block from resident quantized weights on a side
    stream. This bounds additional device memory to two largest-block slots.
``cpu_block``
    Keep the same two device buffers, but stream the next block from pinned
    host memory. The host source is request-scoped and released before large
    decoded-output transfers.
``none``
    Keep no persistent 16-bit cache and reconstruct weights at each precise
    linear call. This is the memory floor and the fallback reference.
``generation``
    Materialize all generation-path precise weights on the device. This is the
    full-cache speed/reference endpoint at a substantially larger memory cost.
``all``
    Materialize precise weights for both reasoner and generation paths. This is
    the largest cache mode and is mainly useful as a reference.

Only the intended public configuration, runtime, strategy, and factory are
re-exported here. Staging providers, per-linear state, and dispatch wrappers
remain implementation details in their respective modules.
"""

from .config import (
    Cosmos3MixedPrecisionConfig,
    DenseWeightCacheMode,
    MixedPrecisionFormat,
    PrecisionPath,
    ReasonerPolicy,
    W8A16CacheMode,
)
from .registry import create_cosmos3_precision_strategy
from .runtime import Cosmos3MixedPrecisionRuntime
from .strategies.fp8 import Fp8W8A8W8A16Strategy
from .strategy import Cosmos3PrecisionStrategy

__all__ = [
    "Cosmos3MixedPrecisionConfig",
    "Cosmos3MixedPrecisionRuntime",
    "Cosmos3PrecisionStrategy",
    "DenseWeightCacheMode",
    "Fp8W8A8W8A16Strategy",
    "MixedPrecisionFormat",
    "PrecisionPath",
    "ReasonerPolicy",
    "W8A16CacheMode",
    "create_cosmos3_precision_strategy",
]
