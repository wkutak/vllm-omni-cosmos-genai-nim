# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import mixed_precision
from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    Cosmos3MixedPrecisionRuntime,
    Cosmos3PrecisionLayerState,
    Fp8W8A8W8A16Strategy,
    _Cosmos3MixedPrecisionLinearMethod,
    _CpuW8A16BlockWeightProvider,
    _GpuW8A16BlockWeightProvider,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_config_parses_asymmetric_schedule_and_reasoner_policy() -> None:
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {
            "cosmos3_mixed_precision_format": "fp8",
            "cosmos3_mixed_precision_first_steps": 2,
            "cosmos3_mixed_precision_last_steps": 4,
            "cosmos3_mixed_precision_reasoner_policy": "base_precision",
            "cosmos3_mixed_precision_w8a16_cache": "all",
        }
    )

    assert config.enabled
    assert config.reasoner_policy == "base_precision"
    assert config.w8a16_cache == "all"
    selected = [index for index in range(10) if config.use_high_precision(index, 10)]
    assert selected == [0, 1, 6, 7, 8, 9]


@pytest.mark.parametrize(
    ("first_steps", "last_steps", "selected"),
    [
        (0, 2, [5, 6]),
        (2, 0, [0, 1]),
        (0, 0, []),
        (4, 4, list(range(7))),
    ],
)
def test_schedule_boundaries_and_overlap(
    first_steps: int,
    last_steps: int,
    selected: list[int],
) -> None:
    config = Cosmos3MixedPrecisionConfig(
        format="fp8",
        first_steps=first_steps,
        last_steps=last_steps,
    )
    assert [index for index in range(7) if config.use_high_precision(index, 7)] == selected


def test_one_step_engine_execution_uses_base_precision() -> None:
    config = Cosmos3MixedPrecisionConfig(format="fp8", first_steps=1, last_steps=1)
    assert not config.use_high_precision(0, 1)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"cosmos3_mixed_precision_format": "nvfp4"}, "must be one of"),
        ({"cosmos3_mixed_precision_first_steps": -1}, "non-negative"),
        ({"cosmos3_mixed_precision_reasoner_policy": "fp16"}, "must be one of"),
        ({"cosmos3_mixed_precision_w8a16_cache": "disk"}, "must be one of"),
    ],
)
def test_config_rejects_invalid_values(values: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Cosmos3MixedPrecisionConfig.from_additional_config(values)


@pytest.mark.parametrize("cache_mode", ["gpu_block", "cpu_block"])
def test_config_accepts_block_cache_modes(cache_mode: str) -> None:
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {"cosmos3_mixed_precision_w8a16_cache": cache_mode}
    )
    assert config.w8a16_cache == cache_mode


def test_config_defaults_to_bounded_gpu_block_cache() -> None:
    config = Cosmos3MixedPrecisionConfig.from_additional_config({})
    assert config.w8a16_cache == "gpu_block"


class _Layer(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size
        self.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(weight_scale, requires_grad=False),
        )


class _BaseMethod:
    def __init__(self) -> None:
        self.processed = False
        self.apply_calls = 0
        self.fp8_linear = object()

    def create_weights(self, *args, **kwargs) -> None:
        pass

    def process_weights_after_loading(self, layer) -> None:
        self.processed = True

    def apply(self, layer, x, bias=None):
        del layer, bias
        self.apply_calls += 1
        return torch.full(
            (*x.shape[:-1], 3),
            17,
            dtype=x.dtype,
            device=x.device,
        )


def _fp8_layer() -> _Layer:
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [99.0, 99.0, 99.0],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    return _Layer(
        weight,
        torch.tensor([0.5], dtype=torch.float32),
        input_size=2,
        output_size=3,
    )


def _runtime_and_method(
    *,
    path: str = "generation",
    reasoner_policy: str = "high_precision",
    cache_mode: str = "generation",
):
    config = Cosmos3MixedPrecisionConfig(
        format="fp8",
        first_steps=1,
        last_steps=1,
        reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
        w8a16_cache=cache_mode,  # type: ignore[arg-type]
    )
    strategy = Fp8W8A8W8A16Strategy(
        cache_mode=cache_mode,  # type: ignore[arg-type]
    )
    runtime = Cosmos3MixedPrecisionRuntime(config, strategy)
    runtime.installed_counts[path] = 1  # type: ignore[index]
    base = _BaseMethod()
    method = _Cosmos3MixedPrecisionLinearMethod(
        base,  # type: ignore[arg-type]
        runtime,
        f"{path}.linear",
        path,  # type: ignore[arg-type]
    )
    layer = _fp8_layer()
    method.process_weights_after_loading(layer)
    return runtime, base, method, layer


def test_generation_dispatches_w8a16_edges_and_base_w8a8_middle() -> None:
    runtime, base, method, layer = _runtime_and_method()
    x = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16)

    runtime.set_step(1, 3)
    middle = method.apply(layer, x)
    assert torch.equal(middle, torch.full((1, 3), 17, dtype=torch.bfloat16))
    assert base.apply_calls == 1

    runtime.set_step(0, 3)
    edge = method.apply(layer, x)
    expected_weight = layer.weight[:2, :3].to(torch.bfloat16) * 0.5
    expected = torch.nn.functional.linear(x, expected_weight.t())
    assert torch.equal(edge, expected)
    assert base.apply_calls == 1


def test_generation_cache_is_contiguous_reused_and_nonpersistent() -> None:
    runtime, _, method, layer = _runtime_and_method()
    assert method.state is not None
    cache_name = method.state.cached_weight_name
    assert cache_name is not None
    cached_weight = getattr(layer, cache_name)
    assert cached_weight.shape == (3, 2)
    assert cached_weight.dtype == torch.bfloat16
    assert cached_weight.is_contiguous()
    assert cache_name not in layer.state_dict()

    runtime.set_step(0, 3)
    x = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16)
    first = method.apply(layer, x)
    second = method.apply(layer, x)

    assert getattr(layer, cache_name).data_ptr() == cached_weight.data_ptr()
    assert torch.equal(first, second)


def test_generation_cache_avoids_redequantizing_mutated_fp8_weight() -> None:
    cached_runtime, _, cached_method, cached_layer = _runtime_and_method()
    uncached_runtime, _, uncached_method, uncached_layer = _runtime_and_method(
        cache_mode="none"
    )
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    cached_runtime.set_step(0, 3)
    uncached_runtime.set_step(0, 3)
    cached_before = cached_method.apply(cached_layer, x)
    uncached_before = uncached_method.apply(uncached_layer, x)

    cached_layer.weight.data.zero_()
    uncached_layer.weight.data.zero_()

    assert torch.equal(cached_method.apply(cached_layer, x), cached_before)
    assert not torch.equal(uncached_method.apply(uncached_layer, x), uncached_before)


@pytest.mark.parametrize(
    ("cache_mode", "path", "is_cached"),
    [
        ("none", "generation", False),
        ("generation", "generation", True),
        ("generation", "reasoner", False),
        ("all", "reasoner", True),
    ],
)
def test_cache_scope(cache_mode: str, path: str, is_cached: bool) -> None:
    _, _, method, _ = _runtime_and_method(
        cache_mode=cache_mode,
        path=path,
    )
    assert method.state is not None
    assert (method.state.cached_weight_name is not None) is is_cached


def test_cached_weight_dtype_must_match_activations() -> None:
    runtime, _, method, layer = _runtime_and_method()
    runtime.set_step(0, 3)

    with pytest.raises(TypeError, match="does not match activation dtype"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.float16))


@pytest.mark.parametrize(
    ("policy", "uses_base"),
    [("high_precision", False), ("base_precision", True)],
)
def test_reasoner_policy_is_independent_of_generation_step(
    policy: str,
    uses_base: bool,
) -> None:
    runtime, base, method, layer = _runtime_and_method(
        path="reasoner",
        reasoner_policy=policy,
    )
    runtime.set_step(1, 3)
    method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))
    assert (base.apply_calls == 1) is uses_base


def test_reset_clears_live_state_and_preserves_trace() -> None:
    runtime, _, _, _ = _runtime_and_method()
    for index in range(5):
        runtime.set_step(index, 5)
    runtime.reset()

    assert runtime.last_trace == ("W8A16", "W8A8", "W8A8", "W8A8", "W8A16")
    assert not runtime.use_high_precision("generation")


def test_marlin_is_rejected_before_base_processing(monkeypatch) -> None:
    class _FakeMarlin:
        pass

    monkeypatch.setattr(
        mixed_precision,
        "MarlinFP8ScaledMMLinearKernel",
        _FakeMarlin,
    )
    runtime, base, method, layer = _runtime_and_method()
    base.processed = False
    method.state = None
    base.fp8_linear = _FakeMarlin()

    with pytest.raises(ValueError, match="Marlin FP8"):
        method.process_weights_after_loading(layer)
    assert not base.processed


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (torch.empty(0, dtype=torch.float8_e4m3fn), "rank-2"),
        (torch.zeros(2, 3, dtype=torch.int32), "canonical float8"),
        (torch.zeros(1, 2).to(torch.float8_e4m3fn), "does not cover"),
    ],
)
def test_post_load_layout_validation_rejects_hidden_or_repacked_weight(
    weight: torch.Tensor,
    message: str,
) -> None:
    layer = _Layer(
        weight,
        torch.ones(1),
        input_size=2,
        output_size=3,
    )
    with pytest.raises((TypeError, ValueError), match=message):
        Fp8W8A8W8A16Strategy().bind(
            layer,
            module_name="generation.linear",
            path="generation",
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_weight_scale_must_be_finite_and_positive(scale: float) -> None:
    layer = _fp8_layer()
    layer.weight_scale.data.fill_(scale)
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        Fp8W8A8W8A16Strategy().bind(
            layer,
            module_name="generation.linear",
            path="generation",
        )


def test_smoothquant_parameter_is_rejected() -> None:
    layer = _fp8_layer()
    layer.register_parameter(
        "pre_quant_scale",
        torch.nn.Parameter(torch.ones(2), requires_grad=False),
    )
    with pytest.raises(ValueError, match="SmoothQuant"):
        Fp8W8A8W8A16Strategy().bind(
            layer,
            module_name="generation.linear",
            path="generation",
        )


def test_install_discovers_both_components_without_fixed_inventory(monkeypatch) -> None:
    class _FakeLinear(torch.nn.Module):
        def __init__(self, prefix: str) -> None:
            super().__init__()
            self.prefix = prefix
            self.quant_method = SimpleNamespace(name="fp8")

    monkeypatch.setattr(mixed_precision, "LinearBase", _FakeLinear)
    strategy = Fp8W8A8W8A16Strategy()
    monkeypatch.setattr(strategy, "accepts", lambda method: getattr(method, "name", None) == "fp8")
    runtime = Cosmos3MixedPrecisionRuntime(
        Cosmos3MixedPrecisionConfig(format="fp8"),
        strategy,
    )
    transformer = SimpleNamespace(
        language_model=SimpleNamespace(
            layers=torch.nn.Sequential(_FakeLinear("language_model.layers.0.q_proj"))
        ),
        gen_layers=torch.nn.Sequential(
            _FakeLinear("gen_layers.0.q_proj"),
            _FakeLinear("gen_layers.0.out_proj"),
        ),
    )

    runtime.install(transformer)

    assert runtime.installed_counts == {"reasoner": 1, "generation": 2}


def test_compiled_dispatch_has_bounded_base_and_high_variants() -> None:
    runtime, base, method, layer = _runtime_and_method()
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    compile_count = 0

    def apply_without_python_mutation(layer, value, bias=None):
        del layer, bias
        return torch.full(
            (*value.shape[:-1], 3),
            17,
            dtype=value.dtype,
            device=value.device,
        )

    # The numerical dispatch test checks base-method call counts. Avoid that
    # fake's Python counter here because it intentionally invalidates Dynamo's
    # guards and is not representative of ModelOptFp8LinearMethod.apply.
    base.apply = apply_without_python_mutation

    def backend(graph_module, example_inputs):
        del example_inputs
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled = torch.compile(
        lambda value: method.apply(layer, value),
        backend=backend,
    )
    runtime.set_step(1, 3)
    compiled(x)
    runtime.set_step(0, 3)
    compiled(x)
    runtime.set_step(1, 3)
    compiled(x)

    assert 1 <= compile_count <= 2


class _StagedLinearBlock(torch.nn.Module):
    def __init__(self, state: Cosmos3PrecisionLayerState) -> None:
        super().__init__()
        self.state = state

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        assert self.state.staged_weight is not None
        return torch.nn.functional.linear(value, self.state.staged_weight)


def _cuda_block_provider_fixture(provider_cls):
    states = [
        Cosmos3PrecisionLayerState(
            module_name=f"generation.{index}.linear",
            path="generation",
            input_size=2,
            output_size=2,
            block_index=index,
            linear_index=0,
        )
        for index in range(2)
    ]
    source_weights = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda"),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]], device="cuda"),
    ]
    layers = [
        _Layer(
            weight.to(torch.float8_e4m3fn),
            torch.tensor([scale], device="cuda"),
            input_size=2,
            output_size=2,
        ).cuda()
        for weight, scale in zip(source_weights, (1.0, 0.5), strict=True)
    ]
    blocks = [_StagedLinearBlock(state).cuda() for state in states]
    provider = provider_cls(torch.bfloat16)
    for state, layer in zip(states, layers, strict=True):
        provider.add(state, layer)
    provider.install(blocks, lambda: True)
    provider.initialize()
    return provider, states, blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "provider_cls",
    [_GpuW8A16BlockWeightProvider, _CpuW8A16BlockWeightProvider],
)
def test_block_provider_double_buffers_repeated_cfg_passes(provider_cls) -> None:
    provider, states, blocks = _cuda_block_provider_fixture(provider_cls)
    pointers = [state.staged_weight.data_ptr() for state in states]
    value = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16, device="cuda")

    for _ in range(2):
        actual = value
        for block in blocks:
            actual = block(actual)
        assert torch.equal(actual, value)

    torch.accelerator.synchronize()
    assert [state.staged_weight.data_ptr() for state in states] == pointers
    assert provider.device_bytes == 2 * 2 * 2 * 2
    if provider_cls is _CpuW8A16BlockWeightProvider:
        assert provider.host_bytes == 2 * 2 * 2 * 2
        assert all(block.is_pinned() for block in provider._host_blocks)
    else:
        assert provider.host_bytes == 0
    # Completing the last block wraps block zero for another CFG pass.
    assert provider._loaded_block_for_slot[0] == 0
    provider.reset()
    if provider_cls is _CpuW8A16BlockWeightProvider:
        assert not provider._host_blocks
        provider.preload_first()
        actual = value
        for block in blocks:
            actual = block(actual)
        assert torch.equal(actual, value)
        assert all(block.is_pinned() for block in provider._host_blocks)
        provider.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_block_provider_post_hook_is_exception_safe() -> None:
    class _RaisingBlock(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            del value
            raise RuntimeError("boom")

    state = Cosmos3PrecisionLayerState(
        module_name="generation.0.linear",
        path="generation",
        input_size=2,
        output_size=2,
    )
    layer = _Layer(
        torch.eye(2, device="cuda").to(torch.float8_e4m3fn),
        torch.ones(1, device="cuda"),
        input_size=2,
        output_size=2,
    ).cuda()
    block = _RaisingBlock().cuda()
    provider = _GpuW8A16BlockWeightProvider(torch.bfloat16)
    provider.add(state, layer)
    provider.install([block], lambda: True)
    provider.initialize()

    with pytest.raises(RuntimeError, match="boom"):
        block(torch.ones(1, 2, device="cuda", dtype=torch.bfloat16))
    provider.reset()
    assert provider._slot_was_used[0]
