# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    Cosmos3MixedPrecisionRuntime,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision.cache import (
    Cosmos3BlockWeightStager,
    Cosmos3PrecisionLayerState,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision import runtime as runtime_impl
from vllm_omni.diffusion.models.cosmos3.mixed_precision.runtime import (
    Cosmos3MixedPrecisionLinearMethod,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision.strategy import (
    Fp8W8A8W8A16Strategy,
    Nvfp4W4A4W4A16Strategy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_config_parses_nested_schedule_and_reasoner_policy() -> None:
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {
            "cosmos3_mixed_precision": {
                "first_steps": 2,
                "last_steps": 4,
                "reasoner": "native",
                "cache": "full",
            }
        }
    )

    assert config is not None
    assert config.reasoner == "native"
    assert config.cache == "full"
    assert [index for index in range(10) if config.use_high_precision(index, 10)] == [0, 1, 6, 7, 8, 9]


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
        first_steps=first_steps,
        last_steps=last_steps,
    )
    assert [index for index in range(7) if config.use_high_precision(index, 7)] == selected


def test_one_step_request_honors_boundary_precision() -> None:
    config = Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1)
    assert config.use_high_precision(0, 1)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"cosmos3_mixed_precision": True}, "must be a mapping"),
        ({"cosmos3_mixed_precision": {"first_steps": -1}}, "non-negative"),
        ({"cosmos3_mixed_precision": {"reasoner": "fp16"}}, "must be one of"),
        ({"cosmos3_mixed_precision": {"cache": "disk"}}, "must be one of"),
        ({"cosmos3_mixed_precision": {"unknown": 1}}, "Unknown"),
    ],
)
def test_config_rejects_invalid_values(values: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Cosmos3MixedPrecisionConfig.from_additional_config(values)


def test_config_presence_enables_defaults() -> None:
    assert Cosmos3MixedPrecisionConfig.from_additional_config({}) is None
    config = Cosmos3MixedPrecisionConfig.from_additional_config(
        {"cosmos3_mixed_precision": {}}
    )
    assert config == Cosmos3MixedPrecisionConfig()


class _Layer(torch.nn.Module):
    def __init__(self, output_size: int, input_size: int) -> None:
        super().__init__()
        self.output_size_per_partition = output_size
        self.input_size_per_partition = input_size


class _BaseMethod:
    def __init__(self, *, mutate: bool = False) -> None:
        self.mutate = mutate
        self.processed = False
        self.apply_calls = 0

    def create_weights(self, *args, **kwargs) -> None:
        pass

    def process_weights_after_loading(self, layer) -> None:
        self.processed = True
        if self.mutate:
            layer.weight = torch.zeros(1)
            layer.weight_scale = torch.zeros(1)

    def apply(self, layer, x, bias=None):
        del layer, bias
        self.apply_calls += 1
        return torch.full(
            (*x.shape[:-1], 2),
            17,
            dtype=x.dtype,
            device=x.device,
        )


def _fp8_layer(output_size: int = 2, input_size: int = 4) -> _Layer:
    layer = _Layer(output_size, input_size)
    layer.weight = torch.ones(output_size, input_size, dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.tensor([0.5], dtype=torch.float32)
    return layer


def _nvfp4_layer(output_size: int = 2, input_size: int = 16) -> _Layer:
    layer = _Layer(output_size, input_size)
    layer.weight = torch.full((output_size, input_size // 2), 0x22, dtype=torch.uint8)
    layer.weight_scale = torch.ones(output_size, input_size // 16, dtype=torch.float8_e4m3fn)
    layer.weight_scale_2 = torch.ones(1)
    return layer


def _runtime_and_method(
    strategy,
    layer: _Layer,
    *,
    path: str = "generation",
    reasoner: str = "native",
    mutate: bool = False,
    cache: str = "none",
):
    config = Cosmos3MixedPrecisionConfig(
        first_steps=1,
        last_steps=1,
        reasoner=reasoner,
        cache=cache,
    )
    runtime = Cosmos3MixedPrecisionRuntime(config)
    base = _BaseMethod(mutate=mutate)
    method = Cosmos3MixedPrecisionLinearMethod(
        base,
        strategy,
        runtime,
        f"{path}.linear",
        path,
        0,
        0,
    )
    method.process_weights_after_loading(layer)
    return runtime, base, method


@pytest.mark.parametrize(
    ("strategy", "layer"),
    [
        (Fp8W8A8W8A16Strategy(), _fp8_layer()),
        (Nvfp4W4A4W4A16Strategy(), _nvfp4_layer()),
    ],
)
def test_generation_dispatches_native_middle_and_a16_edges(strategy, layer) -> None:
    runtime, base, method = _runtime_and_method(strategy, layer)
    x = torch.ones(1, layer.input_size_per_partition, dtype=torch.bfloat16)

    runtime.set_step(1, 3)
    assert torch.equal(method.apply(layer, x), torch.full((1, 2), 17, dtype=x.dtype))
    runtime.set_step(0, 3)
    assert torch.equal(
        method.apply(layer, x),
        torch.nn.functional.linear(x, strategy.materialize(layer)),
    )
    assert base.apply_calls == 1


@pytest.mark.parametrize(
    ("strategy", "layer"),
    [
        (Fp8W8A8W8A16Strategy(), _fp8_layer()),
        (Nvfp4W4A4W4A16Strategy(), _nvfp4_layer()),
    ],
)
def test_full_cache_reuses_nonpersistent_dense_weight(strategy, layer) -> None:
    runtime, _, method = _runtime_and_method(strategy, layer, cache="full")
    assert method.state is not None
    dense_weight = method.state.dense_weight
    assert dense_weight is not None
    assert "_cosmos3_dense_a16_weight" not in layer.state_dict()

    runtime.set_step(0, 3)
    x = torch.ones(1, layer.input_size_per_partition, dtype=torch.bfloat16)
    first = method.apply(layer, x)
    second = method.apply(layer, x)
    assert torch.equal(first, second)
    assert method.state.dense_weight.data_ptr() == dense_weight.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_block_cache_double_buffers_generation_blocks() -> None:
    class _Block(torch.nn.Module):
        def __init__(self, state: Cosmos3PrecisionLayerState) -> None:
            super().__init__()
            self.state = state

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert self.state.dense_weight is not None
            return torch.nn.functional.linear(value, self.state.dense_weight)

    strategy = Fp8W8A8W8A16Strategy()
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
    layers = []
    blocks = []
    stager = Cosmos3BlockWeightStager(torch.bfloat16)
    for state in states:
        layer = _fp8_layer(output_size=2, input_size=2)
        layer.weight = layer.weight.cuda()
        layer.weight_scale = layer.weight_scale.cuda()
        strategy.snapshot_before_processing(layer, state.module_name)
        layers.append(layer)
        blocks.append(_Block(state).cuda())
        stager.add(state, layer, strategy)
    stager.install(blocks, lambda: True)
    stager.initialize()

    pointers = [state.dense_weight.data_ptr() for state in states]
    value = torch.ones(1, 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):
        output = value
        for block in blocks:
            output = block(output)
        assert torch.equal(output, value)
    assert [state.dense_weight.data_ptr() for state in states] == pointers
    stager.reset()


@pytest.mark.parametrize(
    ("strategy", "layer"),
    [
        (Fp8W8A8W8A16Strategy(), _fp8_layer()),
        (Nvfp4W4A4W4A16Strategy(), _nvfp4_layer()),
    ],
)
def test_snapshot_precedes_native_backend_repacking(strategy, layer) -> None:
    original = layer.weight.clone()
    runtime, base, method = _runtime_and_method(strategy, layer, mutate=True)
    assert base.processed
    assert torch.equal(
        layer._cosmos3_precision_weight.view(torch.uint8),
        original.view(torch.uint8),
    )

    runtime.set_step(0, 3)
    output = method.apply(
        layer,
        torch.ones(1, layer.input_size_per_partition, dtype=torch.bfloat16),
    )
    assert output.shape == (1, layer.output_size_per_partition)


def test_snapshots_are_nonpersistent_buffers() -> None:
    layer = _fp8_layer()
    Fp8W8A8W8A16Strategy().snapshot_before_processing(layer, "gen.linear")
    assert "_cosmos3_precision_weight" in dict(layer.named_buffers())
    assert "_cosmos3_precision_weight" not in layer.state_dict()
    assert "_cosmos3_precision_weight_scale" not in layer.state_dict()


@pytest.mark.parametrize("scale_shape", [(2, 1, 4, 1), (1, 1, 8, 1)])
def test_block_scaled_fp8_is_rejected(scale_shape: tuple[int, ...]) -> None:
    layer = _fp8_layer(output_size=8, input_size=16)
    layer.weight_scale = torch.ones(*scale_shape)
    with pytest.raises(ValueError, match="block-scaled FP8"):
        Fp8W8A8W8A16Strategy().snapshot_before_processing(layer, "gen.linear")


def test_fp8_reference_materialization_supports_per_row_scales() -> None:
    layer = _fp8_layer(output_size=3, input_size=2)
    layer.weight_scale = torch.tensor([[1.0], [2.0], [3.0]])
    strategy = Fp8W8A8W8A16Strategy()
    strategy.snapshot_before_processing(layer, "gen.linear")
    output = strategy.materialize(layer)
    assert output[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_nvfp4_reference_materialization_unpacks_e2m1() -> None:
    layer = _nvfp4_layer(output_size=1)
    layer.weight[0, 0] = 0x21
    layer.weight_scale.fill_(2.0)
    layer.weight_scale_2.fill_(0.5)
    strategy = Nvfp4W4A4W4A16Strategy()
    strategy.snapshot_before_processing(layer, "gen.linear")
    output = strategy.materialize(layer)
    assert output.shape == (1, 16)
    assert output[0, :4].tolist() == [0.5, 1.0, 1.0, 1.0]


def test_nvfp4_rejects_fused_global_scales() -> None:
    layer = _nvfp4_layer(output_size=3)
    layer.weight_scale_2 = torch.tensor([1.0, 2.0])
    strategy = Nvfp4W4A4W4A16Strategy()
    with pytest.raises(ValueError, match="one NVFP4 global scale"):
        strategy.snapshot_before_processing(layer, "gen.qkv")


@pytest.mark.parametrize(
    ("policy", "uses_native"),
    [("a16", False), ("native", True)],
)
def test_reasoner_policy_is_independent_of_generation_step(
    policy: str,
    uses_native: bool,
) -> None:
    layer = _fp8_layer()
    runtime, base, method = _runtime_and_method(
        Fp8W8A8W8A16Strategy(),
        layer,
        path="reasoner",
        reasoner=policy,
    )
    runtime.set_step(1, 3)
    method.apply(layer, torch.ones(1, 4, dtype=torch.bfloat16))
    assert (base.apply_calls == 1) is uses_native


def test_reset_clears_generation_state() -> None:
    runtime = Cosmos3MixedPrecisionRuntime(
        Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1)
    )
    runtime.set_step(0, 5)
    assert runtime.use_high_precision("generation")
    runtime.reset()
    assert not runtime.use_high_precision("generation")


def test_install_discovers_generation_and_opt_in_reasoner(monkeypatch) -> None:
    class _FakeLinear(torch.nn.Module):
        def __init__(self, prefix: str) -> None:
            super().__init__()
            self.prefix = prefix
            self.input_size_per_partition = 4
            self.output_size_per_partition = 2
            self.quant_method = SimpleNamespace(name="fp8")

    monkeypatch.setattr(runtime_impl, "LinearBase", _FakeLinear)
    strategy = Fp8W8A8W8A16Strategy()
    monkeypatch.setattr(strategy, "accepts", lambda method: getattr(method, "name", None) == "fp8")
    monkeypatch.setattr(runtime_impl, "_STRATEGIES", (strategy,))
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="a16"))
    reasoner = _FakeLinear("reasoner.q_proj")
    generation = [
        _FakeLinear("generation.q_proj"),
        _FakeLinear("generation.out_proj"),
    ]
    transformer = SimpleNamespace(
        language_model=SimpleNamespace(layers=torch.nn.Sequential(reasoner)),
        gen_layers=torch.nn.Sequential(*generation),
    )
    runtime.install(transformer)
    assert isinstance(reasoner.quant_method, Cosmos3MixedPrecisionLinearMethod)
    assert all(isinstance(layer.quant_method, Cosmos3MixedPrecisionLinearMethod) for layer in generation)


def test_pipeline_helpers_forward_and_reset_schedule() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    calls = []
    pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
    pipeline.transformer = SimpleNamespace(
        set_mixed_precision_step=lambda step, count: calls.append((step, count)),
        reset_mixed_precision=lambda: calls.append("reset"),
    )
    pipeline._set_mixed_precision_step(2, 7)
    pipeline._reset_mixed_precision()
    assert calls == [(2, 7), "reset"]
