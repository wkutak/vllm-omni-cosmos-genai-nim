# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Format-specific tensor capture and reference A16 materialization."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8LinearMethod,
    ModelOptFp8PcPtLinearMethod,
    ModelOptNvFp4LinearMethod,
)

_FP8_WEIGHT_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
)
_WEIGHT = "_cosmos3_precision_weight"
_WEIGHT_SCALE = "_cosmos3_precision_weight_scale"
_WEIGHT_GLOBAL_SCALE = "_cosmos3_precision_weight_global_scale"
_NVFP4_BLOCK_SIZE = 16
_NVFP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class Cosmos3PrecisionStrategy:
    """Quantization-format boundary behind the shared precision schedule."""

    def accepts(self, method: object | None) -> bool:
        """Return whether this strategy owns a native linear method."""
        raise NotImplementedError

    def snapshot_before_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        """Capture canonical tensors before the native backend may repack them."""
        raise NotImplementedError

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        """Materialize the captured weight as a dense BF16 matrix."""
        raise NotImplementedError

    def materialize_into(
        self,
        target: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Reference fill for a preallocated cache or staging view."""
        target.copy_(self.materialize(layer))

    def apply_high(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the reference dense A16 path."""
        input_size = int(layer.input_size_per_partition)
        output_size = int(layer.output_size_per_partition)
        if weight is None:
            weight = self.materialize(layer)
        expected_shape = (output_size, input_size)
        if tuple(weight.shape) != expected_shape:
            raise RuntimeError(f"Dense weight shape {tuple(weight.shape)} does not match {expected_shape}")
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[1] != input_size:
            raise ValueError(f"Expected activation width {input_size}, got {x_2d.shape[1]}")
        weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        output = F.linear(x_2d, weight, bias)
        return output.view(*x.shape[:-1], output_size)


class Fp8W8A8W8A16Strategy(Cosmos3PrecisionStrategy):
    """Use native ModelOpt W8A8 or reference dense W8A16."""

    def accepts(self, method: object | None) -> bool:
        return isinstance(
            method,
            (ModelOptFp8LinearMethod, ModelOptFp8PcPtLinearMethod),
        )

    def snapshot_before_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype not in _FP8_WEIGHT_DTYPES:
            dtype = getattr(weight, "dtype", None)
            raise TypeError(f"{module_name} requires a canonical FP8 weight, got {dtype}")
        if weight.ndim != 2:
            raise ValueError(f"{module_name} requires a rank-2 FP8 weight, got {tuple(weight.shape)}")
        if not isinstance(scale, torch.Tensor):
            raise ValueError(f"{module_name} has an FP8 weight but no weight_scale")
        if not _fp8_scale_is_supported(weight, scale):
            raise ValueError(
                f"{module_name} uses unsupported block-scaled FP8: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
            )
        expected = (
            int(layer.output_size_per_partition),
            int(layer.input_size_per_partition),
        )
        if tuple(weight.shape) != expected:
            raise ValueError(f"{module_name} expected FP8 weight shape {expected}, got {tuple(weight.shape)}")
        _validate_positive_finite_scale(scale, module_name)
        _register_snapshot(layer, _WEIGHT, weight)
        _register_snapshot(layer, _WEIGHT_SCALE, scale)

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        weight = _required_snapshot(layer, _WEIGHT, "FP8 weight")
        scale = _required_snapshot(layer, _WEIGHT_SCALE, "FP8 weight scale")
        values = weight.to(torch.float32)
        scale = scale.reshape(-1).to(device=values.device, dtype=torch.float32)
        if scale.numel() == 1:
            values = values * scale.reshape(())
        else:
            values = values * scale.reshape(-1, 1)
        return values.to(torch.bfloat16)


class Nvfp4W4A4W4A16Strategy(Cosmos3PrecisionStrategy):
    """Use native ModelOpt W4A4 or reference dense W4A16."""

    def accepts(self, method: object | None) -> bool:
        return isinstance(method, ModelOptNvFp4LinearMethod)

    def snapshot_before_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        packed = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        global_scale = _nvfp4_global_scale(layer)
        if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
            dtype = getattr(packed, "dtype", None)
            raise TypeError(f"{module_name} requires a canonical packed NVFP4 weight, got {dtype}")
        if not isinstance(scale, torch.Tensor) or global_scale is None:
            raise ValueError(f"{module_name} is missing ModelOpt NVFP4 scales")

        output_size = int(layer.output_size_per_partition)
        input_size = int(layer.input_size_per_partition)
        if input_size % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"{module_name} input size {input_size} is not divisible by {_NVFP4_BLOCK_SIZE}"
            )
        expected_weight = (output_size, input_size // 2)
        if tuple(packed.shape) != expected_weight:
            raise ValueError(
                f"{module_name} expected packed NVFP4 weight shape {expected_weight}, got {tuple(packed.shape)}"
            )
        expected_scale = (output_size, input_size // _NVFP4_BLOCK_SIZE)
        if scale.ndim != 2 or scale.shape[0] < expected_scale[0] or scale.shape[1] < expected_scale[1]:
            raise ValueError(
                f"{module_name} NVFP4 scale shape {tuple(scale.shape)} does not cover {expected_scale}"
            )

        _clamp_nvfp4_scales(layer)
        scale = layer.weight_scale[: expected_scale[0], : expected_scale[1]]
        global_scale = _nvfp4_global_scale(layer)
        if global_scale is None:
            raise RuntimeError(f"{module_name} lost its NVFP4 global scale during snapshot preparation")
        _validate_positive_finite_scale(scale, module_name)
        _validate_positive_finite_scale(global_scale, module_name)
        if global_scale.numel() != 1:
            raise ValueError(f"{module_name} requires one NVFP4 global scale")

        _register_snapshot(layer, _WEIGHT, packed)
        _register_snapshot(layer, _WEIGHT_SCALE, scale)
        _register_snapshot(layer, _WEIGHT_GLOBAL_SCALE, global_scale.reshape(1))

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        packed = _required_snapshot(layer, _WEIGHT, "NVFP4 weight")
        scale = _required_snapshot(layer, _WEIGHT_SCALE, "NVFP4 block scale")
        global_scale = _required_snapshot(layer, _WEIGHT_GLOBAL_SCALE, "NVFP4 global scale")
        return _dequantize_nvfp4_reference(packed, scale, global_scale[0])


def _fp8_scale_is_supported(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    if all(dim == 1 for dim in scale.shape):
        return True
    return bool(scale.shape) and scale.shape[0] == weight.shape[0] and all(dim == 1 for dim in scale.shape[1:])


def _dequantize_nvfp4_reference(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Unpack E2M1 values and apply unswizzled ModelOpt block scales."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1).long()
    lut = torch.tensor(
        [
            (-1.0 if code & 0x8 else 1.0) * _NVFP4_E2M1_VALUES[code & 0x7]
            for code in range(16)
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    values = lut[codes]
    block_scales = scale.to(device=packed.device, dtype=torch.float32).repeat_interleave(
        _NVFP4_BLOCK_SIZE,
        dim=1,
    )
    return (values * block_scales * global_scale.float()).to(torch.bfloat16)


def _register_snapshot(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    layer.register_buffer(name, value.detach().clone(), persistent=False)


def _required_snapshot(layer: torch.nn.Module, name: str, description: str) -> torch.Tensor:
    value = getattr(layer, name, None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Missing Cosmos3 precision-schedule {description} snapshot")
    return value


def _validate_positive_finite_scale(scale: torch.Tensor, module_name: str) -> None:
    values = scale.detach().float()
    if not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"{module_name} has a non-finite or non-positive weight scale")


def _nvfp4_global_scale(layer: torch.nn.Module) -> torch.Tensor | None:
    for name in ("weight_scale_2", "weight_global_scale"):
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _clamp_nvfp4_scales(layer: torch.nn.Module) -> None:
    if os.environ.get("VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    from vllm_omni.patch import _clamp_nvfp4_weight_scale_nans

    _clamp_nvfp4_weight_scale_nans(layer)
