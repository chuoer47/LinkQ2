"""Backend independent, logical representation of a compressed linear weight."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class QuantizedTensor:
    """Integer values and quantization parameters before backend packing."""

    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    group_size: int
    bits: int = 4
    symmetric: bool = True
    qmin: int | None = None
    qmax: int | None = None

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("quantized linear values must have shape [out, in]")
        out_features, in_features = self.values.shape
        if self.bits != 4:
            raise ValueError(f"unsupported bit width: {self.bits}")
        qmin = (-8 if self.symmetric else 0) if self.qmin is None else self.qmin
        qmax = (7 if self.symmetric else 15) if self.qmax is None else self.qmax
        if qmax - qmin != 15 or (self.symmetric and (qmin, qmax) != (-8, 7)):
            raise ValueError("W4 quantization range must be [-8, 7] or [0, 15]")
        if self.group_size <= 0 or in_features % self.group_size:
            raise ValueError("input width must be divisible by group_size")
        if self.group_size % 8:
            raise ValueError("W4 group_size must be divisible by 8")
        shape = (out_features, in_features // self.group_size)
        if tuple(self.scale.shape) != shape or tuple(self.zero_point.shape) != shape:
            raise ValueError(f"scale and zero_point must have shape {shape}")
        if self.values.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32):
            raise TypeError("quantized values must use an integer dtype")
        if self.values.numel() and (self.values.min() < qmin or self.values.max() > qmax):
            raise ValueError(f"W4 values must be in [{qmin}, {qmax}]")
        if self.zero_point.numel() and (self.zero_point.min() < qmin or self.zero_point.max() > qmax):
            raise ValueError(f"W4 zero points must be in [{qmin}, {qmax}]")


@dataclass(slots=True)
class CompressionArtifact:
    """Portable in-memory compression result, independent of runtime layout."""

    model: dict[str, Any]
    algorithm: str
    weights: dict[str, QuantizedTensor]
    transforms: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    skipped_layers: list[str] = field(default_factory=list)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported artifact schema {self.schema_version}")
        for name, weight in self.weights.items():
            try:
                weight.validate()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid quantized tensor {name}: {exc}") from exc
