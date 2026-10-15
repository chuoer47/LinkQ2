"""Offline quantization results, before any execution-specific packing."""
from dataclasses import dataclass, field

import torch


@dataclass
class QuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    group_size: int
    symmetric: bool = True

    def validate(self):
        n, k = self.values.shape
        if self.group_size <= 0 or k % self.group_size:
            raise ValueError("input width must be divisible by group_size")
        shape = (n, k // self.group_size)
        if self.scale.shape != shape or self.zero_point.shape != shape:
            raise ValueError(f"scale and zero_point must have shape {shape}")


@dataclass
class CompressionArtifact:
    weights: dict[str, QuantizedTensor]
    transforms: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def validate(self):
        for weight in self.weights.values():
            weight.validate()
