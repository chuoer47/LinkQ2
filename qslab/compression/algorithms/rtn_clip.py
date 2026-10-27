"""Symmetric W4 with per-group clipping search."""
from dataclasses import dataclass

import torch

from qslab.compression.schema import QuantizedTensor
from .base import WeightQuantizer


@dataclass
class RTNClip(WeightQuantizer):
    n_grid: int = 40

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]:
        out_features, in_features = weight.shape
        if in_features % group_size:
            raise ValueError("input width must be divisible by group_size")
        grouped = weight.float().reshape(out_features, -1, group_size)
        maximum = grouped.abs().amax(-1, keepdim=True)
        best_error = torch.full_like(maximum.squeeze(-1), float("inf"))
        best_values = torch.zeros_like(grouped, dtype=torch.int8)
        best_scale = torch.zeros_like(maximum)
        for step in range(self.n_grid):
            ratio = 1.0 - step / self.n_grid * 0.6
            scale = (maximum * ratio / 7).clamp_min(1e-12)
            values = torch.round(grouped / scale).clamp(-8, 7)
            error = ((values * scale - grouped) ** 2).sum(-1)
            keep = error < best_error
            best_error = torch.where(keep, error, best_error)
            best_values = torch.where(keep.unsqueeze(-1), values.to(torch.int8), best_values)
            best_scale = torch.where(keep.unsqueeze(-1), scale, best_scale)
        zero = torch.zeros_like(best_scale.squeeze(-1))
        result = QuantizedTensor(best_values.reshape_as(weight).contiguous(),
                                 best_scale.squeeze(-1).to(torch.float16),
                                 zero.to(torch.float16), group_size)
        result.validate()
        return result, None
