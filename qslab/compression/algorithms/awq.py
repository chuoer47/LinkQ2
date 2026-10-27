"""Activation-aware input scaling followed by symmetric W4 quantization."""
from dataclasses import dataclass

import torch

from qslab.compression.schema import QuantizedTensor
from .base import WeightQuantizer
from .rtn_clip import RTNClip


@dataclass
class AWQ(WeightQuantizer):
    requires_calibration = True
    n_grid: int = 20
    max_shrink: float = 1.0

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]:
        """Search activation-aware input scaling, then quantize the scaled weights."""
        out_features, in_features = weight.shape
        if activation_absmean is None or in_features % group_size or activation_absmean.numel() != in_features:
            raise ValueError("activation statistic shape or group_size does not match weight")
        x = activation_absmean.float() + 1e-6
        original = weight.float()
        best_error = torch.full((in_features // group_size,), float("inf"), device=weight.device)
        best_scale = torch.ones_like(x)
        for step in range(self.n_grid):
            ratio = 1.0 - step * self.max_shrink / self.n_grid
            candidate = x.pow(ratio)
            candidate = candidate / candidate.mean()
            scaled = original * candidate[None, :]
            grouped = scaled.reshape(out_features, -1, group_size)
            scale = (grouped.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12)
            values = torch.round(grouped / scale).clamp(-8, 7)
            restored_error = ((values * scale).reshape_as(original) / candidate - original) * x
            error = restored_error.square().reshape(out_features, -1, group_size).sum((0, 2))
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            groups = candidate.reshape(-1, group_size)
            best_scale = torch.where(better[:, None], groups, best_scale.reshape(-1, group_size))
            best_scale = best_scale.reshape(-1)
        act_scale = best_scale
        quantized, _ = RTNClip().quantize(weight.float() * act_scale[None, :], group_size=group_size)
        return quantized, act_scale.to(torch.float16)
