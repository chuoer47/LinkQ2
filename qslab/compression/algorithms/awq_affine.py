"""Activation-aware input scaling followed by affine W4 quantization."""
from dataclasses import dataclass

import torch

from qslab.artifacts.schema import QuantizedTensor
from .base import WeightQuantizer


@dataclass
class AWQAffine(WeightQuantizer):
    requires_calibration = True
    default_group_size = 64
    n_grid: int = 20
    max_shrink: float = 1.0

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]:
        """Activation-aware affine W4 quantization."""
        out_features, in_features = weight.shape
        if activation_absmean is None or in_features % group_size or activation_absmean.numel() != in_features:
            raise ValueError("activation statistic shape or group_size does not match weight")
        original = weight.float()
        act = activation_absmean.float().clamp_min(1e-6)
        best_error = torch.full((), float("inf"), device=weight.device)
        best_values = best_scale = best_zero = best_act = None
        for step in range(self.n_grid):
            ratio = 1.0 - step * self.max_shrink / self.n_grid
            candidate = act.pow(ratio)
            candidate = candidate / candidate.mean()
            grouped = (original * candidate[None, :]).reshape(out_features, -1, group_size)
            lo = grouped.amin(-1, keepdim=True)
            hi = grouped.amax(-1, keepdim=True)
            scale = ((hi - lo) / 15).clamp_min(1e-12)
            zero = torch.round(-lo / scale).clamp(0, 15)
            values = torch.round(grouped / scale + zero).clamp(0, 15)
            restored = (values - zero) * scale / candidate.reshape(1, -1, group_size)
            error = ((restored - original.reshape(out_features, -1, group_size))
                     .square() * act.reshape(1, -1, group_size)).sum()
            if error < best_error:
                best_error = error
                best_values = values.to(torch.uint8)
                best_scale = scale.squeeze(-1)
                best_zero = zero.squeeze(-1)
                best_act = candidate
        assert best_values is not None and best_scale is not None
        result = QuantizedTensor(
            best_values.reshape_as(weight).contiguous(),
            best_scale.to(torch.float16), best_zero.to(torch.float16),
            group_size, symmetric=False)
        result.validate()
        return result, best_act.to(torch.float16)
