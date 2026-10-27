"""Round-to-nearest symmetric W4 quantization."""
from dataclasses import dataclass

import torch

from qslab.compression.schema import QuantizedTensor
from .base import WeightQuantizer


@dataclass
class RTN(WeightQuantizer):
    clip_ratio: float = 1.0

    @torch.no_grad()
    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]:
        out_features, in_features = weight.shape
        if in_features % group_size:
            raise ValueError("input width must be divisible by group_size")
        grouped = weight.float().reshape(out_features, -1, group_size)
        scale = (grouped.abs().amax(-1, keepdim=True) * self.clip_ratio / 7).clamp_min(1e-12)
        values = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
        zero = torch.zeros_like(scale.squeeze(-1))
        result = QuantizedTensor(values.reshape_as(weight).contiguous(),
                                 scale.squeeze(-1).to(torch.float16), zero.to(torch.float16),
                                 group_size)
        result.validate()
        return result, None
