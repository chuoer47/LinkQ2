"""Thin adapter around the upstream Marlin Layer.pack/forward API."""
import torch
from torch import nn

from qslab.models.quant_linear import QuantLinear


class MarlinLinear(QuantLinear):
    @classmethod
    @torch.no_grad()
    def from_quantized(cls, values, scale, *, group_size, symmetric,
                       zero_point=None, input_scale=None, device, dtype):
        import marlin

        if dtype != torch.float16 or not symmetric:
            raise ValueError('upstream Marlin requires FP16 and symmetric W4 weights')
        n, k = values.shape
        layer = cls()
        layer.in_features, layer.out_features = k, n
        layer.operator = marlin.Layer(k, n, groupsize=group_size).to(device)
        # Official pack accepts a temporary fake-quantized FP16 Linear and its scales.
        scales = scale.to(device=device, dtype=torch.float16)
        dense = (values.to(device=device, dtype=torch.float32).reshape(n, -1, layer.operator.groupsize)
                 * scales.float().unsqueeze(-1)).reshape(n, k).half()
        linear = nn.Linear(k, n, bias=False, device='meta', dtype=torch.float16)
        linear.weight = nn.Parameter(dense, requires_grad=False)
        layer.operator.pack(linear, scales)
        layer.register_buffer('input_scale', None if input_scale is None else
                              input_scale.to(device=device, dtype=torch.float16))
        return layer

    def forward(self, x):
        if x.dtype != torch.float16:
            raise TypeError('upstream Marlin requires FP16 activations')
        if self.input_scale is not None:
            x = x / self.input_scale
        return self.operator(x.contiguous())
