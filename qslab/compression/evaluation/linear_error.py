"""Small reusable error measures for quantized linear layers."""
from __future__ import annotations

import torch


@torch.no_grad()
def relative_output_error(weight: torch.Tensor, quantized_weight: torch.Tensor,
                          inputs: torch.Tensor) -> torch.Tensor:
    reference = torch.nn.functional.linear(inputs, weight)
    compressed = torch.nn.functional.linear(inputs, quantized_weight)
    numerator = (compressed.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum().clamp_min(1e-12)
    return numerator / denominator
