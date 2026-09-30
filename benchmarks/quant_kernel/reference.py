"""Logical quantized-weight dequantization and high-precision reference."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qslab.artifacts.schema import QuantizedTensor


def dequantize(weight: QuantizedTensor, device: torch.device) -> torch.Tensor:
    out_features, in_features = weight.values.shape
    values = weight.values.to(device=device, dtype=torch.float32).reshape(
        out_features, in_features // weight.group_size, weight.group_size)
    scales = weight.scale.to(device=device, dtype=torch.float32).unsqueeze(-1)
    if weight.symmetric:
        restored = values * scales
    else:
        zeros = weight.zero_point.to(device=device, dtype=torch.float32).unsqueeze(-1)
        restored = (values - zeros) * scales
    return restored.reshape(out_features, in_features)


def reference_linear(x: torch.Tensor, weight: QuantizedTensor,
                     act_scale: torch.Tensor | None) -> torch.Tensor:
    x32 = x.to(torch.float32)
    if act_scale is not None:
        x32 = x32 / act_scale.to(device=x.device, dtype=torch.float32)
    w32 = dequantize(weight, x.device)
    return F.linear(x32, w32)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    denom = expected.float().norm().clamp_min(1e-12)
    actual_flat, expected_flat = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat, expected_flat, dim=0, eps=1e-12)
    return {
        "max_abs": float(delta.abs().max().item()),
        "rms": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(delta.norm().div(denom).item()),
        "cosine_similarity": float(cosine.item()),
    }
