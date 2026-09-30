"""Logical W4 quantization algorithms. Backend packing lives in conversion/."""
from __future__ import annotations

import torch

from qslab.artifacts.schema import QuantizedTensor


@torch.no_grad()
def rtn(weight: torch.Tensor, group_size: int = 128,
        activation_absmean: torch.Tensor | None = None,
        clip_ratio: float = 1.0) -> tuple[QuantizedTensor, None]:
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError("input width must be divisible by group_size")
    grouped = weight.float().reshape(out_features, -1, group_size)
    scale = (grouped.abs().amax(-1, keepdim=True) * clip_ratio / 7).clamp_min(1e-12)
    values = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
    zero = torch.zeros_like(scale.squeeze(-1))
    result = QuantizedTensor(values.reshape_as(weight).contiguous(),
                             scale.squeeze(-1).to(torch.float16), zero.to(torch.float16),
                             group_size)
    result.validate()
    return result, None


@torch.no_grad()
def rtn_clip(weight: torch.Tensor, group_size: int = 128,
             activation_absmean: torch.Tensor | None = None,
             n_grid: int = 40) -> tuple[QuantizedTensor, None]:
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError("input width must be divisible by group_size")
    grouped = weight.float().reshape(out_features, -1, group_size)
    maximum = grouped.abs().amax(-1, keepdim=True)
    best_error = torch.full_like(maximum.squeeze(-1), float("inf"))
    best_values = torch.zeros_like(grouped, dtype=torch.int8)
    best_scale = torch.zeros_like(maximum)
    for step in range(n_grid):
        ratio = 1.0 - step / n_grid * 0.6
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


@torch.no_grad()
def awq(weight: torch.Tensor, activation_absmean: torch.Tensor,
        group_size: int = 128, n_grid: int = 20,
        max_shrink: float = 1.0) -> tuple[QuantizedTensor, torch.Tensor]:
    """Search activation-aware input scaling, then quantize the scaled weights."""
    x = activation_absmean.float() + 1e-6
    out_features, in_features = weight.shape
    if in_features % group_size or x.numel() != in_features:
        raise ValueError("activation statistic shape or group_size does not match weight")
    original = weight.float()
    best_error = torch.full((in_features // group_size,), float("inf"), device=weight.device)
    best_scale = torch.ones_like(x)
    for step in range(n_grid):
        ratio = 1.0 - step * max_shrink / n_grid
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
    quantized, _ = rtn_clip(weight.float() * act_scale[None, :], group_size)
    return quantized, act_scale.to(torch.float16)


@torch.no_grad()
def nunchaku_awq(weight: torch.Tensor, activation_absmean: torch.Tensor,
                 group_size: int = 64, n_grid: int = 20,
                 max_shrink: float = 1.0
                 ) -> tuple[QuantizedTensor, torch.Tensor]:
    """Activation-aware affine W4 quantization for Nunchaku's AWQ GEMV."""
    out_features, in_features = weight.shape
    if group_size != 64:
        raise ValueError("Nunchaku AWQ GEMV requires group_size=64")
    if in_features % group_size or activation_absmean.numel() != in_features:
        raise ValueError("activation statistic shape or group_size does not match weight")
    original = weight.float()
    act = activation_absmean.float().clamp_min(1e-6)
    best_error = torch.full((), float("inf"), device=weight.device)
    best_values = best_scale = best_zero = best_act = None
    for step in range(n_grid):
        ratio = 1.0 - step * max_shrink / n_grid
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
        group_size, symmetric=False, qmin=0, qmax=15)
    result.validate()
    return result, best_act.to(torch.float16)
