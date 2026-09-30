"""Nunchaku AWQ GEMV CUDA extension adapter."""
from __future__ import annotations

import torch

from qslab.artifacts.schema import QuantizedTensor
from qslab.conversion.nunchaku_awq.format import pack_nunchaku_awq


def pack(weight: QuantizedTensor):
    if weight.symmetric or weight.group_size != 64:
        raise ValueError("Nunchaku AWQ GEMV requires affine W4 with group_size=64")
    return pack_nunchaku_awq(weight.values, weight.scale, weight.zero_point)


def prepare(weight: QuantizedTensor, act_scale: torch.Tensor | None,
            device: torch.device):
    if weight.symmetric:
        raise ValueError("Nunchaku AWQ backend requires affine unsigned W4")
    if weight.group_size != 64:
        raise ValueError("Nunchaku AWQ GEMV requires group_size=64")
    qweight, scales, zeros = pack(weight)
    qweight, scales, zeros = (qweight.to(device), scales.to(device), zeros.to(device))
    act = None if act_scale is None else act_scale.to(device=device, dtype=torch.float16)
    try:
        from nunchaku._C import ops
        gemv = ops.gemv_awq
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Nunchaku AWQ CUDA extension is unavailable") from exc
    n, k = weight.values.shape

    def run(x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, k)
        if act is not None:
            x2 = x2 / act
        chunks = [gemv(part.contiguous(), qweight, scales, zeros,
                       int(part.shape[0]), n, k, weight.group_size)
                  for part in x2.split(8, dim=0)]
        return torch.cat(chunks, dim=0)

    return run
