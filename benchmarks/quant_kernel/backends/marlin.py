"""Marlin operator adapter from qslab logical W4 weights."""
from __future__ import annotations

import torch

from qslab.artifacts.schema import QuantizedTensor
from qslab.conversion.qslab_w4 import pack_quantized_w4
from qslab.kernels.marlin.ops import marlin_gemm, pack_v1_to_marlin


def prepare(weight: QuantizedTensor, act_scale: torch.Tensor | None,
            device: torch.device):
    packed, scale, _ = pack_quantized_w4(weight)
    packed, scale = packed.to(device), scale.to(device)
    b, s = pack_v1_to_marlin(packed, scale, weight.values.shape[1],
                             weight.group_size)
    b, s = b.to(device), s.to(device)
    n = weight.values.shape[0]
    workspace = torch.zeros(n // 128 * 16, dtype=torch.int32, device=device)
    act = None if act_scale is None else act_scale.to(device=device, dtype=torch.float16)

    def run(x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if act is not None:
            x2 = x2 / act
        return marlin_gemm(x2.contiguous(), b, s, workspace)

    return run
