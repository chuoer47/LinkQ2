"""Project W4A16 operator adapter."""
from __future__ import annotations

import torch

from qslab.artifacts.schema import QuantizedTensor
from qslab.conversion.qslab_w4 import pack_quantized_w4


def prepare(weight: QuantizedTensor, act_scale: torch.Tensor | None,
            device: torch.device):
    packed, scale, _ = pack_quantized_w4(weight)
    packed, scale = packed.to(device), scale.to(device)
    act = None if act_scale is None else act_scale.to(device=device, dtype=torch.float16)
    from qslab.kernels.w4a16 import w4a16_gemm

    def run(x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if act is not None:
            x2 = x2 / act
        return w4a16_gemm(packed, scale, x2.contiguous(), weight.group_size)

    return run
