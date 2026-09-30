"""Python entry point for the custom W4A16 CUDA operator."""
from __future__ import annotations

import torch

_extension = None


def w4a16_gemm(qfp: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
               group_size: int = 128) -> torch.Tensor:
    global _extension
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise TypeError("x and scale must be fp16")
    if qfp.dtype != torch.uint32:
        raise TypeError("qfp must be uint32")
    if _extension is None:
        from qslab.kernels.w4a16.build import load_extension
        _extension = load_extension()
    return _extension.w4a16_gemm(qfp, scale, x, group_size)
