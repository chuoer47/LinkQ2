"""torch extension binding for qslab kernels (w4a16 GEMM)."""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load

_mod = None


def _get_mod():
    global _mod
    if _mod is None:
        from pathlib import Path
        import os
        csrc = Path(__file__).resolve().parent / "csrc"
        # wheel has libcudart.so.12 but no linker symlink (leetcuda pitfall #5)
        nvlib = Path(os.environ["CONDA_PREFIX"]) / "lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
        if nvlib.exists():
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
            _mod = load(
                name="qslab_w4a16_ext",
                sources=[str(csrc / "w4a16_gemm_api.cu")],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=[f"-L{nvlib}", "-lcudart",
                               "-Xlinker", f"-rpath={nvlib}"],
                verbose=False,
            )
        else:
            _mod = load(name="qslab_w4a16_ext",
                        sources=[str(csrc / "w4a16_gemm_api.cu")],
                        extra_cuda_cflags=["-O3"], verbose=False)
    return _mod


def w4a16_gemm(qfp: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
               group_size: int = 128) -> torch.Tensor:
    """y = x @ dequant(qfp, scale)^T."""
    assert x.dtype == torch.float16 and scale.dtype == torch.float16
    assert qfp.dtype == torch.uint32
    mod = _get_mod()
    return mod.w4a16_gemm(qfp, scale, x, group_size)
