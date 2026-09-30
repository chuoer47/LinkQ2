"""Load the qslab W4A16 PyTorch CUDA extension."""
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


def load_extension():
    source_dir = Path(__file__).resolve().parent / "csrc"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    nvlib = (Path(os.environ["CONDA_PREFIX"])
             / "lib/python3.11/site-packages/nvidia/cuda_runtime/lib")
    extra_ldflags = []
    if nvlib.exists():
        extra_ldflags = [f"-L{nvlib}", "-lcudart", "-Xlinker", f"-rpath={nvlib}"]
    return load(
        name="qslab_w4a16_ext",
        sources=[str(source_dir / "binding.cpp"), str(source_dir / "kernel.cu")],
        extra_include_paths=[str(source_dir)],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
        verbose=False,
    )
