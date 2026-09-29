"""Build the vendored Marlin kernel as a torch extension."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "third_party" / "marlin" / "marlin"

from torch.utils.cpp_extension import load


def get_marlin():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    site = Path(os.environ.get("CONDA_PREFIX", "")) / "lib/python3.11/site-packages/nvidia"
    # wheel headers scattered across components (leetcuda pitfall #3)
    incs = [site / n / "include" for n in
            ["cuda_runtime", "cusparse", "cublas", "cudnn", "cufft", "curand",
             "cusolver", "nccl", "nvtx"]]
    incs = [str(p) for p in incs if p.exists()]
    nvlib = site / "cuda_runtime/lib"
    extra = []
    if nvlib.exists():
        extra = [f"-L{nvlib}", "-lcudart", "-Xlinker", f"-rpath={nvlib}"]
    return load(
        name="qslab_marlin",
        sources=[str(SRC / "marlin_cuda.cpp"), str(SRC / "marlin_cuda_kernel.cu")],
        extra_cuda_cflags=["-O3"] + sum([["-I", p] for p in incs], []),
        extra_include_paths=incs,
        extra_ldflags=extra,
        verbose=False,
    )
