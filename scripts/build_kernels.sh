#!/usr/bin/env bash
# Build qslab kernels: nvcc 12.4 + conda-forge gcc-13, sm_89 (docs/02 §4).
# Usage: conda activate qslab && ./scripts/build_kernels.sh [--clean]
set -euo pipefail

cd "$(dirname "$0")/../kernels/csrc"
CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
NVCC="$CUDA_HOME/bin/nvcc"
CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

if [[ "${1:-}" == "--clean" ]]; then rm -f test_w4a16_gemm; fi

echo "== nvcc: $("$NVCC" --version | tail -1)"
# pip wheel headers live scattered under site-packages/nvidia (leetcuda pitfall #3)
NVDIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime"
# --cudart shared: no libcudart_static.a in conda/pip (pitfall #2); wheel has
# only libcudart.so.12 without the linker symlink (pitfall #5) -> -L plus rpath.
# libcudadevrt.a missing too (pitfall #2) -> empty stub archive trick.
NVLIB="$NVDIR/lib"
ln -sf "$NVLIB/libcudart.so.12" "$NVLIB/libcudart.so" 2>/dev/null || true
if [[ ! -f /tmp/libcudadevrt_stub.a ]]; then
    ar rcs /tmp/libcudadevrt_stub.a 2>/dev/null || \
        (echo "" > /tmp/devrt_empty.c && "$CCBIN" -c /tmp/devrt_empty.c -o /tmp/devrt_empty.o && ar rcs /tmp/libcudadevrt_stub.a /tmp/devrt_empty.o)
fi
"$NVCC" -O3 -std=c++17 -arch=sm_89 -ccbin "$CCBIN" -I"$NVDIR/include" --cudart shared \
    -L"$NVLIB" -L/tmp -Xlinker -rpath,"$NVLIB" \
    -Xlinker -L/tmp -Xlinker -lcudadevrt \
    test_w4a16_gemm.cu -o test_w4a16_gemm
echo "== built test_w4a16_gemm"
