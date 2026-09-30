#pragma once

#include <cuda_runtime.h>

void launch_w4a16_gemm(const void* qfp, const void* scale, const void* x,
                       void* y, int m, int n, int k, int group_size,
                       cudaStream_t stream);
