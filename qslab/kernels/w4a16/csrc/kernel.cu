#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "w4a16.h"

namespace {

__device__ __forceinline__ float dequant_nibble(uint32_t word, int nib,
                                                float scale) {
    int q = static_cast<int>((word >> (4 * nib)) & 0xF);
    q = (q ^ 8) - 8;
    return static_cast<float>(q) * scale;
}

__global__ void w4a16_gemm_kernel(const uint32_t* __restrict__ qfp,
                                  const __half* __restrict__ scale,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ y,
                                  int m, int n, int k, int group_size) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int batch_row = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (row >= n) return;

    const int words_per_row = k / 8;
    const int words_per_group = group_size / 8;
    const uint32_t* row_qfp = qfp + static_cast<size_t>(row) * words_per_row;
    const __half* row_scale = scale + static_cast<size_t>(row) * (k / group_size);
    const __half* x_row = x + static_cast<size_t>(batch_row) * k;
    float acc = 0.0f;
    const int vec_chunks = words_per_row / 4;
    for (int c = lane; c < vec_chunks; c += 32) {
        const uint4 pack = *reinterpret_cast<const uint4*>(row_qfp + c * 4);
        const int k0 = c * 32;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t word = (&pack.x)[wi];
            const float s = __half2float(row_scale[(c * 4 + wi) / words_per_group]);
            const int kb = k0 + wi * 8;
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib)
                acc += dequant_nibble(word, nib, s) * __half2float(x_row[kb + nib]);
        }
    }
    for (int w = vec_chunks * 4 + lane; w < words_per_row; w += 32) {
        const uint32_t word = row_qfp[w];
        const float s = __half2float(row_scale[w / words_per_group]);
        const int k0 = w * 8;
        #pragma unroll
        for (int nib = 0; nib < 8; ++nib)
            acc += dequant_nibble(word, nib, s) * __half2float(x_row[k0 + nib]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    if (lane == 0) y[static_cast<size_t>(batch_row) * n + row] = __float2half(acc);
}

}  // namespace

void launch_w4a16_gemm(const void* qfp, const void* scale, const void* x,
                       void* y, int m, int n, int k, int group_size,
                       cudaStream_t stream) {
    dim3 grid((n + 3) / 4, m);
    w4a16_gemm_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const uint32_t*>(qfp), static_cast<const __half*>(scale),
        static_cast<const __half*>(x), static_cast<__half*>(y),
        m, n, k, group_size);
}
