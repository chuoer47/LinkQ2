// w4a16_gemm.cu — W4A16 (int4 weights, fp16 activations) GEMM/GEMV kernel.
//
// Decode-only regime: M is small (1..8). Memory-bandwidth bound, so the
// kernel's job is to stream packed int4 weights at full bandwidth, unpack
// in registers, and dot with fp16 activations using fp32 accumulation.
//
// Layout (docs/design-m1.md):
//   qfp   [N, K/8]  uint32 — 8 int4 per uint32, nibble i in bits [4i, 4i+4)
//   scale [N, K/g]  fp16   — group-wise scale (g=128), symmetric (zero=0)
//   x     [M, K]    fp16
//   y     [M, N]    fp16
//
// Strategy: one warp per output row n (for M=1). Threads split K; each
// thread processes consecutive uint32 words (8 weights each), dequantizes,
// dot-products with x, warp-reduces.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float dequant_nibble(uint32_t word, int nib,
                                                float scale) {
    int q = (word >> (4 * nib)) & 0xF;
    if (q >= 8) q -= 16;                       // sign-extend 4-bit
    return static_cast<float>(q) * scale;
}

// y[m, n] = sum_k x[m, k] * dequant(qfp[n, k], scale[n, k/g])
// Grid: (ceil(N/32), M)  Block: 32 threads (one warp per row)
__global__ void w4a16_gemm_kernel(const uint32_t* __restrict__ qfp,
                                  const __half* __restrict__ scale,
                                  const __half* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int N, int K, int group_size) {
    const int warp_id = blockIdx.x;            // one warp per row n
    const int n = warp_id;
    const int m = blockIdx.y;
    if (n >= N) return;

    const int lane = threadIdx.x;
    const int words_per_row = K / 8;
    const int words_per_group = group_size / 8;

    const uint32_t* row_qfp = qfp + (size_t)n * words_per_row;
    const __half* row_scale = scale + (size_t)n * (K / group_size);
    const __half* x_row = x + (size_t)m * K;

    float acc = 0.0f;
    // each lane strides over words: lane, lane+32, lane+64, ...
    for (int w = lane; w < words_per_row; w += 32) {
        uint32_t word = row_qfp[w];
        float s = __half2float(row_scale[w / words_per_group]);
        const int k0 = w * 8;
        #pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
            float wq = dequant_nibble(word, nib, s);
            acc += wq * __half2float(x_row[k0 + nib]);
        }
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(FULL_MASK, acc, off);
    if (lane == 0) y[(size_t)m * N + n] = acc;
}

// M>1 variant: same, but each warp also loops tokens (M small).
// For simplicity launch grid (N/32, M) as above — the same kernel handles
// any M via blockIdx.y. No separate kernel needed.

extern "C" void w4a16_gemm(const void* qfp, const void* scale, const void* x,
                           void* y, int M, int N, int K, int group_size,
                           cudaStream_t stream) {
    dim3 grid((N + 31) / 32, M);
    w4a16_gemm_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(qfp),
        reinterpret_cast<const __half*>(scale),
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<float*>(y), M, N, K, group_size);
}
