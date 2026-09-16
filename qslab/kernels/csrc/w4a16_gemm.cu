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
    // nibble encodes q in [-8,7] as (q & 0xF); decode: (q ^ 8) - 8
    int q = static_cast<int>((word >> (4 * nib)) & 0xF);
    q = (q ^ 8) - 8;
    return static_cast<float>(q) * scale;
}

// y[m, n] = sum_k x[m, k] * dequant(qfp[n, k], scale[n, k/g])
// v5: same layout as v4, but the dot product uses a shared-memory LUT:
// per (row, group) the 16 dequantized values are precomputed once, then
// each weight costs 1 shared load instead of shift+sub+cvt+fma chain.
__global__ void w4a16_gemm_kernel(const uint32_t* __restrict__ qfp,
                                  const __half* __restrict__ scale,
                                  const __half* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int N, int K, int group_size) {
    const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int m = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;

    const int words_per_row = K / 8;
    const int words_per_group = group_size / 8;
    const uint32_t* row_qfp = qfp + (size_t)n * words_per_row;
    const __half* row_scale = scale + (size_t)n * (K / group_size);
    const __half* x_row = x + (size_t)m * K;

    // shared LUT: [warp][16] dequant values for the current group
    __shared__ float lut[8][16];               // up to 8 warps/block
    const int warp_in_block = threadIdx.x >> 5;
    float* my_lut = lut[warp_in_block];

    float acc = 0.0f;
    int cur_group = -1;
    const int vec_chunks = words_per_row / 4;
    for (int c = lane; c < vec_chunks; c += 32) {
        const int gidx = (c * 4) / words_per_group;
        if (gidx != cur_group) {
            cur_group = gidx;
            float s = __half2float(row_scale[gidx]);
            #pragma unroll
            for (int v = 0; v < 16; ++v) {
                int q = (v ^ 8) - 8;               // nibble -> [-8, 7]
                my_lut[v] = (float)q * s;
            }
        }
        __syncwarp();
        const uint4 pack = *reinterpret_cast<const uint4*>(row_qfp + c * 4);
        const int k0 = c * 32;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint32_t word = (&pack.x)[wi];
            const int kb = k0 + wi * 8;
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                int q = static_cast<int>((word >> (4 * nib)) & 0xF);
                acc += my_lut[q] * __half2float(x_row[kb + nib]);
            }
        }
    }
    // tail words (single-word groups may skip LUT update; reuse slow path)
    for (int w = vec_chunks * 4 + lane; w < words_per_row; w += 32) {
        uint32_t word = row_qfp[w];
        float s = __half2float(row_scale[w / words_per_group]);
        const int k0 = w * 8;
        #pragma unroll
        for (int nib = 0; nib < 8; ++nib)
            acc += dequant_nibble(word, nib, s) * __half2float(x_row[k0 + nib]);
    }
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
