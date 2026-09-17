// w4a16_gemm_api.cu — torch extension API around the raw kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float dequant_nibble(uint32_t word, int nib,
                                                float scale) {
    // nibble encodes q in [-8,7] as (q & 0xF); decode: q>=8 ? q-16 : q,
    // branch-free as (q ^ 8) - 8.
    int q = static_cast<int>((word >> (4 * nib)) & 0xF);
    q = (q ^ 8) - 8;
    return static_cast<float>(q) * scale;
}

__global__ void w4a16_gemm_kernel(const uint32_t* __restrict__ qfp,
                                  const __half* __restrict__ scale,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ y,
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
    float acc = 0.0f;
    const int vec_chunks = words_per_row / 4;
    for (int c = lane; c < vec_chunks; c += 32) {
        const uint4 pack = *reinterpret_cast<const uint4*>(row_qfp + c * 4);
        const int k0 = c * 32;
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            uint32_t word = (&pack.x)[wi];
            float s = __half2float(row_scale[(c * 4 + wi) / words_per_group]);
            const int kb = k0 + wi * 8;
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib)
                acc += dequant_nibble(word, nib, s) * __half2float(x_row[kb + nib]);
        }
    }
    for (int w = vec_chunks * 4 + lane; w < words_per_row; w += 32) {
        uint32_t word = row_qfp[w];
        float s = __half2float(row_scale[w / words_per_group]);
        const int k0 = w * 8;
        #pragma unroll
        for (int nib = 0; nib < 8; ++nib)
            acc += dequant_nibble(word, nib, s) * __half2float(x_row[k0 + nib]);
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(acc);
}

torch::Tensor w4a16_gemm(torch::Tensor qfp, torch::Tensor scale,
                         torch::Tensor x, int64_t group_size) {
    TORCH_CHECK(qfp.dtype() == torch::kUInt32, "qfp must be uint32");
    TORCH_CHECK(scale.dtype() == torch::kHalf, "scale must be fp16");
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_cuda() && qfp.is_cuda() && scale.is_cuda());
    const int N = qfp.size(0);
    const int K = x.size(-1);
    const int M = x.dim() == 2 ? x.size(0) : 1;
    auto x2 = x.dim() == 2 ? x : x.unsqueeze(0);
    auto y = torch::empty({M, N}, x.options());          // fp16 out: no cast op
    dim3 grid((N + 3) / 4, M);
    // Launch on the CURRENT stream, not the legacy default one: a CUDA
    // Graph capture happens on a side stream, and kernels put on the default
    // stream are not recorded into the graph — the replayed graph then skips
    // this GEMV entirely and produces zeros.
    auto stream = c10::cuda::getCurrentCUDAStream();
    w4a16_gemm_kernel<<<grid, 128, 0, stream.stream()>>>(
        reinterpret_cast<const uint32_t*>(qfp.data_ptr<uint32_t>()),
        reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x2.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()), M, N, K, (int)group_size);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm, "W4A16 GEMM (dequant in-kernel)");
}
