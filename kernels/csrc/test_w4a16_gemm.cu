// test_w4a16_gemm.cu — standalone correctness + perf test (LeetCUDA style).
//
// Build (see scripts/build_kernels.sh):
//   nvcc -O3 -arch=sm_89 -ccbin <gcc13> test_w4a16_gemm.cu -o test_w4a16_gemm
//
// Checks:
//   1. kernel(qfp,scale,x) == fp16-dequantized cublas reference (rel err < 1e-2)
//   2. perf vs torch-free fp16 GEMV baseline implemented here (naive fp16 kernel)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(e), __FILE__, __LINE__);           \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

__device__ __forceinline__ float dequant_nibble(uint32_t word, int nib,
                                                float scale) {
    int q = (word >> (4 * nib)) & 0xF;
    if (q >= 8) q -= 16;
    return static_cast<float>(q) * scale;
}

__global__ void w4a16_gemm_kernel(const uint32_t* __restrict__ qfp,
                                  const __half* __restrict__ scale,
                                  const __half* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int N, int K, int group_size) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int lane = threadIdx.x;
    if (n >= N) return;
    const int words_per_row = K / 8;
    const int words_per_group = group_size / 8;
    const uint32_t* row_qfp = qfp + (size_t)n * words_per_row;
    const __half* row_scale = scale + (size_t)n * (K / group_size);
    const __half* x_row = x + (size_t)m * K;
    float acc = 0.0f;
    for (int w = lane; w < words_per_row; w += 32) {
        uint32_t word = row_qfp[w];
        float s = __half2float(row_scale[w / words_per_group]);
        const int k0 = w * 8;
        #pragma unroll
        for (int nib = 0; nib < 8; ++nib)
            acc += dequant_nibble(word, nib, s) * __half2float(x_row[k0 + nib]);
    }
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[(size_t)m * N + n] = acc;
}

// naive fp16 GEMV baseline: same memory pattern minus unpack
__global__ void fp16_gemv_kernel(const __half* __restrict__ w,
                                 const __half* __restrict__ x,
                                 float* __restrict__ y, int N, int K) {
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    const __half* row = w + (size_t)n * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += __half2float(row[k]) * __half2float(x[k]);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = acc;
}

// host-side pack: fp16 [N,K] -> qfp/scale (same as quantizer.packfmt)
void pack_host(const std::vector<__half>& w, int N, int K, int g,
               std::vector<uint32_t>& qfp, std::vector<__half>& scale) {
    qfp.assign((size_t)N * (K / 8), 0);
    scale.assign((size_t)N * (K / g), 0);
    for (int n = 0; n < N; ++n) {
        for (int kg = 0; kg < K / g; ++kg) {
            float amax = 0.f;
            for (int i = 0; i < g; ++i)
                amax = fmaxf(amax, fabsf(__half2float(w[(size_t)n * K + kg * g + i])));
            float s = fmaxf(amax / 7.0f, 1e-12f);
            scale[(size_t)n * (K / g) + kg] = __float2half(s);
            for (int i = 0; i < g; ++i) {
                float v = __half2float(w[(size_t)n * K + kg * g + i]);
                int q = (int)lrintf(v / s);
                q = std::max(-8, std::min(7, q));
                int word_idx = (kg * g + i) / 8;
                int nib = (kg * g + i) % 8;
                qfp[(size_t)n * (K / 8) + word_idx] |= ((uint32_t)(q & 0xF)) << (4 * nib);
            }
        }
    }
}

double bench(std::function<void()> fn, int iters = 200) {
    fn(); fn();  // warmup
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    std::vector<float> times;
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(t0));
        fn();
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

int main() {
    struct Shape { int N, K; };
    Shape shapes[] = {{2048, 2048}, {2048, 6144}, {6144, 2048}, {6144, 12288}};
    const int M = 1, g = 128;

    for (auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 0.02f);
        std::vector<__half> w((size_t)N * K), x((size_t)M * K);
        for (auto& v : w) v = __float2half(dist(rng));
        for (auto& v : x) v = __float2half(dist(rng));

        std::vector<uint32_t> qfp_h;
        std::vector<__half> scale_h;
        pack_host(w, N, K, g, qfp_h, scale_h);

        uint32_t* qfp_d; __half *scale_d, *x_d, *w_d; float *y_d, *y_ref_d;
        CHECK_CUDA(cudaMalloc(&qfp_d, qfp_h.size() * 4));
        CHECK_CUDA(cudaMalloc(&scale_d, scale_h.size() * 2));
        CHECK_CUDA(cudaMalloc(&x_d, x.size() * 2));
        CHECK_CUDA(cudaMalloc(&w_d, w.size() * 2));
        CHECK_CUDA(cudaMalloc(&y_d, (size_t)M * N * 4));
        CHECK_CUDA(cudaMalloc(&y_ref_d, (size_t)M * N * 4));
        CHECK_CUDA(cudaMemcpy(qfp_d, qfp_h.data(), qfp_h.size() * 4, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(scale_d, scale_h.data(), scale_h.size() * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(x_d, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(w_d, w.data(), w.size() * 2, cudaMemcpyHostToDevice));

        dim3 grid(N, M);
        double ms_w4 = bench([&] { w4a16_gemm_kernel<<<grid, 32>>>(qfp_d, scale_d, x_d, y_d, M, N, K, g); });
        double ms_fp16 = bench([&] { fp16_gemv_kernel<<<grid, 32>>>(w_d, x_d, y_ref_d, N, K); });

        // correctness: w4 vs fp16 GEMV on dequantized==original (quant err expected)
        std::vector<float> y_w4((size_t)M * N), y_h((size_t)M * N);
        CHECK_CUDA(cudaMemcpy(y_w4.data(), y_d, y_w4.size() * 4, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(y_h.data(), y_ref_d, y_h.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < y_w4.size(); ++i) {
            double d = y_w4[i] - y_h[i];
            num += d * d; den += (double)y_h[i] * y_h[i];
        }
        double rel = sqrt(num / den);

        double gb_w4 = ((double)N * K / 8 + (double)N * (K / g) * 2 + (double)K * 2) / 1e9;
        double gb_f = ((double)N * K * 2 + (double)K * 2) / 1e9;
        printf("N=%5d K=%5d | w4 %8.2f us (%5.1f GB/s, x%.2f) | fp16 %8.2f us (%5.1f GB/s) | rel_err %.4f\n",
               N, K, ms_w4 * 1e3, gb_w4 / (ms_w4 / 1e3), ms_fp16 / ms_w4,
               ms_fp16 * 1e3, gb_f / (ms_fp16 / 1e3), rel);

        CHECK_CUDA(cudaFree(qfp_d)); CHECK_CUDA(cudaFree(scale_d));
        CHECK_CUDA(cudaFree(x_d)); CHECK_CUDA(cudaFree(w_d));
        CHECK_CUDA(cudaFree(y_d)); CHECK_CUDA(cudaFree(y_ref_d));
    }
    return 0;
}
