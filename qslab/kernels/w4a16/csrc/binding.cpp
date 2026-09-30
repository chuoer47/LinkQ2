#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#include "w4a16.h"

torch::Tensor w4a16_gemm(torch::Tensor qfp, torch::Tensor scale,
                         torch::Tensor x, int64_t group_size) {
    TORCH_CHECK(qfp.dtype() == torch::kUInt32, "qfp must be uint32");
    TORCH_CHECK(scale.dtype() == torch::kHalf, "scale must be fp16");
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_cuda() && qfp.is_cuda() && scale.is_cuda(),
                "qfp, scale and x must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                "x must have shape [M, K] or [B, M, K]");

    const int n = qfp.size(0);
    const int k = x.size(-1);
    const int m = x.dim() == 2 ? x.size(0) : x.size(0) * x.size(1);
    auto x2 = x.reshape({m, k});
    auto y = torch::empty({m, n}, x.options());
    auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    launch_w4a16_gemm(qfp.data_ptr<uint32_t>(), scale.data_ptr<at::Half>(),
                      x2.data_ptr<at::Half>(), y.data_ptr<at::Half>(),
                      m, n, k, static_cast<int>(group_size), stream.stream());
    if (x.dim() == 3) {
        return y.reshape({x.size(0), x.size(1), n});
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm, "W4A16 GEMM (dequant in-kernel)");
}
