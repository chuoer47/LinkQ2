"""GPU tests: kernel numerics (need a CUDA device)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


class TestW4KernelNumerics:
    """The v1 W4A16 kernel must match a dequantized matmul exactly."""

    def test_v1_kernel_matches_dequant_matmul(self):
        from qslab.kernels.w4a16 import w4a16_gemm
        from qslab.compression.algorithms.w4 import rtn
        from qslab.conversion.qslab_w4 import pack_quantized_w4, unpack_w4

        torch.manual_seed(0)
        N, K, g = 512, 1024, 128
        w = (torch.randn(N, K) * 0.02).to(torch.float16)
        quantized, _ = rtn(w, g)
        qfp, scale, zero = pack_quantized_w4(quantized)
        x = (torch.randn(1, K) * 0.1).to(torch.float16).cuda()
        qfp, scale = qfp.cuda(), scale.cuda()
        y = w4a16_gemm(qfp, scale, x, g)
        y_ref = x @ unpack_w4(qfp, scale, zero.cuda(), K, g).T
        rel = ((y - y_ref).pow(2).sum() / y_ref.pow(2).sum()).item()
        assert rel < 1e-5, f"kernel numerics off (rel={rel})"
