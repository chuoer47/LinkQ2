"""GPU tests: kernel numerics and KV cache roundtrip (need a CUDA device)."""
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
        from qslab.kernels.ops import w4a16_gemm
        from qslab.quant.packfmt import pack_w4, unpack_w4

        torch.manual_seed(0)
        N, K, g = 512, 1024, 128
        w = (torch.randn(N, K) * 0.02).to(torch.float16)
        qfp, scale, zero = pack_w4(w, g)
        x = (torch.randn(1, K) * 0.1).to(torch.float16).cuda()
        qfp, scale = qfp.cuda(), scale.cuda()
        y = w4a16_gemm(qfp, scale, x, g)
        y_ref = x @ unpack_w4(qfp, scale, zero.cuda(), K, g).T
        rel = ((y - y_ref).pow(2).sum() / y_ref.pow(2).sum()).item()
        assert rel < 1e-5, f"kernel numerics off (rel={rel})"


class TestKVCacheRoundtrip:
    """Every cache implementation must return sane fp16 K/V for its precision."""

    @pytest.mark.parametrize("cls_name,max_err", [
        ("FP16KVCache", 0.0),
        ("KV8Cache", 1e-3),
        ("KV4Cache", 3e-2),
    ])
    def test_roundtrip_error_within_precision(self, cls_name, max_err):
        import qslab.quant.cache.kv_cache as kvc

        torch.manual_seed(0)
        B, H, D, g, T = 1, 8, 128, 64, 128
        cls = getattr(kvc, cls_name)
        cache = cls(B, H, D, 1024, "cuda:0")
        k = (torch.randn(B, H, T, D) * 0.5).to(torch.float16).cuda()
        v = (torch.randn(B, H, T, D) * 0.5).to(torch.float16).cuda()
        kr, vr = cache.update(k, v)
        ek = (kr - k).float().pow(2).sum() / k.float().pow(2).sum()
        ev = (vr - v).float().pow(2).sum() / v.float().pow(2).sum()
        assert ek.item() <= max_err, f"{cls_name} K error {ek.item():.4f}"
        assert ev.item() <= max_err, f"{cls_name} V error {ev.item():.4f}"

    def test_kv4_memory_is_smaller_than_fp16(self):
        from qslab.quant.cache.kv_cache import FP16KVCache, KV4Cache

        args = dict(batch=1, num_kv_heads=8, head_dim=128, max_len=4096,
                    device="cuda:0")
        fp16 = FP16KVCache(**args)
        kv4 = KV4Cache(**args, group=64)
        assert kv4.memory_bytes < fp16.memory_bytes / 3

    def test_kv4_odd_length_updates(self):
        """Decode appends 1 token at a time; tail-group handling must hold."""
        from qslab.quant.cache.kv_cache import KV4Cache

        torch.manual_seed(0)
        cache = KV4Cache(1, 8, 128, 2048, "cuda:0", group=64)
        ref_k, ref_v = [], []
        for T in [5, 1, 1, 1]:
            k = (torch.randn(1, 8, T, 128) * 0.5).to(torch.float16).cuda()
            v = (torch.randn(1, 8, T, 128) * 0.5).to(torch.float16).cuda()
            ref_k.append(k)
            ref_v.append(v)
            kr, vr = cache.update(k, v)
            assert kr.shape[2] == cache.len
        assert cache.len == 8
