"""CPU-fast unit tests: packing roundtrip, registry, sampling math."""
from __future__ import annotations

import pytest
import torch

from qslab.quant.packfmt import pack_w4, unpack_w4
from qslab.registry import Registry


class TestPackFormat:
    """qslab_w4_v1 nibble packing must be a lossless container."""

    def test_roundtrip_within_int4_grid(self):
        torch.manual_seed(0)
        O, I, g = 64, 512, 128
        w = (torch.randn(O, I) * 0.02).to(torch.float16)
        qfp, scale, zero = pack_w4(w, g)
        w_hat = unpack_w4(qfp, scale, zero, I, g)
        err = (w.float() - w_hat.float()).abs().max().item()
        # int4 symmetric quantization error is bounded by half a step
        assert err <= scale.max().item() / 2 + 1e-6

    def test_symmetric_zero_field_is_zero(self):
        torch.manual_seed(1)
        w = (torch.randn(32, 256) * 0.05).to(torch.float16)
        _qfp, _scale, zero = pack_w4(w, 128)
        assert torch.equal(zero, torch.zeros_like(zero))

    def test_pack_shape(self):
        w = torch.zeros(64, 512, dtype=torch.float16)
        qfp, scale, _ = pack_w4(w, 128)
        assert qfp.shape == (64, 512 // 8)
        assert scale.shape == (64, 512 // 128)
        assert qfp.dtype == torch.uint32


class TestRegistry:
    def test_register_and_get(self):
        r = Registry("thing")

        @r.register("a")
        class A:
            pass

        assert r.get("a") is A
        assert "a" in r
        assert r.names() == ["a"]

    def test_duplicate_registration_rejected(self):
        r = Registry("thing")

        @r.register("dup")
        class A:
            pass

        with pytest.raises(ValueError):
            @r.register("dup")
            class B:
                pass

    def test_unknown_name_lists_available(self):
        r = Registry("thing")

        @r.register("known")
        class A:
            pass

        with pytest.raises(KeyError) as e:
            r.get("missing")
        assert "known" in str(e.value)


class TestW4PackResidency:
    """A backend that repacks must not keep the source pack alive (M10: the
    double int4 residency measured 3.335 GB on 8B)."""

    def _pack(self, O=256, I=1024, g=128):
        torch.manual_seed(3)
        w = (torch.randn(O, I) * 0.02).to(torch.float16)
        qfp, scale, _zero = pack_w4(w, g)
        return qfp, scale, g, I, O

    def _packed_bytes(self, qfp, scale):
        return qfp.numel() * 4 + scale.numel() * 2

    def test_marlin_drops_the_v1_pack(self):
        from qslab.models.w4linear import W4Linear
        qfp, scale, g, I, O = self._pack()
        m = W4Linear(qfp, scale, g, I, O, backend="w4.marlin")
        assert m._backend.uses_v1_pack is False
        assert getattr(m, "qfp", None) is None
        assert getattr(m, "scale", None) is None
        assert m.weight_memory_bytes() <= self._packed_bytes(qfp, scale) * 1.02

    def test_v1_backend_keeps_the_pack(self):
        from qslab.models.w4linear import W4Linear
        qfp, scale, g, I, O = self._pack()
        m = W4Linear(qfp, scale, g, I, O, backend="w4.v1")
        assert m._backend.uses_v1_pack is True
        assert m.qfp is qfp and m.scale is scale
        assert m.weight_memory_bytes() == self._packed_bytes(qfp, scale)

    def test_buffer_free_layer_survives_to(self):
        """W4Linear.to() used to read the destination off the first buffer."""
        from qslab.models.w4linear import W4Linear
        qfp, scale, g, I, O = self._pack()
        m = W4Linear(qfp, scale, g, I, O, backend="w4.marlin")
        assert m.to(torch.device("cpu")) is m
        assert m._backend._B.device.type == "cpu"
