"""L1 concrete QuantBackends: v1 custom GEMV, Marlin GEMM, and the hybrid.

These wrap the L0 kernels and expose only ``linear(x)`` to L2. The hybrid
(AutoBackend) is where the M-based dispatch now lives — moved out of
qslab/models/w4linear.py during the R1/R2 refactor.
"""
from __future__ import annotations

import torch

from qslab.quant.backends import QUANT_BACKENDS


class _BackendBase:
    """Shared bookkeeping for a single packed weight."""

    def __init__(self, qfp: torch.Tensor, scale: torch.Tensor,
                 group_size: int, in_features: int, out_features: int,
                 act_scale: torch.Tensor | None = None):
        self.qfp = qfp
        self.scale = scale
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.act_scale = act_scale

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_scale is not None:
            x = x / self.act_scale
        return x


@QUANT_BACKENDS.register("w4.v1")
class W4V1Backend(_BackendBase):
    """Custom w4a16 GEMV kernel. Fast at M=1 (decode), degrades with M."""

    name = "w4.v1"
    bits = 4

    def usable(self, in_features, out_features, group_size) -> bool:
        return in_features % 8 == 0 and in_features % group_size == 0

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        from qslab.kernels.ops import w4a16_gemm
        orig = x.shape
        x2 = x.reshape(-1, orig[-1]) if x.dim() != 2 else x
        x2 = self._prepare_x(x2)
        y = w4a16_gemm(self.qfp, self.scale, x2, self.group_size)
        return y.reshape(*orig[:-1], self.out_features).to(x.dtype)

    def memory_bytes(self) -> int:
        return self.qfp.nelement() * 4 + self.scale.nelement() * 2


@QUANT_BACKENDS.register("w4.marlin")
class W4MarlinBackend(_BackendBase):
    """Marlin (IST-DASLab) tensor-core GEMM. Strong from M>=8 up."""

    name = "w4.marlin"
    bits = 4

    def __init__(self, qfp, scale, group_size, in_features, out_features,
                 act_scale=None):
        super().__init__(qfp, scale, group_size, in_features, out_features,
                         act_scale)
        from qslab.kernels.marlin_backend import pack_v1_to_marlin
        self._B, self._s = pack_v1_to_marlin(qfp, scale, in_features, group_size)
        self._ws = torch.zeros(out_features // 128 * 16, dtype=torch.int32)

    def usable(self, in_features, out_features, group_size) -> bool:
        return (in_features % 128 == 0 and out_features % 256 == 0
                and group_size in (-1, 128))

    def to(self, device):
        self._B = self._B.to(device)
        self._s = self._s.to(device)
        self._ws = self._ws.to(device)
        return self

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        from qslab.kernels.marlin_backend import marlin_gemm
        orig = x.shape
        x2 = x.reshape(-1, orig[-1]) if x.dim() != 2 else x
        x2 = self._prepare_x(x2)
        y = marlin_gemm(x2, self._B, self._s, self._ws)
        return y.reshape(*orig[:-1], self.out_features).to(x.dtype)

    def memory_bytes(self) -> int:
        return (self.qfp.nelement() * 4 + self.scale.nelement() * 2      # v1 pack
                + self._B.nelement() * 4 + self._s.nelement() * 2)      # marlin copy


@QUANT_BACKENDS.register("w4.auto")
class W4AutoBackend(_BackendBase):
    """Hybrid: v1 GEMV for small M, Marlin GEMM above the crossover.

    The crossover (8) comes from the M5 benchmark table
    (results/m5_kernel_bench.json): Marlin's ~100us/call fixed cost loses to
    v1 below M=8, wins clearly above.
    """

    name = "w4.auto"
    bits = 4
    CROSSOVER_M = 8

    def __init__(self, qfp, scale, group_size, in_features, out_features,
                 act_scale=None):
        super().__init__(qfp, scale, group_size, in_features, out_features,
                         act_scale)
        self._v1 = W4V1Backend(qfp, scale, group_size, in_features,
                               out_features, act_scale)
        self._marlin = None
        if W4MarlinBackend.usable(self, in_features, out_features, group_size):
            try:
                self._marlin = W4MarlinBackend(qfp, scale, group_size,
                                               in_features, out_features,
                                               act_scale)
            except Exception:
                self._marlin = None

    def usable(self, in_features, out_features, group_size) -> bool:
        return W4V1Backend.usable(self, in_features, out_features, group_size)

    def to(self, device):
        if self._marlin is not None:
            self._marlin.to(device)
        return self

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        M = x.numel() // orig[-1] if x.dim() > 1 else 1
        if self._marlin is not None and M > self.CROSSOVER_M:
            return self._marlin.linear(x)
        return self._v1.linear(x)

    def memory_bytes(self) -> int:
        base = self._v1.memory_bytes()
        return base + (self._marlin.memory_bytes() - base // 2 if self._marlin
                       else 0)


def get_backend(name: str, qfp, scale, group_size, in_features, out_features,
                act_scale=None):
    """Factory: build a backend instance by registry name.

    This module IS the registration site: importing it (which using this
    factory does) is what populates QUANT_BACKENDS.
    """
    cls = QUANT_BACKENDS.get(name)
    return cls(qfp, scale, group_size, in_features, out_features, act_scale)
