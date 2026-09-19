"""L1 concrete QuantBackends: v1 custom GEMV, Marlin GEMM, and the hybrid.

These wrap the L0 kernels and expose only ``linear(x)`` to L2. The hybrid
(AutoBackend) is where the M-based dispatch now lives — moved out of
qslab/models/w4linear.py during the R1/R2 refactor.
"""
from __future__ import annotations

import torch

from qslab.quant.backends import QUANT_BACKENDS


def _bytes(*ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


class _BackendBase:
    """Shared bookkeeping for a single packed weight."""

    #: False once the backend has repacked into its own layout and no longer
    #: reads the v1 pack. L2 (W4Linear) releases its qfp/scale buffers on this
    #: flag — holding both was measured as 3.335 GB of dead weight on 8B.
    uses_v1_pack = True

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
        return _bytes(self.qfp, self.scale)


@QUANT_BACKENDS.register("w4.marlin")
class W4MarlinBackend(_BackendBase):
    """Marlin (IST-DASLab) tensor-core GEMM. Strong from M>=8 up."""

    name = "w4.marlin"
    bits = 4
    uses_v1_pack = False

    def __init__(self, qfp, scale, group_size, in_features, out_features,
                 act_scale=None):
        super().__init__(qfp, scale, group_size, in_features, out_features,
                         act_scale)
        from qslab.kernels.marlin_backend import pack_v1_to_marlin
        self._B, self._s = pack_v1_to_marlin(qfp, scale, in_features, group_size)
        self._ws = torch.zeros(out_features // 128 * 16, dtype=torch.int32)
        # nothing on this path reads the v1 pack again; keeping it cost as much
        # memory as Marlin's own copy (results/m10_w4_residency.txt)
        self.qfp = self.scale = None

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
        return _bytes(self._B, self._s, self._ws)


@QUANT_BACKENDS.register("w4.auto")
class W4AutoBackend(_BackendBase):
    """Marlin wherever it is usable; v1 GEMV only as the fallback.

    The M5 microbenchmark concluded v1 wins below M=8 (Marlin's per-call
    fixed cost), and w4.auto dispatched on that for M5-M8. M9 overturned it
    e2e, twice, on both models (graph decode, 4090, median of 3):

      M=1  1.7B: v1 145.0 vs Marlin 155.2 tok/s  (x1.07)
      M=1  8B : v1  54.1 vs Marlin  91.1 tok/s  (x1.69)  <- the "8B 55.8
      M=5  8B : v1 65ms/step vs Marlin 11.5ms (spec verify: a v1 batch
      re-reads the weights per row; Marlin reads them once for any M)

    So the microbenchmark's fixed-cost model didn't transfer to a 252-linear
    forward inside a CUDA graph. Marlin and v1 agree numerically to 0.047
    logits (both ~4.0 from fp16, i.e. the W4 quantization itself), so the
    switch costs no accuracy — only the greedy near-tie flips documented in
    notes/M9. When Marlin is usable the v1 copy is not built at all (it was
    dead weight: auto previously held BOTH packed layouts in memory), and
    `uses_v1_pack` is reported False so L2 releases the qfp/scale buffers that
    fed the repack — the last resident copy of them was the 3.335 GB M10
    measured on 8B.
    """

    name = "w4.auto"
    bits = 4
    CROSSOVER_M = 0

    def __init__(self, qfp, scale, group_size, in_features, out_features,
                 act_scale=None):
        super().__init__(qfp, scale, group_size, in_features, out_features,
                         act_scale)
        self._marlin = None
        if W4MarlinBackend.usable(self, in_features, out_features, group_size):
            try:
                self._marlin = W4MarlinBackend(qfp, scale, group_size,
                                               in_features, out_features,
                                               act_scale)
            except Exception:
                self._marlin = None
        # built only when it can actually be reached (Marlin missing, or a
        # nonzero crossover someone sets for experimentation)
        self._v1 = None
        if self._marlin is None or self.CROSSOVER_M >= 1:
            self._v1 = W4V1Backend(qfp, scale, group_size, in_features,
                                   out_features, act_scale)
        self.uses_v1_pack = self._v1 is not None
        if not self.uses_v1_pack:
            self.qfp = self.scale = None

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
        if self._v1 is not None:
            return self._v1.linear(x)
        return self._marlin.linear(x)

    def memory_bytes(self) -> int:
        # each child reports only its own copy; the two coexist exactly when
        # both were built (a nonzero CROSSOVER_M, or Marlin being unusable)
        return ((self._v1.memory_bytes() if self._v1 else 0)
                + (self._marlin.memory_bytes() if self._marlin else 0))


def get_backend(name: str, qfp, scale, group_size, in_features, out_features,
                act_scale=None):
    """Factory: build a backend instance by registry name.

    This module IS the registration site: importing it (which using this
    factory does) is what populates QUANT_BACKENDS.
    """
    cls = QUANT_BACKENDS.get(name)
    return cls(qfp, scale, group_size, in_features, out_features, act_scale)
