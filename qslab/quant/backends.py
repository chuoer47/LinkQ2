"""L1: QuantBackend — the pluggable weight-quantization strategy interface.

The backend owns BOTH halves of a quantization scheme:
  1. packing a fake-quantized fp16 weight into the on-device format
  2. running ``y = x @ dequant(W)^T`` for arbitrary M

Crucially, **the M-dependent dispatch decision lives inside the backend**
(previously an ``if backend == "autodetect"`` in L2's W4Linear). L2 just
calls ``backend.linear(x)`` and the backend knows whether to use the custom
GEMV kernel (small M) or the Marlin GEMM (large M).

Layer note: this module is L1 and MUST NOT import transformers. It may
import L0 (qslab.kernels) — that is the allowed downward direction.
"""
from __future__ import annotations

from typing import Protocol

import torch

from qslab.registry import Registry

QUANT_BACKENDS = Registry("quant backend")


class QuantBackend(Protocol):
    """A weight-only weight-quantization scheme (e.g. W4A16)."""

    #: human name, also the registry key
    name: str
    #: bits per weight
    bits: int
    #: False once the backend repacked into its own layout and no longer reads
    #: the v1 pack; L2 (W4Linear) releases its qfp/scale buffers on this flag
    uses_v1_pack: bool = True
    #: whether this backend is usable for the given linear shape
    def usable(self, in_features: int, out_features: int, group_size: int) -> bool: ...

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """y = x @ dequant(W)^T. x is [..., in_features] fp16; returns fp16."""
        ...

    def memory_bytes(self) -> int:
        """On-device weight footprint, counting only the copies kept resident."""
        ...
