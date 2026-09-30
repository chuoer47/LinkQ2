"""Interfaces shared by offline weight quantization algorithms."""
from __future__ import annotations

from typing import Protocol

import torch

from qslab.artifacts.schema import QuantizedTensor


class WeightQuantizer(Protocol):
    name: str

    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]: ...
