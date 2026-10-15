"""Shared interface for offline weight quantization algorithms."""
from abc import ABC, abstractmethod

import torch

from qslab.artifacts.schema import QuantizedTensor


class WeightQuantizer(ABC):
    requires_calibration = False
    default_group_size = 128

    @abstractmethod
    def quantize(self, weight: torch.Tensor, *, group_size: int,
                 activation_absmean: torch.Tensor | None = None
                 ) -> tuple[QuantizedTensor, torch.Tensor | None]:
        """Return logical W4 tensors and an optional input-channel scale."""
