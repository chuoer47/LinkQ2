"""Common interface for online quantized Linear implementations."""
from abc import ABC, abstractmethod

from torch import nn


class QuantLinear(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def from_quantized(cls, values, scale, *, group_size, symmetric,
                       zero_point=None, input_scale=None, device, dtype):
        """Prepare a layer once from logical quantized tensors."""

    @abstractmethod
    def forward(self, x):
        """Compute the projection without materializing a dense weight matrix."""

    def weight_memory_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in self.buffers())
