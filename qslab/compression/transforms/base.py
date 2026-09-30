"""Protocol for invertible or inference-preserved compression transforms."""
from __future__ import annotations

from typing import Protocol

import torch


class ModelTransform(Protocol):
    name: str

    def apply(self, model: torch.nn.Module, **kwargs) -> dict[str, torch.Tensor]:
        """Apply a transform and return tensors required to reproduce its semantics."""
