"""Model-backed speculative execution and its common proposal result."""
from dataclasses import dataclass

import torch


@dataclass
class DraftProposal:
    tokens: list[int]
    probabilities: torch.Tensor | None = None
