"""Sampling math for qslab: temperature/top-k/top-p + speculative rejection.

The rejection sampler implements the lossless speculative decoding rule
(Leviathan & Chen 2023): accept draft token x with prob min(1, q(x)/p(x));
on rejection resample from normalized max(0, q - p).
"""
from __future__ import annotations

import torch


def sample_token(logits: torch.Tensor, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 1.0,
                 generator: torch.Generator | None = None) -> int:
    """Greedy when temperature==0; otherwise temperature/top-k/top-p sample.
    logits: [vocab] (1D)."""
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1))
    z = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(z, min(top_k, z.shape[-1])).values[-1]
        z = torch.where(z < kth, torch.full_like(z, float("-inf")), z)
    if top_p < 1.0:
        sorted_z, sorted_idx = torch.sort(z, descending=True)
        probs = torch.softmax(sorted_z, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum - probs < top_p          # tokens whose cumulative mass (exclusive) < p
        sorted_z = torch.where(keep, sorted_z, torch.full_like(sorted_z, float("-inf")))
        z = torch.full_like(z, float("-inf"))
        z[sorted_idx] = sorted_z
    probs = torch.softmax(z, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def speculative_reject_sample(p_logits: torch.Tensor, q_logits: torch.Tensor,
                              generator: torch.Generator | None = None
                              ) -> tuple[int, bool]:
    """One step of lossless speculative acceptance.

    p_logits: draft distribution [vocab]
    q_logits: target distribution [vocab]
    Returns (token, accepted). Token is the final token for this position:
    either the draft token (accepted) or a resample from max(0, q-p).
    """
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    x = int(torch.multinomial(p, 1, generator=generator))
    ratio = (q[x] / p[x].clamp_min(1e-12)).item()
    accept_prob = min(1.0, ratio)
    u = torch.rand(1, generator=generator).item()
    if u < accept_prob:
        return x, True
    # resample from normalized max(0, q - p)
    diff = (q - p).clamp_min(0)
    total = diff.sum()
    if total <= 0:
        # q == p everywhere accepted or q dominates nowhere: fall back to q
        return int(torch.multinomial(q, 1, generator=generator)), False
    return int(torch.multinomial(diff / total, 1, generator=generator)), False
