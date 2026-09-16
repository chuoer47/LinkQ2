"""Losslessness test for speculative rejection sampling (M3-S5 prerequisite).

Simulate: target distribution q fixed; draft p varies (close to q and far).
Run 50k acceptance steps; the empirical token distribution of the output
must match sampling from q directly (chi-square / total-variation check).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.sampler import speculative_reject_sample, sample_token


def run_case(name: str, p: torch.Tensor, q: torch.Tensor, n: int = 50_000):
    gen = torch.Generator().manual_seed(42)
    counts = torch.zeros_like(q)
    for _ in range(n):
        x, _ = speculative_reject_sample(p, q, generator=gen)
        counts[x] += 1
    empirical = counts / counts.sum()
    direct = torch.softmax(q, dim=-1)
    tv = 0.5 * (empirical - direct).abs().sum().item()
    print(f"{name:22s} total-variation vs target-only: {tv:.4f}")
    assert tv < 0.02, f"{name}: distribution mismatch (tv={tv})"


def main():
    torch.manual_seed(0)
    V = 50
    # target q: skewed distribution
    logits_q = torch.randn(V)
    q = torch.softmax(logits_q, -1)

    # case 1: draft == target (accept everything, trivially lossless)
    run_case("p == q", q, q)

    # case 2: draft close to target (small perturbation)
    p_close = torch.softmax(logits_q + 0.3 * torch.randn(V), -1)
    run_case("p ~ q (close)", p_close, q)

    # case 3: draft very different (low acceptance, still must be lossless)
    p_far = torch.softmax(-logits_q + 0.5 * torch.randn(V), -1)
    run_case("p vs anti-q (far)", p_far, q)

    # greedy sampler sanity
    z = torch.tensor([0.1, 3.0, 0.2])
    assert sample_token(z) == 1
    assert sample_token(z, temperature=1.0, top_k=1) == 1
    print("SAMPLER PASS: speculative rejection is distribution-lossless")


if __name__ == "__main__":
    main()
