"""Acceptance operates on tensors/tokens and never mutates an inference request."""
import torch


def rejection_sample(logits, temperatures, drafts, draft_probs, sampled):
    """Probability-ratio acceptance; a deterministic proposer has one-hot probability.

    Rows without a proposal retain their normal sample. Rejected rows are sampled
    from normalized max(target - draft, 0), so cannot match the rejected proposal.
    """
    out = sampled.clone()
    rows = (drafts >= 0) & (temperatures > 1e-6)
    if not bool(rows.any()):
        return out
    t = temperatures[rows].unsqueeze(1)
    q = torch.softmax(logits[rows].float().div_(t), dim=-1)
    x = drafts[rows]
    if draft_probs is None:
        p_x = torch.ones_like(x, dtype=torch.float32)
        residual = q.clone()
        residual.scatter_(1, x.unsqueeze(1), 0.0)
    else:
        p = draft_probs[rows].float()
        p_x = p.gather(1, x.unsqueeze(1)).squeeze(1)
        residual = (q - p).clamp_min(0)
    q_x = q.gather(1, x.unsqueeze(1)).squeeze(1)
    accept = torch.rand(q_x.shape, device=q.device) < (q_x / p_x.clamp_min(1e-12)).clamp(max=1.0)
    total = residual.sum(1, keepdim=True)
    residual = torch.where(total > 0, residual / total.clamp_min(1e-12), q)
    resampled = torch.multinomial(residual, 1).squeeze(1)
    idx = rows.nonzero(as_tuple=True)[0]
    out[idx[accept]] = x[accept]
    out[idx[~accept]] = resampled[~accept]
    return out


def accepted_prefix(drafts: list[int], verified: list[int]) -> tuple[list[int], int]:
    """Return the matched prefix plus its correction/bonus, and accepted draft count."""
    accepted = 0
    while accepted < len(drafts) and verified[accepted] == drafts[accepted]:
        accepted += 1
    return drafts[:accepted] + [verified[accepted]], accepted
