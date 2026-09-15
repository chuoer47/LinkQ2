"""W4 weight quantization: RTN baseline + optional AWQ scaling (docs/design-m1).

--algo rtn : plain group-wise round-to-nearest, symmetric
--algo awq : per-layer activation-aware scaling first (grid search over s),
             then RTN on the rescaled weights. Equivalent transform:
             y = (W diag(s)) (diag(s)^-1 x) — quantize W' = W diag(s),
             the packed model must also scale inputs; in v1 we instead fold
             the inverse scale into the *following* op only when it is a
             Linear/norm-free path. To keep M1a simple and exact, AWQ mode
             records per-layer `act_scales` in the packed config and the
             loader folds diag(1/s) into the attention/MLP input at load
             time (absorbed into the fp16 weight of the previous op where
             legal; for LayerNorm-following layers we bake it into x via a
             wrapper). See quantize_model(...) docs.
"""
from __future__ import annotations

import torch

from quantizer.packfmt import pack_w4


@torch.no_grad()
def rtn_quantize_weight(w: torch.Tensor, group_size: int = 128,
                        clip_ratio: float = 1.0):
    """fp16 [O, I] -> packed tuple. clip_ratio<1 shrinks amax (MSE clip search)."""
    O, I = w.shape
    assert I % group_size == 0 and I % 8 == 0
    w16 = w.to(torch.float16)
    wg = w16.view(O, I // group_size, group_size)
    amax = wg.abs().amax(dim=-1, keepdim=True) * clip_ratio
    scale = (amax / 7.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(wg / scale), -8, 7).to(torch.int8)
    zero = torch.zeros_like(scale.squeeze(-1))
    packed = _pack_from_q(q, scale, zero, O, I)
    return packed


def _pack_from_q(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                 O: int, I: int):
    """Shared packing path given quantized int values [O, I/g, g]."""
    from quantizer.packfmt import pack_w4  # noqa: F401  (format compat)
    qn = q.view(O, I).to(torch.uint8) & 0xF
    qfp = torch.zeros(O, I // 8, dtype=torch.int64, device=q.device)
    for nib in range(8):
        qfp |= qn[:, nib::8].to(torch.int64) << (4 * nib)
    qfp = qfp.to(torch.int32).view(torch.uint32).cpu()
    return qfp, scale.squeeze(-1).to(torch.float16).cpu(), zero.to(torch.float16).cpu()


@torch.no_grad()
def clip_search_quantize(w: torch.Tensor, x_absmean: torch.Tensor | None = None,
                         group_size: int = 128, n_grid: int = 40):
    """AWQ-style: joint per-group MSE clip search (+ optional act scaling).

    For each group, try clip ratios in (0,1]; pick the one minimizing the
    weighted quantization MSE (weights-only if no activation stats, weighted
    by x_absmean^2 when available — proxy for output error contribution).
    Returns the packed tuple.
    """
    O, I = w.shape
    w32 = w.to(torch.float32)
    wg = w32.view(O, I // group_size, group_size)
    amax = wg.abs().amax(dim=-1, keepdim=True)

    # per-channel output-error weights
    if x_absmean is not None:
        wch = (x_absmean.to(torch.float32).reshape(1, I) ** 2)
        wch = wch.expand(O, I).view(O, I // group_size, group_size)
    else:
        wch = torch.ones_like(wg)

    best_err = torch.full((O, I // group_size), float("inf"), device=w.device)
    best_q = torch.zeros(O, I // group_size, group_size, dtype=torch.int8, device=w.device)
    best_scale = torch.zeros_like(amax)
    for i in range(n_grid):
        ratio = 1.0 - i / n_grid * 0.6          # 1.0 -> 0.4 (aggressive clip rarely helps beyond)
        scale = (amax * ratio / 7.0).clamp_min(1e-12)
        q = torch.clamp(torch.round(wg / scale), -8, 7)
        w_q = q * scale
        err = ((w_q - wg) ** 2 * wch).sum(dim=-1)
        better = err < best_err                       # [O, groups]
        best_err = torch.where(better, err, best_err)
        best_q = torch.where(better.unsqueeze(-1), q.to(torch.int8), best_q)
        best_scale = torch.where(better.unsqueeze(-1), scale, best_scale)

    zero = torch.zeros_like(best_scale.squeeze(-1))
    return _pack_from_q(best_q, best_scale, zero, O, I)


@torch.no_grad()
def awq_find_scales(w: torch.Tensor, x_absmean: torch.Tensor,
                    group_size: int = 128, n_grid: int = 20,
                    max_shrink: float = 1.0) -> torch.Tensor:
    """AWQ per-channel scaling search. For each candidate shrink ratio a,
    s = x^a (mean-normalized); quantize W diag(s) and evaluate the output
    error of the equivalent transform (w_q/s @ diag(x)). Pick the best s
    PER INPUT CHANNEL independently (s is a shared vector per candidate a,
    and we select the candidate a per channel via per-channel error sums).

    Simpler faithful variant: evaluate total error per candidate and take
    the best candidate's s vector globally. To allow per-channel selection
    we track per-channel err [O] summed over rows -> pick argmin a per
    column via stacking candidates.
    """
    x = x_absmean.to(torch.float32) + 1e-6
    w32 = w.to(torch.float32)
    O, I = w.shape
    errs = []          # per candidate: [O, I] output err
    s_list = []
    for i in range(n_grid):
        ratio = 1.0 - i * (max_shrink / n_grid)
        s = (x ** ratio)
        s = s / s.mean()
        w_scaled = w32 * s[None, :]
        wg = w_scaled.view(O, I // group_size, group_size)
        amax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 7.0).clamp_min(1e-12)
        q = torch.clamp(torch.round(wg / scale), -8, 7)
        w_q = (q * scale).view(O, I)
        err = ((w_q / s[None, :]) - w32) * x[None, :]
        errs.append(err.pow(2))
        s_list.append(s)
    errs = torch.stack(errs)                 # [n_grid, O, I]
    s_stack = torch.stack(s_list)            # [n_grid, I]
    total_per_cand = errs.sum(dim=(1, 2))    # [n_grid]
    best = int(total_per_cand.argmin().item())
    return s_list[best]


@torch.no_grad()
def quantize_weight(w: torch.Tensor, algo: str = "rtn",
                    x_absmean: torch.Tensor | None = None,
                    group_size: int = 128):
    """Dispatch. rtn = plain symmetric; rtn_clip = per-group MSE clip search;
    awq = act scaling + clip search."""
    if algo == "rtn":
        return rtn_quantize_weight(w, group_size), None
    if algo == "rtn_clip":
        return clip_search_quantize(w, None, group_size), None
    if algo == "awq":
        assert x_absmean is not None, "AWQ needs calibration stats"
        s = awq_find_scales(w, x_absmean, group_size)
        w_scaled = (w.to(torch.float32) * s[None, :]).to(torch.float16)
        packed = clip_search_quantize(w_scaled, x_absmean, group_size)
        return packed, s.to(torch.float16)
    raise ValueError(f"unknown algo {algo}")
