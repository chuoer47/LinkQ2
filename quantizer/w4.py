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
def rtn_quantize_weight(w: torch.Tensor, group_size: int = 128):
    """fp16 [O, I] -> packed tuple."""
    return pack_w4(w, group_size)


@torch.no_grad()
def awq_find_scales(w: torch.Tensor, x_absmean: torch.Tensor,
                    group_size: int = 128, n_grid: int = 20,
                    max_shrink: float = 1.0) -> torch.Tensor:
    """Grid-search per-channel scaling s in (0,1] minimizing quantized MSE.

    w          : fp16 weight [O, I]
    x_absmean  : fp16 per-input-channel mean abs activation [I]
    returns    : s [I] in (0, 1]
    """
    x = x_absmean.to(torch.float32) + 1e-6
    best_err = torch.full((w.shape[0],), float("inf"), device=w.device)
    best_s = torch.ones(w.shape[1], device=w.device)
    w32 = w.to(torch.float32)
    for i in range(n_grid):
        ratio = 1.0 - i * (max_shrink / n_grid)          # 1.0 -> ~0
        s = (x ** ratio)
        # normalize so mean scale ~1 (keeps magnitudes stable)
        s = s / s.mean()
        w_scaled = (w32 * s[None, :])
        # simulate group quantization error
        O, I = w_scaled.shape
        wg = w_scaled.view(O, I // group_size, group_size)
        amax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 7.0).clamp_min(1e-12)
        q = torch.clamp(torch.round(wg / scale), -8, 7)
        w_q = (q * scale).view(O, I)
        # err = || (w_q / s) X - W X ||^2 with X = diag(x): per out-channel
        err = (((w_q / s[None, :]) - w32) * x[None, :]).pow(2).sum(dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better[None, :].T.expand_as(best_s.T).T, s[None, :].expand(O, -1),
                             best_s[None, :].expand(O, -1))[0] if False else best_s
        # simpler update: keep s vector when any channel improved
        if better.any():
            best_s = s.clone()
    return best_s


@torch.no_grad()
def quantize_weight(w: torch.Tensor, algo: str = "rtn",
                    x_absmean: torch.Tensor | None = None,
                    group_size: int = 128):
    """Dispatch: returns (packed_tuple, s_or_None). If AWQ, w is pre-scaled
    and caller must store s to fold back at load time."""
    if algo == "rtn":
        return rtn_quantize_weight(w, group_size), None
    if algo == "awq":
        assert x_absmean is not None, "AWQ needs calibration stats"
        s = awq_find_scales(w, x_absmean, group_size)
        w_scaled = (w.to(torch.float32) * s[None, :]).to(torch.float16)
        return rtn_quantize_weight(w_scaled, group_size), s.to(torch.float16)
    raise ValueError(f"unknown algo {algo}")
