"""Offline calibration: SmoothAttention factors + static KV scales.

Implements QServe's SmoothAttention (QoQ, arXiv:2405.04532 §IV-B, Eq. 7-9)
as a preprocessing step for qslab's KV4 cache:

  lambda[layer][h, d]   per-channel smoothing factor, alpha=0.5. Outlier
                        channels of K are divided down, and the same factor
                        is multiplied onto Q (which is never quantized), so
                        the attention logits are unchanged. The optimizer's
                        hard constraint lambda_d == lambda_{d+D/2} keeps it
                        commutative with RoPE's channel pairing.

  kscale[layer][h, d]   static per-channel int4 scale for the SMOOTHED key,
                        = max|K/lambda| / 7 over the calibration set.

Why static for K: at runtime a per-token key scale cannot be recomputed when
new tokens arrive without rewriting already-stored slots, and any write whose
address depends on data breaks CUDA Graph capture. A frozen per-channel scale
makes the K write a pure function of slot_mapping. V has no such constraint —
its outliers are token-local, so it keeps a dynamic per-token scale.

The factors fold into the existing per-head RMSNorm weights:
    q_norm.weight *= lambda ;  k_norm.weight /= lambda
so no new runtime kernel is needed.

Usage:
    python scripts/build_smooth_kv.py --model models/Qwen3-1.7B \
        --out results/smooth_kv4_qwen3-1.7b.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qslab.models.loader import load_reference_model  # noqa: E402

ALPHA = 0.5
DEFAULT_CALIB = "results/frozen/calib_c4_128x2048.pt"
_MIN_SCALE = 1e-4


def apply_rope_halfsplit(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """HF-style rotate_half RoPE. x [T, H, D]; cos/sin [T, D] or [T, 1, D]."""
    if cos.dim() == 2:
        cos, sin = cos[:, None, :], sin[:, None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[..., :half], sin[..., :half]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def calibrate(model_path: str, calib_path: str, n_seqs: int, seq_len: int,
              alpha: float = ALPHA, verbose: bool = True):
    blob = torch.load(calib_path, weights_only=False)
    seqs = [list(s)[:seq_len] for s in blob["token_ids"][:n_seqs]]
    if verbose:
        print(f"calibration set: {blob['dataset']}  {len(seqs)} seqs x {seq_len} tokens")

    model = load_reference_model(model_path)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)

    k_absmax = torch.zeros(n_layers, n_kv, head_dim, dtype=torch.float64)
    caps: dict[int, torch.Tensor] = {}

    def mk_hook(li):
        def hook(mod, args, kwargs, out):
            caps[li] = (args[0] if args else kwargs.get("hidden_states")).detach()
        return hook

    handles = [model.model.layers[li].self_attn.register_forward_hook(
        mk_hook(li), with_kwargs=True) for li in range(n_layers)]

    with torch.inference_mode():
        for si, ids in enumerate(seqs):
            caps.clear()
            tokens = torch.tensor([ids], device="cuda")
            model(tokens)
            emb = model.model.embed_tokens(tokens)
            pos = torch.arange(tokens.shape[1], device="cuda").view(1, -1)
            cos, sin = model.model.rotary_emb(emb, pos)
            cos, sin = cos[0], sin[0]
            for li in range(n_layers):
                attn = model.model.layers[li].self_attn
                k = attn.k_norm(attn.k_proj(caps[li]).view(-1, n_kv, head_dim)).float()
                k_rope = apply_rope_halfsplit(k, cos, sin)
                k_absmax[li] = torch.maximum(k_absmax[li],
                                             k_rope.abs().amax(0).double().cpu())
            if verbose and si % 8 == 0:
                print(f"  seq {si + 1}/{len(seqs)}", flush=True)

    for h in handles:
        h.remove()

    # Eq. 9: RoPE pairs channel d with d + D/2, so lambda must be equal across
    # each pair for the scaling to commute with RoPE.
    half = head_dim // 2
    paired = torch.maximum(k_absmax[..., :half], k_absmax[..., half:])
    lam_half = paired.clamp_min(1e-6) ** alpha
    lam = torch.cat([lam_half, lam_half], dim=-1)

    # static per-channel int4 scale of the smoothed key
    kscale = ((k_absmax / lam.double()).clamp_min(_MIN_SCALE) / 7.0)

    if verbose:
        raw_ratio = (k_absmax[0].amax(-1) / k_absmax[0].mean(-1)).mean().item()
        sm_ratio = ((k_absmax[0] / lam[0].double()).amax(-1)
                    / (k_absmax[0] / lam[0].double()).mean(-1)).mean().item()
        print(f"calibrated: alpha={alpha}  K channel spread "
              f"{raw_ratio:.1f}x -> {sm_ratio:.1f}x after smoothing; "
              f"lambda range [{lam.min():.3f}, {lam.max():.3f}]")

    return lam.float(), kscale.float(), dict(
        model=model_path, calib=calib_path, alpha=alpha,
        n_seqs=len(seqs), seq_len=seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    lam, kscale, meta = calibrate(args.model, args.calib, args.n_seqs,
                                  args.seq_len, args.alpha)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lambda": lam.half(), "kscale": kscale.half(), "meta": meta}, out)
    print(f"wrote {out}  lambda{tuple(lam.shape)} kscale{tuple(kscale.shape)}")


if __name__ == "__main__":
    main()
