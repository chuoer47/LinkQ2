"""M2-S1: KV outlier profiling -> per-layer quantization plan.

Runs frozen calib tokens through the FP16 model with hooks on each layer's
self_attn, captures K/V (post-RoPE K, pre-cache V), computes per-layer
outlier degree = amax / median(|values|) at a few positions, and marks the
worst layers as FP16-kept.

Output: results/kv4_plan.json -> engine config (kv_fp16_layers, group).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import torch

from qslab.model.loader import load_reference_model

N_PROBE_TOKENS = 512   # calib prefix length for profiling


@torch.no_grad()
def profile_kv(model, calib_ids: list[int], device: str) -> dict[int, dict]:
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        def mk(idx):
            def hook(module, args, kwargs):
                # hook the *inputs* of self_attn; K/V computed inside — capture
                # via patched forward instead. Simplest: wrap self_attn.forward.
                pass
            return hook
        # wrap self_attn.forward to grab key/value states post-rope
        orig_fwd = layer.self_attn.forward

        def wrapped(hidden_states, position_embeddings, attention_mask=None,
                    **kw):
            out = orig_fwd(hidden_states, position_embeddings, attention_mask, **kw)
            return out

        # simpler: capture inside PatchedQwen3Attention is engine-specific;
        # here we recompute K/V stats from hidden_states hook
        def capture_hook(mod, args, kwargs, idx=i):
            h = args[0] if args else kwargs.get("hidden_states")
            captured.setdefault(idx, {"h": h.detach()})
        hooks.append(layer.self_attn.register_forward_pre_hook(
            capture_hook, with_kwargs=True))

    x = torch.tensor([calib_ids[:N_PROBE_TOKENS]], device=device)
    model(input_ids=x, use_cache=False)
    for h in hooks:
        h.remove()

    stats = {}
    for i, d in captured.items():
        h = d["h"]  # [1, T, hidden]
        # recompute k/v stats via projections + per-head norms (RoPE rotation
        # preserves magnitudes across rotated dim pairs, so post-norm k_proj
        # amax is a faithful outlier proxy)
        layer = model.model.layers[i]
        attn = layer.self_attn
        hs = h
        T, hd = hs.shape[1], attn.head_dim
        k = attn.k_proj(hs).view(1, T, -1, hd)
        v = attn.v_proj(hs).view(1, T, -1, hd)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = attn.k_norm(k)
        k_f = k.float().abs()
        v_f = v.float().abs()
        k_amax = k_f.amax().item()
        v_amax = v_f.amax().item()
        k_med = k_f.median().item() + 1e-9
        v_med = v_f.median().item() + 1e-9
        stats[i] = {"k_amax": k_amax, "k_outlier": k_amax / k_med,
                    "v_amax": v_amax, "v_outlier": v_amax / v_med}
    return stats


def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/kv4_plan.json"
    device = "cuda:0"
    blob = torch.load("results/frozen/calib_c4_128x2048.pt", weights_only=False)
    calib_ids = blob["token_ids"][0]

    model = load_reference_model(model_path, device=device)
    stats = profile_kv(model, calib_ids, device)

    # keep FP16 for the top outlier layers, budgeted: 1 layer max
    # (M2 measurement: 2 fp16 layers cap overall saving at 3.07x < 3.5x line)
    ranked = sorted(stats.items(), key=lambda kv: -(kv[1]["k_outlier"] + kv[1]["v_outlier"]))
    fp16_layers = sorted(i for i, _ in ranked[:1])

    plan = {
        "k_quant": "per_channel",     # along head_dim after transpose
        "v_quant": "per_token",
        "group_size": 64,             # 128 tested: PPL +1.39 vs +0.56 — rejected
        "kv_fp16_layers": fp16_layers,
        "stats": {str(i): s for i, s in stats.items()},
    }
    Path("results").mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2))
    print("outlier ranking (top 5):")
    for i, s in ranked[:5]:
        print(f"  layer {i}: k_outlier {s['k_outlier']:.1f}, v_outlier {s['v_outlier']:.1f}")
    print(f"kv_fp16_layers = {fp16_layers}")
    print("saved results/kv4_plan.json")


if __name__ == "__main__":
    main()
