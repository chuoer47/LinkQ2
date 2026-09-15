"""Quick diagnostic: per-layer quantization error, rtn vs rtn_clip vs awq.

Measures relative output MSE on real calibrated activations for a few
representative layers — fast iteration without full-model PPL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.model.loader import load_reference_model
from quantizer.calibrate import collect_activations
from quantizer.w4 import rtn_quantize_weight, clip_search_quantize, awq_find_scales
from quantizer.packfmt import unpack_w4

PROBE_LAYERS = ["model.layers.0.self_attn.q_proj",
                "model.layers.14.mlp.gate_proj",
                "model.layers.27.mlp.down_proj"]

blob = torch.load("results/frozen/calib_c4_128x2048.pt", weights_only=False)
calib_ids = blob["token_ids"][:16]  # small set for speed

model = load_reference_model("models/Qwen3-1.7B", device="cuda:0")
stats = collect_activations(model, calib_ids, "cuda:0")

# capture real inputs per probe layer
captured: dict[str, torch.Tensor] = {}
hooks = []
for name, mod in model.named_modules():
    if name in PROBE_LAYERS:
        def mk(nm):
            def hook(module, args, kwargs):
                x = args[0] if args else kwargs["input"]
                captured[nm] = x.detach()
            return hook
        hooks.append(mod.register_forward_pre_hook(mk(name), with_kwargs=True))
with torch.no_grad():
    for ids in calib_ids[:4]:
        model(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False)
for h in hooks:
    h.remove()

for name in PROBE_LAYERS:
    mod = dict(model.named_modules())[name]
    w = mod.weight.detach()
    x = captured[name].reshape(-1, w.shape[1]).to(torch.float16)[:2048]
    y_ref = (x @ w.T.to(torch.float16)).float()
    xam = stats[name]

    variants = {
        "rtn": rtn_quantize_weight(w, 128),
        "rtn_clip": clip_search_quantize(w, None, 128),
        "rtn_clip_x": clip_search_quantize(w, xam, 128),
    }
    s = awq_find_scales(w, xam, 128)
    w_scaled = (w.float() * s[None, :]).to(torch.float16)
    variants["awq"] = clip_search_quantize(w_scaled, xam, 128)

    print(f"\n== {name} ==")
    for vname, (qfp, scale, zero) in variants.items():
        w_hat = unpack_w4(qfp, scale, zero, w.shape[1], 128).to(x.device)
        if vname == "awq":
            # equivalent transform: y = (x/s) @ W'^T  ==  x @ W^T
            y = ((x.float() / s) @ w_hat.float().T)
        else:
            y = (x @ w_hat.to(torch.float16).T).float()
        rel = ((y - y_ref).pow(2).sum() / y_ref.pow(2).sum()).item()
        print(f"  {vname:12s} rel-MSE {rel:.6f}")
