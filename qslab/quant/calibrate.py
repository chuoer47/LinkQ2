"""Activation statistics collector for calibration (docs/design-m1).

Hooks every target Linear in a loaded HF model, feeds frozen calibration
tokens, accumulates per-input-channel mean(|x|) and max(|x|) online.
No raw activations stored.
"""
from __future__ import annotations

from collections import defaultdict

import torch


@torch.no_grad()
def collect_activations(model, calib_ids: list[list[int]], device: str,
                        max_batches: int | None = None) -> dict[str, torch.Tensor]:
    """Returns {module_name: absmean fp32 [in_features]} over all Linear modules."""
    stats: dict[str, dict[str, torch.Tensor]] = defaultdict(
        lambda: {"sum": None, "count": 0})
    hooks = []

    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear)]
    for name, mod in linears:
        def make_hook(nm):
            def hook(module, args, kwargs):
                x = args[0] if args else kwargs["input"]
                xf = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
                s = xf.abs().sum(dim=0)
                st = stats[nm]
                if st["sum"] is None:
                    st["sum"] = s
                else:
                    st["sum"] += s
                st["count"] += xf.shape[0]
            return hook
        hooks.append(mod.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    n_batches = 0
    for ids in calib_ids:
        x = torch.tensor([ids], device=device)
        model(input_ids=x, use_cache=False)
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    for h in hooks:
        h.remove()

    out = {}
    for name, st in stats.items():
        assert st["sum"] is not None and st["count"] > 0
        out[name] = (st["sum"] / st["count"]).to(torch.float16)
    return out
