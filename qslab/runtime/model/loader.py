"""qslab runtime weight loader — vendored from nano-vllm (MIT), adapted.

Differences from upstream:
- model construction happens in qslab (the runner builds Qwen3ForCausalLM
  from our runtime qwen3.py); this loader only fills weights.
- `swap_w4` replaces the fp16 Linear layers with L1's packed W4Linear, so a
  qslab_w4_v1 checkpoint drives the runtime end to end.

The checkpoints are keyed by HF module name and the runtime model keeps the
projections separate for exactly that reason (see qwen3.py), so the swap is a
direct name match — no shard mapping, no repacking.
"""
from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def swap_w4(model: nn.Module, packed_dir: str,
            backend: str = "w4.auto") -> int:
    """Replace runtime Linear layers with packed W4Linear ones, in place.

    `packed_dir` is a qslab_w4_v1 directory (safetensors + config.json, and
    awq_scales.json when the checkpoint used AWQ). Modules without packed
    weights — embeddings, norms, an untied lm_head — are left in fp16, which
    is what the checkpoints expect.
    """
    from qslab.models.w4linear import W4Linear
    from qslab.quant.packfmt import load_qslab_w4
    from qslab.runtime.model.primitives import Linear as RuntimeLinear

    packed_dir = Path(packed_dir)
    config, st, _calib = load_qslab_w4(packed_dir)
    group = config["group_size"]

    awq_scales = {}
    awq_file = packed_dir / "awq_scales.json"
    if config.get("algo") == "awq" and awq_file.exists():
        awq_scales = json.loads(awq_file.read_text())

    # free the fp16 weights before allocating the packed ones, so the peak
    # stays near (fp16 model - swapped + packed) instead of the sum
    targets = []
    for name, mod in model.named_modules():
        key = f"{name}.weight"
        if isinstance(mod, RuntimeLinear) and f"{key}.qfp" in st:
            targets.append((name, mod, key))
    if not targets:
        return 0
    dev = targets[0][1].weight.device
    for _, mod, _ in targets:
        mod.weight = None          # type: ignore[assignment]
    torch.cuda.empty_cache()

    for name, mod, key in targets:
        qfp = st[f"{key}.qfp"].to(dev)
        scale = st[f"{key}.scale"].to(dev)
        act = None
        if key in awq_scales:
            act = torch.tensor(awq_scales[key], device=dev, dtype=torch.float16)
        w4 = W4Linear(qfp, scale, group, mod.input_size, mod.output_size,
                      act_scale=act, backend=backend)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], w4)
        w4.to(dev)

    return len(targets)
