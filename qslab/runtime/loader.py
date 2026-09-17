"""qslab runtime weight loader — vendored from nano-vllm (MIT), adapted.

Differences from upstream:
- model construction happens in qslab (the runner builds Qwen3ForCausalLM
  from our runtime qwen3.py); this loader only fills weights.
- the packed_modules_mapping is read from the model class (our qwen3.py
  declares q/k/v -> QKVLinear shards and gate/up -> MergedLinear shards).
"""
from __future__ import annotations

import os
from glob import glob

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
