"""HF model assembly and packed Linear replacement, independent of the engine."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from benchmarks.real_quant.config import TARGET_SUFFIXES
from qslab.artifacts.io import load_artifact
from qslab.conversion.qslab_w4 import load_qslab_w4
from qslab.conversion.nunchaku_awq.format import load_nunchaku_awq
from qslab.models.w4linear import W4Linear
from qslab.runtime.backends.nunchaku_awq import NunchakuAWQLinear


def load_hf(model_path):
    return AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16,
        attn_implementation="eager").eval()


def target_linears(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)}


def validate_targets(model, names):
    expected = {name + ".weight" for name in target_linears(model)}
    actual = set(names)
    if actual != expected:
        raise ValueError(f"quantized layer coverage mismatch: missing={sorted(expected-actual)}, "
                         f"extra={sorted(actual-expected)}")
    return len(expected)


def restore_weight(weight, act_scale=None):
    n, k = weight.values.shape
    values = weight.values.float().reshape(n, -1, weight.group_size)
    scales = weight.scale.float().unsqueeze(-1)
    restored = values * scales if weight.symmetric else (
        values - weight.zero_point.float().unsqueeze(-1)) * scales
    restored = restored.reshape(n, k)
    if act_scale is not None:
        restored = restored / act_scale.float().reshape(1, -1)
    return restored.to(torch.float16)


@torch.no_grad()
def install_fake(model, artifact):
    count = validate_targets(model, artifact.weights)
    scales = artifact.transforms.get("input_scale", {})
    for name, module in target_linears(model).items():
        key = name + ".weight"
        if tuple(artifact.weights[key].values.shape) != tuple(module.weight.shape):
            raise ValueError(f"weight shape mismatch: {key}")
        module.weight.copy_(restore_weight(artifact.weights[key], scales.get(key)))
    return count


def make_packed_linear(backend, state, key, group, in_features, out_features, device):
    if backend == "nunchaku_awq":
        act = state.get(key + ".act_scale")
        return NunchakuAWQLinear(
            state[key + ".qweight"].to(device), state[key + ".scales"].to(device),
            state[key + ".zeros"].to(device), group, in_features, out_features,
            None if act is None else act.to(device))
    act = state.get(key + ".act_scale")
    layer = W4Linear(state[key + ".qfp"].to(device),
                     state[key + ".scale"].to(device), group,
                     in_features, out_features,
                     None if act is None else act.to(device),
                     backend={"qslab_w4a16": "w4.v1", "marlin": "w4.marlin"}[backend])
    # Marlin holds its repack/workspace outside registered module buffers.
    return layer.to(device)


@torch.no_grad()
def install_packed(model, packed_dir, backend, device, factory=make_packed_linear):
    packed_dir = Path(packed_dir)
    if backend == "nunchaku_awq":
        meta, state = load_nunchaku_awq(packed_dir)
    else:
        meta, state, _ = load_qslab_w4(packed_dir)
        scale_path = packed_dir / "awq_scales.json"
        if scale_path.exists():
            for key, value in json.loads(scale_path.read_text()).items():
                state[key + ".act_scale"] = torch.tensor(value, dtype=torch.float16)
    count = validate_targets(model, meta["quantized_layers"])
    # Store names and dimensions, not the original modules or their FP16 weights.
    targets = [(name, mod.in_features, mod.out_features) for name, mod in
               target_linears(model).items()]
    layers = []
    for name, input_size, output_size in targets:
        old = model.get_submodule(name)
        if old.bias is not None:
            raise ValueError(f"this Qwen3 benchmark expects bias-free projections: {name}")
        key = name + ".weight"
        group = int(meta["group_size"])
        if backend == "nunchaku_awq":
            shape = (output_size // 4, input_size)
            tensor = state[key + ".qweight"]
        else:
            shape = (output_size, input_size // 8)
            tensor = state[key + ".qfp"]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"packed shape mismatch: {key}, {tensor.shape} vs {shape}")
        if backend == "marlin" and (input_size % 128 or output_size % 256 or group != 128):
            raise ValueError(f"unsupported Marlin shape/group: {name}")
        layer = factory(backend, state, key, group, input_size, output_size, device)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, layer)
        del old
        layers.append({"name": name, "backend": backend,
                       "in_features": input_size, "out_features": output_size})
    del state
    model.to(device)
    if target_linears(model):
        raise AssertionError("FP16 target Linear remained after replacement")
    return count, layers


def load_model(model_path, phase, backend, artifact_dir, device):
    model = load_hf(model_path)
    count, layers = 0, []
    if phase == "reference" and artifact_dir is not None:
        artifact = load_artifact(Path(artifact_dir) / "logical")
        count = install_fake(model, artifact)
        del artifact
        model.to(device)
    elif backend is not None:
        fmt = "nunchaku_awq" if backend == "nunchaku_awq" else "qslab_w4"
        count, layers = install_packed(model, Path(artifact_dir) / fmt, backend, device)
    else:
        model.to(device)
    return model.eval(), count, layers


def memory_inventory(model):
    seen = set()

    def size(tensors):
        total = 0
        for tensor in tensors:
            if tensor is None or tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    fp16 = size(model.parameters())
    packed, scales, workspace = 0, 0, 0
    for module in model.modules():
        if isinstance(module, W4Linear):
            backend = module._backend
            packed += size([getattr(backend, "qfp", None), getattr(backend, "_B", None)])
            scales += size([getattr(backend, "scale", None), getattr(backend, "_s", None),
                            module.act_scale])
            workspace += size([getattr(backend, "_ws", None)])
        elif isinstance(module, NunchakuAWQLinear):
            packed += size([module.qweight])
            scales += size([module.scales, module.zeros, module.act_scale])
    return {"fp16_parameter_bytes": fp16, "packed_weight_bytes": packed,
            "quantization_parameter_bytes": scales, "workspace_bytes": workspace,
            "resident_weight_bytes": fp16 + packed + scales,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved()}
