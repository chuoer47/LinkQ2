"""Offline artifact writer and reader. The online reader is independent."""
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from .schema import CompressionArtifact, QuantizedTensor


def save_artifact(artifact, path):
    artifact.validate()
    settings = {(w.group_size, w.symmetric) for w in artifact.weights.values()}
    if len(settings) != 1:
        raise ValueError("one artifact uses one group size and symmetry setting")
    group_size, symmetric = settings.pop()
    tensors = {}
    for name, weight in artifact.weights.items():
        tensors[name + ".values"] = weight.values.to(device="cpu", dtype=torch.int8).contiguous()
        tensors[name + ".scale"] = weight.scale.cpu().contiguous()
        if not symmetric:
            tensors[name + ".zero_point"] = weight.zero_point.cpu().contiguous()
    for name, tensor in artifact.transforms.get("input_scale", {}).items():
        tensors["transforms.input_scale." + name] = tensor.cpu().contiguous()
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "tensors.safetensors"))
    (root / "manifest.json").write_text(json.dumps({"version": 1,
        "group_size": group_size, "symmetric": symmetric}, indent=2))


def load_artifact(path):
    root = Path(path)
    config = json.loads((root / "manifest.json").read_text())
    if config["version"] != 1:
        raise ValueError("unsupported artifact version")
    tensors = load_file(str(root / "tensors.safetensors"))
    weights, input_scales = {}, {}
    for key, values in tensors.items():
        if key.endswith(".values"):
            name = key.removesuffix(".values")
            scale = tensors[name + ".scale"]
            zero = torch.zeros_like(scale) if config["symmetric"] else tensors[name + ".zero_point"]
            weights[name] = QuantizedTensor(values, scale, zero, config["group_size"], config["symmetric"])
        elif key.startswith("transforms.input_scale."):
            input_scales[key.removeprefix("transforms.input_scale.")] = values
    return CompressionArtifact(weights, {"input_scale": input_scales} if input_scales else {})
