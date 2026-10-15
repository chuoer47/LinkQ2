"""Online artifact loading and model replacement; no offline imports."""
import json
from pathlib import Path

from safetensors import safe_open
import torch

from .backends.quantization import make_quant_linear


class QuantizedWeights:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / "manifest.json").read_text())
        if self.config["version"] != 1:
            raise ValueError("unsupported artifact version")
        with safe_open(str(self.path / "tensors.safetensors"), framework="pt", device="cpu") as file:
            self.names = [key.removesuffix(".values") for key in file.keys() if key.endswith(".values")]

    def layer(self, name):
        with safe_open(str(self.path / "tensors.safetensors"), framework="pt", device="cpu") as file:
            keys = file.keys()
            return {"values": file.get_tensor(name + ".values"),
                    "scale": file.get_tensor(name + ".scale"),
                    "zero_point": None if self.config["symmetric"] else file.get_tensor(name + ".zero_point"),
                    "input_scale": file.get_tensor("transforms.input_scale." + name)
                        if "transforms.input_scale." + name in keys else None}


def replace_quantized_linears(model, path, *, backend="gemlite", device="cuda:0", dtype=torch.float16):
    weights = QuantizedWeights(path)
    for key in weights.names:
        name = key.removesuffix(".weight")
        old = model.get_submodule(name)
        data = weights.layer(key)
        if tuple(old.weight.shape) != tuple(data["values"].shape):
            raise ValueError(f"weight shape differs: {key}")
        if old.bias is not None:
            raise ValueError(f"biased quantized projections are not supported: {name}")
        # Release the source projection before allocating its packed replacement.
        old.weight = None
        layer = make_quant_linear(backend, **data,
            group_size=weights.config["group_size"], symmetric=weights.config["symmetric"],
            device=device, dtype=dtype)
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, layer)
    return len(weights.names)
