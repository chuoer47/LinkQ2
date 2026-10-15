"""Repack existing logical integers for GemLite; never quantize again."""
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from benchmarks.real_quant.model import target_linears, validate_targets
from qslab.artifacts.io import load_artifact
from benchmarks.quant_kernel.reference import dequantize


class FusedLinear(nn.Module):
    def __init__(self, weight, act, dtype, device, accumulation="fp32", dispatch="module"):
        super().__init__()
        import gemlite
        n, k = weight.values.shape
        self.in_features, self.out_features = k, n
        kind = gemlite.DType.FP16 if dtype == torch.float16 else gemlite.DType.BF16
        self.op = gemlite.GemLiteLinear(4, weight.group_size, k, n,
            input_dtype=kind, output_dtype=kind,
            acc_dtype=gemlite.DType.FP32 if accumulation == "fp32" else gemlite.DType.FP16)
        q = weight.values.to(device=device, dtype=torch.int16)
        if weight.symmetric:
            q = q + 8
            zero = 8
        else:
            zero = weight.zero_point.to(device)
        assert int(q.min()) >= 0 and int(q.max()) <= 15
        # fma_mode=False preserves (q-zero)*scale, without rounding zero*scale.
        self.op.pack(q.to(torch.uint8), weight.scale.to(device), zero,
                     fma_mode=False)
        self.register_buffer("act_scale", None if act is None else act.to(device=device, dtype=dtype))
        self.dispatch = dispatch
        # Benchmark-only adapter: packing and all kernels remain upstream code.
        # Tensors stay on their construction device for the lifetime of this run.
        self._tensor_args = tuple(self.op.get_tensor_args())
        self._meta_args = tuple(self.op.get_meta_args()[1:-1])
        self._routes = gemlite.core.GEMLITE_TRITON_MAPPING
        self._route_for_rows = gemlite.core.get_matmul_type
        input_type = self.op.input_dtype.value
        self._type_id = (1 if input_type == 2 else input_type) * 100 + 4

    def forward(self, x):
        if self.act_scale is not None:
            x = x / self.act_scale
        if self.dispatch == "module":
            return self.op(x)
        shape = x.shape[:-1] + (self.out_features,)
        x = x.contiguous().view(-1, self.in_features)
        kernel = self._routes[self._route_for_rows(x.shape[0], 4)]
        return kernel.forward(x, *self._tensor_args, None, *self._meta_args,
                              self.op.data_contiguous, self._type_id).view(shape)


@torch.no_grad()
def build_model(model_path, artifact_path, mode, dtype, device, accumulation="fp32", dispatch="module"):
    model = AutoModelForCausalLM.from_pretrained(str(model_path),
        torch_dtype=dtype, attn_implementation="eager").eval()
    if artifact_path is None:
        return model.to(device), []
    artifact = load_artifact(Path(artifact_path) / "logical")
    validate_targets(model, artifact.weights)
    scales = artifact.transforms.get("input_scale", {})
    targets = [(name, module.in_features, module.out_features)
               for name, module in target_linears(model).items()]
    layers = []
    for index, (name, k, n) in enumerate(targets):
        key = name + ".weight"
        weight, act = artifact.weights[key], scales.get(key)
        old = model.get_submodule(name)
        if old.bias is not None:
            raise ValueError("unexpected biased projection")
        if mode == "fake":
            # Fold activation scaling offline, matching the previous fake path.
            dense = dequantize(weight, torch.device("cpu"))
            if act is not None:
                dense = dense / act.float().reshape(1, -1)
            old.weight.copy_(dense.to(dtype))
            del dense
        else:
            replacement = FusedLinear(weight, act, dtype, device, accumulation, dispatch)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, replacement)
        layers.append({"name": name, "in_features": k, "out_features": n})
        del old
        if (index + 1) % 63 == 0:
            print(f"ASSEMBLY {mode} {index+1}/{len(targets)}", flush=True)
    del artifact
    model.to(device)
    if mode == "gemlite" and target_linears(model):
        raise AssertionError("dense target projections remain")
    return model.eval(), layers


def inventory(model):
    seen, counts = set(), {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        category = "packed" if name.endswith(".W_q") else (
            "quant_metadata" if any(name.endswith("."+s) for s in
                ("scales", "zeros", "metadata", "orig_shape", "act_scale")) else "dense")
        counts[category] = counts.get(category, 0) + storage.nbytes()
    return {**counts, "total_storage_bytes": sum(counts.values())}
