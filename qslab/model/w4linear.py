"""W4Linear: a Linear replacement that computes y = x @ dequant(W)^T
using the qslab w4a16 CUDA kernel, reading packed weights directly.

Used by the engine when a model is loaded from a qslab_w4_v1 directory:
quantized Linears become W4Linear(qfp, scale, group_size) instances; unquantized
Linears (embed, lm_head, norms) stay untouched.
"""
from __future__ import annotations

import torch

from kernels.qslab_kernels.ops import w4a16_gemm


class W4Linear(torch.nn.Module):
    def __init__(self, qfp: torch.Tensor, scale: torch.Tensor,
                 group_size: int, in_features: int, out_features: int,
                 act_scale: torch.Tensor | None = None):
        super().__init__()
        # keep packed tensors as buffers (move with .to(device))
        self.register_buffer("qfp", qfp)          # [O, I/8] uint32
        self.register_buffer("scale", scale)      # [O, I/g] fp16
        if act_scale is not None:
            self.register_buffer("act_scale", act_scale)
        else:
            self.act_scale = None
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_scale is not None:
            x = x / self.act_scale
        # fast path: 2D input (decode GEMV) — no reshape, kernel returns fp32
        # and we cast once; contiguous row-major x hits the kernel directly.
        if x.dim() == 2:
            y = w4a16_gemm(self.qfp, self.scale, x, self.group_size)
            return y.to(x.dtype)
        orig = x.shape
        y = w4a16_gemm(self.qfp, self.scale, x.reshape(-1, orig[-1]), self.group_size)
        return y.reshape(*orig[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, g={self.group_size}"


def swap_w4_linears(model: torch.nn.Module, packed_dir: str) -> int:
    """Replace quantized Linears with W4Linear using the packed checkpoint.
    Mirrors load_w4_model's weight mapping but keeps weights packed."""
    import json
    from quantizer.packfmt import load_qslab_w4

    config, st, _calib = load_qslab_w4(packed_dir)
    group = config["group_size"]
    awq_scales = {}
    if config["algo"] == "awq":
        awq_file = Path_ = __import__("pathlib").Path(packed_dir) / "awq_scales.json"
        if awq_file.exists():
            awq_scales = json.loads(awq_file.read_text())

    count = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if f"{key}.qfp" not in st:
            continue
        qfp = st[f"{key}.qfp"].to(mod.weight.device)
        scale = st[f"{key}.scale"].to(mod.weight.device)
        act_s = None
        if key in awq_scales:
            act_s = torch.tensor(awq_scales[key], device=mod.weight.device,
                                 dtype=torch.float16)
        w4 = W4Linear(qfp, scale, group, mod.in_features, mod.out_features,
                      act_scale=act_s)
        # splice into the parent module
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], w4)
        count += 1
    return count
