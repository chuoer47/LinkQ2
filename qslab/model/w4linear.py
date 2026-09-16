"""W4Linear: a Linear replacement that computes y = x @ dequant(W)^T.

Two backends:
  "v1"     — our custom w4a16 CUDA kernel (fast at M=1, degrades with M)
  "marlin" — vendored Marlin kernel (IST-DASLab, Apache 2.0) via v1-pack
             conversion; strong from M=8 up (GEMM tensor-core path)

`autodetect` dispatch (default): marlin when M>1, v1 when M==1 — per the
M5 benchmark table (results/m5_kernel_bench.json).
"""
from __future__ import annotations

import torch

from kernels.qslab_kernels.ops import w4a16_gemm


class W4Linear(torch.nn.Module):
    def __init__(self, qfp: torch.Tensor, scale: torch.Tensor,
                 group_size: int, in_features: int, out_features: int,
                 act_scale: torch.Tensor | None = None,
                 backend: str = "autodetect"):
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
        self.backend = backend
        self._marlin = None
        if backend in ("marlin", "autodetect"):
            try:
                from kernels.qslab_kernels.marlin_backend import pack_v1_to_marlin
                B, s = pack_v1_to_marlin(qfp, scale, in_features, group_size)
                self.register_buffer("marlin_B", B)   # int32 [I/16, O*2]
                self.register_buffer("marlin_s", s)   # fp16 [I/g, O]
                self.register_buffer("marlin_ws",
                                     torch.zeros(out_features // 128 * 16,
                                                 dtype=torch.int32))
                self._marlin_ok = (in_features % 128 == 0 and out_features % 256 == 0
                                   and group_size in (-1, 128))
            except Exception:
                self._marlin_ok = False
        else:
            self._marlin_ok = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_scale is not None:
            x = x / self.act_scale
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]) if x.dim() != 2 else x
        M = x2.shape[0]
        # autodetect: marlin from M>8 (true prefill; marlin's ~100us/call fixed
        # cost loses to v1 below that — spec-decode verify uses M<=gamma<=8)
        use_marlin = (self._marlin_ok and M > 8) if self.backend == "autodetect" \
            else self._marlin_ok
        if use_marlin:
            from kernels.qslab_kernels.marlin_backend import marlin_gemm
            y = marlin_gemm(x2, self.marlin_B, self.marlin_s, self.marlin_ws)
        else:
            y = w4a16_gemm(self.qfp, self.scale, x2, self.group_size)
        y = y.reshape(*orig_shape[:-1], self.out_features)
        return y.to(x.dtype)

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
        dev = mod.weight.device
        # free the fp16 weight BEFORE allocating the packed tensors — keeps
        # peak memory at (fp16 model - swapped weights + packed weights)
        mod.weight = None  # type: ignore[assignment]
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        qfp = st[f"{key}.qfp"].to(dev)
        scale = st[f"{key}.scale"].to(dev)
        act_s = None
        if key in awq_scales:
            act_s = torch.tensor(awq_scales[key], device=dev, dtype=torch.float16)
        w4 = W4Linear(qfp, scale, group, mod.in_features, mod.out_features,
                      act_scale=act_s)
        # splice into the parent module, then move ALL buffers (incl. marlin
        # B/s/workspace created on CPU in __init__) to the module's device
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], w4)
        w4.to(dev)
        count += 1
    return count
