"""W4Linear: a Linear replacement backed by a pluggable QuantBackend (L1).

L2 note: this module MUST NOT import qslab.kernels (L0) directly. All kernel
access goes through the L1 backend returned by ``quant.get_backend`` — that
is the layering rule established in docs/design-r1.md and the reason the
per-M dispatch now lives in the backend instead of here.

Backend names (see qslab/quant/w4_backends.py):
  "w4.v1"     — custom GEMV kernel (M=1 champion)
  "w4.marlin" — Marlin tensor-core GEMM (M>=8 champion)
  "w4.auto"   — hybrid: switches at M>8 using the M5 benchmark crossover
"""
from __future__ import annotations

import torch


class W4Linear(torch.nn.Module):
    def __init__(self, qfp: torch.Tensor, scale: torch.Tensor,
                 group_size: int, in_features: int, out_features: int,
                 act_scale: torch.Tensor | None = None,
                 backend: str = "w4.auto"):
        super().__init__()
        # keep packed tensors as buffers so .to(device) moves them
        self.register_buffer("qfp", qfp)          # [O, I/8] uint32
        self.register_buffer("scale", scale)      # [O, I/g] fp16
        if act_scale is not None:
            self.register_buffer("act_scale", act_scale)
        else:
            self.act_scale = None
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.backend_name = backend

        # importing this module registers all built-in backends
        from qslab.quant.w4_backends import get_backend  # noqa: F401
        self._backend = get_backend(
            backend, qfp, scale, group_size, in_features, out_features,
            act_scale=(self.act_scale if act_scale is not None else None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.linear(x)

    def to(self, *args, **kwargs):
        # backend may hold tensors outside the module (marlin B/s/workspace)
        super().to(*args, **kwargs)
        dev = next(iter(self.buffers())).device
        to_fn = getattr(self._backend, "to", None)
        if to_fn is not None:
            to_fn(dev)
        return self

    def weight_memory_bytes(self) -> int:
        return self._backend.memory_bytes()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"g={self.group_size}, backend={self.backend_name}")


def swap_w4_linears(model: torch.nn.Module, packed_dir: str,
                    backend: str = "w4.auto") -> int:
    """Replace quantized Linears with W4Linear using the packed checkpoint.

    Mirrors load_w4_model's weight mapping but keeps weights packed.
    """
    import json
    from pathlib import Path
    from qslab.quant.packfmt import load_qslab_w4

    config, st, _calib = load_qslab_w4(packed_dir)
    group = config["group_size"]
    awq_scales = {}
    if config["algo"] == "awq":
        awq_file = Path(packed_dir) / "awq_scales.json"
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
        # free the fp16 weight BEFORE allocating packed tensors — keeps peak
        # memory at (fp16 model - swapped weights + packed weights)
        mod.weight = None  # type: ignore[assignment]
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        qfp = st[f"{key}.qfp"].to(dev)
        scale = st[f"{key}.scale"].to(dev)
        act_s = None
        if key in awq_scales:
            act_s = torch.tensor(awq_scales[key], device=dev, dtype=torch.float16)
        w4 = W4Linear(qfp, scale, group, mod.in_features, mod.out_features,
                      act_scale=act_s, backend=backend)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], w4)
        w4.to(dev)
        count += 1
    return count
