"""W4Linear: nn.Linear replacement backed by a pluggable QuantBackend."""
from __future__ import annotations

import torch


class W4Linear(torch.nn.Module):
    def __init__(self, qfp: torch.Tensor, scale: torch.Tensor,
                 group_size: int, in_features: int, out_features: int,
                 act_scale: torch.Tensor | None = None,
                 backend: str = "w4.auto"):
        super().__init__()
        if act_scale is not None:
            self.register_buffer("act_scale", act_scale)
        else:
            self.act_scale = None
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.backend_name = backend

        # importing this module registers all built-in backends
        from qslab.runtime.backends.w4 import get_backend  # noqa: F401
        self._backend = get_backend(
            backend, qfp, scale, group_size, in_features, out_features,
            act_scale=(self.act_scale if act_scale is not None else None))

        # keep the packed tensors as buffers only while a kernel reads them:
        # the Marlin repack replaces the v1 pack, and registering it anyway
        # pinned a second full int4 copy per layer
        if self._backend.uses_v1_pack:
            self.register_buffer("qfp", qfp)        # [O, I/8] uint32
            self.register_buffer("scale", scale)    # [O, I/g] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backend.linear(x)

    def to(self, *args, **kwargs):
        # backend may hold tensors outside the module (marlin B/s/workspace)
        super().to(*args, **kwargs)
        dev = self._target_device(args, kwargs)
        to_fn = getattr(self._backend, "to", None)
        if dev is not None and to_fn is not None:
            to_fn(dev)
        return self

    def _target_device(self, args, kwargs):
        # after super().to() a buffer's device *is* the destination; a layer
        # that released every buffer has to read it from the call instead
        for buf in self.buffers():
            return buf.device
        dev = kwargs.get("device")
        if dev is None:
            dev = next((a for a in args if isinstance(a, (torch.device, str))),
                       None)
        return None if dev is None else torch.device(dev)

    def weight_memory_bytes(self) -> int:
        return self._backend.memory_bytes()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"g={self.group_size}, backend={self.backend_name}")
