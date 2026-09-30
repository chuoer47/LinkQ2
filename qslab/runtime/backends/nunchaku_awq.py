"""Runtime adapter for Nunchaku's AWQ W4A16 GEMV operator."""
from __future__ import annotations

import torch


class NunchakuAWQLinear(torch.nn.Module):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor,
                 zeros: torch.Tensor, group_size: int, in_features: int,
                 out_features: int, act_scale: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if act_scale is None:
            self.act_scale = None
        else:
            self.register_buffer("act_scale", act_scale)
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from nunchaku._C import ops
            gemv_awq = ops.gemv_awq
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Nunchaku AWQ weights require the nunchaku package with its CUDA extension"
            ) from exc
        orig = x.shape
        x2 = x.reshape(-1, self.in_features)
        if self.act_scale is not None:
            x2 = x2 / self.act_scale
        # Nunchaku GEMV supports up to eight rows per call.
        chunks = []
        for start in range(0, x2.shape[0], 8):
            part = x2[start:start + 8].contiguous()
            chunks.append(gemv_awq(
                part, self.qweight, self.scales, self.zeros, part.shape[0],
                self.out_features, self.in_features, self.group_size))
        y = torch.cat(chunks, dim=0)
        return y.reshape(*orig[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group={self.group_size}, backend=nunchaku.awq")
