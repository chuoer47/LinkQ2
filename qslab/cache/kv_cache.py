"""KV cache: one interface, three precision implementations.

M0: FP16KVCache (plain contiguous buffer, append + read).
M2: KV8/KV4 quantized caches will follow the same interface so the attention
    replacement class needs no changes when precision switches.

Layout choice (M0): a single contiguous tensor [B, H, max_len, D] per K/V.
Paging can come later as v2 if needed; contiguous is the simplest correct
thing for decode-only, batch=1.
"""
from __future__ import annotations

import torch


class BaseKVCache:
    """Per-layer KV cache. All tensors are fp16 in M0."""

    def __init__(self, batch: int, num_kv_heads: int, head_dim: int,
                 max_len: int, device: str, dtype: torch.dtype = torch.float16):
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.k = torch.zeros(batch, num_kv_heads, max_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.len = 0  # current valid length

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor,
               start: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V of shape [B, H, T, D]; return full K/V slices."""
        T = k_new.shape[2]
        start = self.len if start is None else start
        self.k[:, :, start:start + T, :] = k_new.to(self.dtype)
        self.v[:, :, start:start + T, :] = v_new.to(self.dtype)
        if start + T > self.len:
            self.len = start + T
        return self.k[:, :, :self.len], self.v[:, :, :self.len]

    @property
    def memory_bytes(self) -> int:
        return 2 * self.k.nelement() * self.k.element_size()

    def memory_bytes_valid(self) -> int:
        return 2 * self.k[:, :, :self.len].nelement() * self.k.element_size()


class FP16KVCache(BaseKVCache):
    """M0 default: plain fp16 buffer."""


class KV8Cache(BaseKVCache):
    """Placeholder alias for M2 (int8 kv). Same interface."""


class KV4Cache(BaseKVCache):
    """Placeholder alias for M2 (4bit kv, asymmetric). Same interface."""
