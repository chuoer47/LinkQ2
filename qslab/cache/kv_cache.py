"""KV cache: one interface, three precision implementations.

M0: FP16KVCache (plain contiguous buffer, append + read).
M2: KV8Cache (int8, per-token symmetric) and KV4Cache (int4, K per-channel /
    V per-token per design-m2). Same update() interface returns fp16 K/V so
    attention code is precision-agnostic.

Layouts:
  FP16/KV8: K,V [B, H, max_len, D]
  KV4     : K stored TRANSPOSED as packed [B, H, D, max_len/8] uint32
            (per-channel quantization groups run along the token axis);
            V packed [B, H, max_len, D/8] uint32 (per-token groups along D).
  Scales stored fp16 alongside: K_scale [B, H, D, max_len/g],
  V_scale [B, H, max_len, D/g].
"""
from __future__ import annotations

import torch

PACK_G = 8  # int4 per uint32


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


def _quant_sym(x: torch.Tensor, g: int):
    """Symmetric group-wise quantize last dim by g: -> int4 in uint32 packs,
    fp16 scale. x [..., N] with N % g == 0."""
    shape = x.shape
    xg = x.to(torch.float32).reshape(*shape[:-1], shape[-1] // g, g)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 7.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(xg / scale), -8, 7).to(torch.int8)
    qn = (q & 0xF).to(torch.uint8).reshape(*shape[:-1], shape[-1])
    packs = torch.zeros(*shape[:-1], shape[-1] // PACK_G, dtype=torch.int64,
                        device=x.device)
    for nib in range(PACK_G):
        packs |= qn[..., nib::PACK_G].to(torch.int64) << (4 * nib)
    qfp = packs.to(torch.int32).view(torch.uint32)
    return qfp, scale.squeeze(-1).to(torch.float16)


def _dequant_sym(qfp: torch.Tensor, scale: torch.Tensor, g: int) -> torch.Tensor:
    shape = qfp.shape  # [..., N/8] uint32
    N = shape[-1] * PACK_G
    qi = qfp.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    qn = torch.zeros(*shape[:-1], N, dtype=torch.uint8, device=qfp.device)
    for nib in range(PACK_G):
        qn[..., nib::PACK_G] = ((qi >> (4 * nib)) & 0xF).to(torch.uint8)
    q = qn.reshape(*shape[:-1], N // g, g).to(torch.int16)
    q = torch.where(q >= 8, q - 16, q).to(torch.float32)
    return (q * scale.unsqueeze(-1)).reshape(*shape[:-1], N).to(torch.float16)


class KV8Cache(BaseKVCache):
    """int8 per-token symmetric (pipeline validation for KV4)."""

    def __init__(self, batch, num_kv_heads, head_dim, max_len, device,
                 group: int = 64):
        super().__init__(batch, num_kv_heads, head_dim, max_len, device)
        self.group = group
        self.k_q = torch.zeros(batch, num_kv_heads, max_len, head_dim,
                               device=device, dtype=torch.int8)
        self.k_s = torch.zeros(batch, num_kv_heads, max_len, head_dim // group,
                               device=device, dtype=torch.float16)
        self.v_q = self.k_q.clone()
        self.v_s = self.k_s.clone()

    def update(self, k_new, v_new, start=None):
        T = k_new.shape[2]
        start = self.len if start is None else start
        B_, H_, D_ = self.batch, self.num_kv_heads, self.head_dim
        # int8 per-token: per-group scale along D, vectorized over tokens
        for dst_q, dst_s, src in ((self.k_q, self.k_s, k_new),
                                  (self.v_q, self.v_s, v_new)):
            xg = src.to(torch.float32)                       # [B,H,T,D]
            xg = xg.view(B_, H_, T, self.head_dim // self.group, self.group)
            amax = xg.abs().amax(dim=-1, keepdim=True)
            s = (amax / 127.0).clamp_min(1e-12)
            q = torch.clamp(torch.round(xg / s), -127, 127).to(torch.int8)
            dst_q[:, :, start:start + T] = q.view(B_, H_, T, self.head_dim)
            dst_s[:, :, start:start + T] = s.squeeze(-1).to(torch.float16)
        if start + T > self.len:
            self.len = start + T
        B_, H_, D_ = self.batch, self.num_kv_heads, self.head_dim
        k = (self.k_q[:, :, :self.len].to(torch.float32)
             .view(B_, H_, self.len, D_ // self.group, self.group) *
             self.k_s[:, :, :self.len].unsqueeze(-1)).view(B_, H_, self.len, D_).to(torch.float16)
        v = (self.v_q[:, :, :self.len].to(torch.float32)
             .view(B_, H_, self.len, D_ // self.group, self.group) *
             self.v_s[:, :, :self.len].unsqueeze(-1)).view(B_, H_, self.len, D_).to(torch.float16)
        return k, v

    def memory_bytes_valid(self) -> int:
        n = self.k_q[:, :, :self.len].nelement()
        ns = self.k_s[:, :, :self.len].nelement()
        return 2 * (n * 1 + ns * 2)

    @property
    def memory_bytes(self) -> int:
        n = self.k_q.nelement()
        ns = self.k_s.nelement()
        return 2 * (n * 1 + ns * 2)


class KV4Cache(BaseKVCache):
    """int4: K per-channel (stored transposed), V per-token. g = group."""

    def __init__(self, batch, num_kv_heads, head_dim, max_len, device,
                 group: int = 64):
        super().__init__(batch, num_kv_heads, head_dim, max_len, device)
        self.group = group
        D, L = head_dim, max_len
        # K transposed: [B, H, D, L/8] uint32 packs + scale [B, H, D, L/g]
        self.k_q = torch.zeros(batch, num_kv_heads, D, L // PACK_G,
                               device=device, dtype=torch.uint32)
        self.k_s = torch.zeros(batch, num_kv_heads, D, L // group,
                               device=device, dtype=torch.float16)
        # V: [B, H, L, D/8] packs + scale [B, H, L, D/g]
        self.v_q = torch.zeros(batch, num_kv_heads, L, D // PACK_G,
                               device=device, dtype=torch.uint32)
        self.v_s = torch.zeros(batch, num_kv_heads, L, D // group,
                               device=device, dtype=torch.float16)

    def update(self, k_new, v_new, start=None):
        T = k_new.shape[2]
        start = self.len if start is None else start
        # K: quantize per-channel along token axis -> transpose to [B,H,D,T],
        # groups of `group` consecutive tokens share a scale
        kt = k_new.to(torch.float32).transpose(-1, -2)   # [B,H,D,T]
        assert T % self.group == 0, "T must be multiple of group for K"
        kq, ks = _quant_sym(kt, self.group)
        self.k_q[:, :, :, start // PACK_G:(start + T) // PACK_G] = kq
        self.k_s[:, :, :, start // self.group:(start + T) // self.group] = ks
        # V: per-token along D
        vq, vs = _quant_sym(v_new.to(torch.float32), self.group)
        self.v_q[:, :, start:start + T] = vq
        self.v_s[:, :, start:start + T] = vs
        if start + T > self.len:
            self.len = start + T
        L = ((self.len + self.group - 1) // self.group) * self.group
        k = _dequant_sym(self.k_q[:, :, :, :L // PACK_G],
                         self.k_s[:, :, :, :L // self.group], self.group)
        k = k.transpose(-1, -2)[..., :self.len, :]        # back to [B,H,T,D]
        v = _dequant_sym(self.v_q[:, :, :self.len],
                         self.v_s[:, :, :self.len], self.group)
        return k.contiguous(), v

    @property
    def memory_bytes(self) -> int:
        kq = self.k_q[:, :, :, :self.max_len // PACK_G].nelement() * 4
        ks = self.k_s.nelement() * 2
        vq = self.v_q[:, :, :self.max_len].nelement() * 4
        vs = self.v_s.nelement() * 2
        return kq + ks + vq + vs

    def memory_bytes_valid(self) -> int:
        L = ((self.len + self.group - 1) // self.group) * self.group
        kq = self.k_q[:, :, :, :L // PACK_G].nelement() * 4
        ks = self.k_s[:, :, :, :L // self.group].nelement() * 2
        vq = self.v_q[:, :, :self.len].nelement() * 4
        vs = self.v_s[:, :, :self.len].nelement() * 2
        return kq + ks + vq + vs
