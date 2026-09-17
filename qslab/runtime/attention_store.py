"""L0: slot-mapped int4 KV store — quantize one token's K/V into its slot.

Per-token symmetric quantization (amax over D). Grid = (tokens*heads, words):
each program quantizes one (token, head) pair's D channels and packs them
into D/8 uint32 words written to the token's slot.

Write counterpart of runtime/attention.py's read kernel. Layout per slot:
k_q [D/8] uint32 + k_s fp16 (same for V); channel d lives in word d//8 at
nibble d%8.

Quantization scheme: per-token (amax over the whole head dim) for both K and
V — per-token is the only scheme whose write is independent of other tokens,
which slot-mapped storage requires (see design-m8.md, "per-token" decision).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_quant_kernel(
    k_ptr, v_ptr,               # [N*H, D] fp16 flattened
    kq_ptr, ks_ptr, vq_ptr, vs_ptr,   # paged buffers for this layer
    slot_ptr,                   # [N*H] int32: slot per (token, head) pair
    D: tl.constexpr, PACK_G: tl.constexpr = 8,
):
    pid = tl.program_id(0)                      # one program per (token, head)
    slot = tl.load(slot_ptr + pid)
    if slot == -1:
        return
    offs_d = tl.arange(0, D)
    word_idx = offs_d // PACK_G                 # [D]
    nib_idx = offs_d % PACK_G                   # [D]
    NW: tl.constexpr = D // PACK_G

    # ---- K: quantize then pack ----
    k = tl.load(k_ptr + pid * D + offs_d).to(tl.float32)
    amax = tl.max(tl.abs(k), axis=0)
    scale = tl.maximum(amax / 7.0, 1e-4)
    qi = k / scale
    qi = tl.minimum(tl.maximum(qi, -8.0), 7.0).to(tl.int32)
    nib = ((qi + 8) & 0xF).to(tl.int32)                          # [0,15]
    # one-hot scatter: each channel's nibble lands in word w at nibble i
    packed = tl.sum(
        tl.where((word_idx[:, None] == tl.arange(0, NW)[None, :])
                 & (nib_idx[:, None] == tl.arange(0, PACK_G)[None, :]),
                 nib[:, None] << (nib_idx[:, None] * 4), 0), axis=0)   # [NW]
    tl.store(kq_ptr + slot * NW + tl.arange(0, NW), packed)
    tl.store(ks_ptr + slot, scale.to(tl.float16))

    # ---- V ----
    v = tl.load(v_ptr + pid * D + offs_d).to(tl.float32)
    amax = tl.max(tl.abs(v), axis=0)
    scale = tl.maximum(amax / 7.0, 1e-4)
    qi = v / scale
    qi = tl.minimum(tl.maximum(qi, -8.0), 7.0).to(tl.int32)
    nib = ((qi + 8) & 0xF).to(tl.int32)
    packed = tl.sum(
        tl.where((word_idx[:, None] == tl.arange(0, NW)[None, :])
                 & (nib_idx[:, None] == tl.arange(0, PACK_G)[None, :]),
                 nib[:, None] << (nib_idx[:, None] * 4), 0), axis=0)
    tl.store(vq_ptr + slot * NW + tl.arange(0, NW), packed)
    tl.store(vs_ptr + slot, scale.to(tl.float16))


def store_kv_quant(k: torch.Tensor, v: torch.Tensor,
                   k_cache, v_cache, slot_mapping: torch.Tensor):
    """k, v: [N, H, D] fp16. slot_mapping: [N, H] int32 (slot per token,head).

    Slot layout: slot = seq_slot_base * N_KV_HEADS + kv_head, computed by the
    caller (model_runner) from the block table.
    """
    N, H, D = k.shape
    kq, ks = k_cache
    vq, vs = v_cache
    k_flat = k.reshape(N * H, D).contiguous()
    v_flat = v.reshape(N * H, D).contiguous()
    slots = slot_mapping.reshape(N * H).contiguous()
    store_kv_quant_kernel[(N * H,)](k_flat, v_flat, kq, ks, vq, vs, slots, D=D)
