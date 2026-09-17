"""L0: slot-mapped int4 paged KV — store (quantize+write) and decode kernels.

Layout (nano-vllm slot semantics; ONE slot per token, all heads inside):
  k_q [total_slots, H, D/8]  uint32
  k_s [total_slots, H]       fp16
  v_q [total_slots, H, D/8]  uint32
  v_s [total_slots, H]       fp16
Channel d of head h lives in word d//8 at nibble d%8 of row [slot, h].

slot_mapping semantics (from the model runner): slot = block_table[b] * BLOCK_N
+ token_offset_in_block; a token's H heads occupy rows [slot*H, slot*H+H).

Quantization: per-token per-head symmetric int4 (amax over D), scale floor
1e-4 (fp16 underflow guard), rounding floor(x+0.5).

The decode kernel indexes K/V rows directly at
  row = slot*H + kv_head,   word = d//8,   nibble = d%8
so the GQA mapping is a plain address offset (kv_head*NW) — no masked
selection needed.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_quant_kernel(
    k_ptr, v_ptr,               # [N, H, D] fp16 (pre-quantization)
    kq_ptr, ks_ptr, vq_ptr, vs_ptr,
    slot_ptr,                   # [N] int32: slot per token
    H: tl.constexpr, D: tl.constexpr, PACK_G: tl.constexpr = 8,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_ptr + idx)
    if slot == -1:
        return
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    NW: tl.constexpr = D // PACK_G
    nib_idx = offs_d % PACK_G

    k = tl.load(k_ptr + idx * H * D + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
    amax = tl.max(tl.abs(k), axis=1)
    scale = tl.maximum(amax / 7.0, 1e-4)
    qi = k / scale[:, None]
    qi = tl.minimum(tl.maximum(qi, -8.0), 7.0)
    qi = tl.floor(qi + 0.5).to(tl.int32)
    nib = (qi + 8) & 0xF
    nib_g = tl.reshape(nib, (H, NW, PACK_G))
    shifts = tl.arange(0, PACK_G)[None, None, :] * 4
    packed = tl.sum(nib_g << shifts, axis=2)
    tl.store(kq_ptr + slot * H * NW + offs_h[:, None] * NW + tl.arange(0, NW)[None, :],
             packed)
    tl.store(ks_ptr + slot * H + offs_h, scale.to(tl.float16))

    v = tl.load(v_ptr + idx * H * D + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
    amax = tl.max(tl.abs(v), axis=1)
    scale = tl.maximum(amax / 7.0, 1e-4)
    qi = v / scale[:, None]
    qi = tl.minimum(tl.maximum(qi, -8.0), 7.0)
    qi = tl.floor(qi + 0.5).to(tl.int32)
    nib = (qi + 8) & 0xF
    nib_g = tl.reshape(nib, (H, NW, PACK_G))
    packed = tl.sum(nib_g << shifts, axis=2)
    tl.store(vq_ptr + slot * H * NW + offs_h[:, None] * NW + tl.arange(0, NW)[None, :],
             packed)
    tl.store(vs_ptr + slot * H + offs_h, scale.to(tl.float16))


@triton.jit
def kv4_paged_decode_kernel(
    q_ptr, kq_ptr, ks_ptr, vq_ptr, vs_ptr, bt_ptr, out_ptr, context_len,
    N_Q_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, SCALE: tl.constexpr,
    PACK_G: tl.constexpr = 8,
):
    q_head = tl.program_id(0)
    kv_head = q_head // (N_Q_HEADS // N_KV_HEADS)
    n_blocks = tl.cdiv(context_len, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    NW: tl.constexpr = D // PACK_G

    q = tl.load(q_ptr + q_head * D + offs_d).to(tl.float32)

    # K row for (slot, kv_head) starts at slot*H*NW + kv_head*NW
    head_off = kv_head * NW

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for b in range(n_blocks):
        seq_slot = tl.load(bt_ptr + b) * BLOCK_N
        rows = seq_slot + offs_n                 # token slots in this block

        # K tile: [BN, D] via rows*H*NW + head_off + word
        kq = tl.load(kq_ptr + rows[:, None] * N_KV_HEADS * NW + head_off
                     + tl.arange(0, NW)[None, :])            # [BN, NW]
        ks = tl.load(ks_ptr + rows * N_KV_HEADS + kv_head).to(tl.float32)  # [BN]
        shifts = tl.arange(0, PACK_G)[None, :] * 4
        k_shifts = tl.arange(0, PACK_G)[None, None, :] * 4
        k_nib = ((kq[:, :, None] >> k_shifts) & 0xF).to(tl.int32)
        k_nib = tl.where(k_nib >= 8, k_nib - 16, k_nib).to(tl.float32)
        k_deq = tl.reshape(k_nib * ks[:, None, None], (BLOCK_N, D))

        s = tl.sum(q[None, :] * k_deq, axis=1) * SCALE
        pos = b * BLOCK_N + offs_n
        s = tl.where(pos < context_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        vq = tl.load(vq_ptr + rows[:, None] * N_KV_HEADS * NW + head_off
                     + tl.arange(0, NW)[None, :])
        vs = tl.load(vs_ptr + rows * N_KV_HEADS + kv_head).to(tl.float32)
        v_shifts = tl.arange(0, PACK_G)[None, None, :] * 4
        v_nib = ((vq[:, :, None] >> v_shifts) & 0xF).to(tl.int32)
        v_nib = tl.where(v_nib >= 8, v_nib - 16, v_nib).to(tl.float32)
        v_deq = tl.reshape(v_nib * vs[:, None, None], (BLOCK_N, D))

        acc = acc * alpha + tl.sum(p[:, None] * v_deq, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(out_ptr + q_head * D + offs_d, acc.to(tl.float16))


def paged_attention_decode(q: torch.Tensor, k_cache, v_cache,
                           block_table: torch.Tensor, context_len: int,
                           block_n: int = 128,
                           num_kv_heads: int | None = None) -> torch.Tensor:
    """q [N_Q_HEADS, D] fp16 -> out [N_Q_HEADS, D] fp16."""
    N_Q, D = q.shape
    n_kv = num_kv_heads if num_kv_heads is not None else N_Q
    out = torch.empty(N_Q, D, dtype=torch.float16, device=q.device)
    kq, ks = k_cache
    vq, vs = v_cache
    kv4_paged_decode_kernel[(N_Q,)](
        q, kq, ks, vq, vs, block_table, out, context_len,
        N_Q_HEADS=N_Q, N_KV_HEADS=n_kv, D=D,
        BLOCK_N=block_n, SCALE=1.0 / (D ** 0.5),
    )
    return out


def store_kv_quant(k: torch.Tensor, v: torch.Tensor,
                   k_cache, v_cache, slot_mapping: torch.Tensor):
    """k, v: [N, H, D] fp16. slot_mapping: [N] int32 (slot per token)."""
    N, H, D = k.shape
    kq, ks = k_cache
    vq, vs = v_cache
    store_kv_quant_kernel[(N,)](k, v, kq, ks, vq, vs,
                                slot_mapping.to(torch.int32), H=H, D=D)
