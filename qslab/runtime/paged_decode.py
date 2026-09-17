"""L0: slot-mapped int4 paged attention decode kernel (per-token layout).

Read counterpart of runtime/attention_store.py's write kernel. Layout per
slot (one token, one kv head): k_q [D/8] uint32 + k_s fp16; channel d lives
in word d//8 at nibble d%8. Slot layout: slot = seq_slot * N_KV_HEADS + kv_head.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kv4_paged_decode_kernel(
    q_ptr,            # [N_Q_HEADS, D] fp16 (one query token)
    kq_ptr, ks_ptr,   # [total_slots, D/8] uint32, [total_slots] fp16
    vq_ptr, vs_ptr,
    bt_ptr,           # [num_blocks_used] int32: logical block -> first seq slot
    out_ptr,          # [N_Q_HEADS, D] fp16
    context_len,
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
    # GQA: the kv heads for one logical block live in a contiguous span of
    # N_KV_HEADS slots after the seq's BLOCK_N token slots.
    SLOTS_PER_BLOCK: tl.constexpr = BLOCK_N * N_KV_HEADS

    q = tl.load(q_ptr + q_head * D + offs_d).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    word_idx = offs_d // PACK_G
    nib_idx = offs_d % PACK_G

    for b in range(n_blocks):
        seq_base = tl.load(bt_ptr + b) * SLOTS_PER_BLOCK
        # this kv head's tokens inside the block
        slots = seq_base + offs_n * N_KV_HEADS + kv_head

        kq = tl.load(kq_ptr + slots[:, None] * NW + word_idx[None, :])  # [BN, D]
        ks = tl.load(ks_ptr + slots).to(tl.float32)                     # [BN]
        k_nib = ((kq >> (nib_idx[None, :] * 4)) & 0xF).to(tl.int32)
        k_nib = tl.where(k_nib >= 8, k_nib - 16, k_nib).to(tl.float32)
        k_deq = k_nib * ks[:, None]                                     # [BN, D]

        s = tl.sum(q[None, :] * k_deq, axis=1) * SCALE
        pos = b * BLOCK_N + offs_n
        s = tl.where(pos < context_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        vq = tl.load(vq_ptr + slots[:, None] * NW + word_idx[None, :])
        vs = tl.load(vs_ptr + slots).to(tl.float32)
        v_nib = ((vq >> (nib_idx[None, :] * 4)) & 0xF).to(tl.int32)
        v_nib = tl.where(v_nib >= 8, v_nib - 16, v_nib).to(tl.float32)
        v_deq = v_nib * vs[:, None]

        acc = acc * alpha + tl.sum(p[:, None] * v_deq, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(out_ptr + q_head * D + offs_d, acc.to(tl.float16))


def paged_attention_decode(q: torch.Tensor, k_cache, v_cache,
                           block_table: torch.Tensor, context_len: int,
                           block_n: int = 128,
                           num_kv_heads: int | None = None) -> torch.Tensor:
    """q [N_Q_HEADS, D] fp16 -> out [N_Q_HEADS, D] fp16.

    k_cache/v_cache are (packed_u32, scales_fp16) tuples for ONE layer,
    laid out as [total_slots, D/8] / [total_slots].
    """
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
