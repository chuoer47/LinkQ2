"""L0: quantized paged attention (Triton) — int4 KV read directly in-kernel.

Decode-only, M=1: one program per (query token, kv head). Streams over the
context one block at a time, dequantizing each K/V tile on the fly, so no
dense FP16 copy of the cache is ever materialized.

Layout (see docs/design-m7.md) — block size is a multiple of the quant group
so a group never spans two blocks:
  k_q [nb, H, D, BLOCK_N/8]   uint32   K per-channel (channel-major, token packs)
  k_s [nb, H, D, BLOCK_N/G]   fp16
  v_q [nb, H, BLOCK_N, D/8]   uint32   V per-token (token-major, channel packs)
  v_s [nb, H, BLOCK_N, D/G]   fp16
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def kv4_paged_attention_kernel(
    q_ptr, kq_ptr, ks_ptr, vq_ptr, vs_ptr, bt_ptr, out_ptr, context_len,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, GROUP: tl.constexpr, SCALE: tl.constexpr,
    PACK_G: tl.constexpr = 8,
):
    head = tl.program_id(0)
    n_blocks = tl.cdiv(context_len, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    n_words_k: tl.constexpr = BLOCK_N // PACK_G      # K: words per channel
    n_groups_k: tl.constexpr = BLOCK_N // GROUP
    n_words_v: tl.constexpr = D // PACK_G            # V: words per token
    n_groups_v: tl.constexpr = D // GROUP

    q = tl.load(q_ptr + head * D + offs_d).to(tl.float32)      # [D]

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    k_base = head * D * n_words_k
    ks_base = head * D * n_groups_k
    v_base = head * BLOCK_N * n_words_v
    vs_base = head * BLOCK_N * n_groups_v

    for b in range(n_blocks):
        blk = tl.load(bt_ptr + b)
        s_base = blk * H * D * n_words_k
        ss_base = blk * H * D * n_groups_k

        # ---- K tile: [D, BLOCK_N], dequantized word-by-word ----
        # Load all K words as [D, BLOCK_N/8] and expand to [D, BLOCK_N].
        kq = tl.load(kq_ptr + s_base + k_base
                     + offs_d[:, None] * n_words_k + tl.arange(0, n_words_k)[None, :])
        # scales: one per (channel, group), broadcast across the group's tokens
        ks = tl.load(ks_ptr + ss_base + ks_base
                     + offs_d[:, None] * n_groups_k + tl.arange(0, n_groups_k)[None, :])
        # expand [D, n_groups] -> [D, BLOCK_N] along the token axis
        ks_full = tl.reshape(
            tl.broadcast_to(ks[:, :, None], (D, n_groups_k, GROUP)),
            (D, BLOCK_N))
        # expand words [D, nw] -> [D, BLOCK_N]: word w covers tokens [w*8, w*8+8)
        kq_full = tl.reshape(
            tl.broadcast_to(kq[:, :, None], (D, n_words_k, PACK_G)),
            (D, BLOCK_N))
        # nibble index within each word, tiled along tokens
        nib_idx = (offs_n % PACK_G)[None, :]
        shift = nib_idx * 4
        # to int32 BEFORE subtracting: uint32 underflow wraps to ~4e9
        k_nib = ((kq_full >> shift) & 0xF).to(tl.int32)
        k_nib = tl.where(k_nib >= 8, k_nib - 16, k_nib).to(tl.float32)
        k_deq = k_nib * ks_full.to(tl.float32)                  # [D, BLOCK_N]

        s = tl.sum(q[:, None] * k_deq, axis=0) * SCALE          # [BLOCK_N]
        pos = b * BLOCK_N + offs_n
        s = tl.where(pos < context_len, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # ---- V tile: [BLOCK_N, D], dequantized word-by-word ----
        v_base_b = blk * H * BLOCK_N * n_words_v
        vs_base_b = blk * H * BLOCK_N * n_groups_v
        vq = tl.load(vq_ptr + v_base_b + v_base
                     + offs_n[:, None] * n_words_v + tl.arange(0, n_words_v)[None, :])
        vs = tl.load(vs_ptr + vs_base_b + vs_base
                     + offs_n[:, None] * n_groups_v + tl.arange(0, n_groups_v)[None, :])
        vs_full = tl.reshape(
            tl.broadcast_to(vs[:, :, None], (BLOCK_N, n_groups_v, GROUP)),
            (BLOCK_N, D))
        vq_full = tl.reshape(
            tl.broadcast_to(vq[:, :, None], (BLOCK_N, n_words_v, PACK_G)),
            (BLOCK_N, D))
        nib_idx_v = (offs_d % PACK_G)[None, :]
        v_nib = ((vq_full >> (nib_idx_v * 4)) & 0xF).to(tl.int32)
        v_nib = tl.where(v_nib >= 8, v_nib - 16, v_nib).to(tl.float32)
        v_deq = v_nib * vs_full.to(tl.float32)                  # [BLOCK_N, D]

        acc = acc * alpha + tl.sum(p[:, None] * v_deq, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(out_ptr + head * D + offs_d, acc.to(tl.float16))


def kv4_paged_attention(q: torch.Tensor, k_cache, v_cache,
                        block_table: torch.Tensor, context_len: int,
                        block_n: int = 128, group: int = 64) -> torch.Tensor:
    """q [H, D] fp16 -> out [H, D] fp16. Caches are (packed, scales) tuples."""
    H, D = q.shape
    out = torch.empty(H, D, dtype=torch.float16, device=q.device)
    kq, ks = k_cache
    vq, vs = v_cache
    kv4_paged_attention_kernel[(H,)](
        q, kq, ks, vq, vs, block_table, out, context_len,
        H=H, D=D, BLOCK_N=block_n, GROUP=group, SCALE=1.0 / (D ** 0.5),
        PACK_G=8,
    )
    return out
