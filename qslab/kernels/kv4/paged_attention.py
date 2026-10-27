"""Int4 paged KV Triton operators."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .grouped_attention import kv4_grouped_verify_kernel

@triton.jit
def kv4_paged_decode_kernel(
    q_ptr, kq_ptr, ks_ptr, vq_ptr, vs_ptr, bt_ptr, out_ptr, context_lens_ptr,
    N_Q_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, PAGE: tl.constexpr, MAX_BLOCKS: tl.constexpr, SCALE: tl.constexpr,
    SPLITS: tl.constexpr, Q_STRIDE: tl.constexpr,
    STATIC_K: tl.constexpr, M: tl.constexpr = 1,
    PACK_G: tl.constexpr = 8, V_GROUP: tl.constexpr = 64,
):
    q_head = tl.program_id(0)
    row = tl.program_id(1)               # row = seq * M + m (M=1: ordinary decode)
    split = tl.program_id(2)
    seq = row // M
    m = row - seq * M
    kv_head = q_head // (N_Q_HEADS // N_KV_HEADS)
    context_len = tl.load(context_lens_ptr + seq) + m   # reads keys [0, L+m)
    n_tiles = tl.cdiv(tl.minimum(context_len, MAX_BLOCKS * PAGE), BLOCK_N)
    per_split = tl.cdiv(n_tiles, SPLITS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    NW: tl.constexpr = D // PACK_G
    NG: tl.constexpr = D // V_GROUP
    nib_shifts = tl.arange(0, PACK_G)[None, None, :] * 4

    q = tl.load(q_ptr + row * Q_STRIDE + q_head * D + offs_d).to(tl.float32)
    head_off = kv_head * NW
    v_head_off = kv_head * NG

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for b in range(split * per_split, tl.minimum((split + 1) * per_split, n_tiles)):
        logical = b * BLOCK_N + offs_n
        raw = tl.load(bt_ptr + seq * MAX_BLOCKS + logical // PAGE)
        # A verify row may run past its sequence's reserved blocks (a capped
        # draft window); the padded table reads -1 there. Clamp for address
        # safety and mask the scores instead — the row's output is discarded
        # by the acceptance loop, but it must not read wild memory.
        blk = tl.maximum(raw, 0)
        rows = blk * PAGE + logical % PAGE

        kq = tl.load(kq_ptr + rows[:, None] * N_KV_HEADS * NW + head_off
                     + tl.arange(0, NW)[None, :])          # [BN, NW] uint32
        k_nib = ((kq[:, :, None] >> nib_shifts) & 0xF).to(tl.int32) - 8
        k_deq = tl.reshape(k_nib.to(tl.float32), (BLOCK_N, D))
        if STATIC_K:
            ks = tl.load(ks_ptr + kv_head * D + offs_d)    # [D]
            k_deq = k_deq * ks[None, :]
        else:
            ks = tl.load(ks_ptr + rows * N_KV_HEADS + kv_head).to(tl.float32)
            k_deq = k_deq * ks[:, None]

        s = tl.sum(q[None, :] * k_deq, axis=1) * SCALE
        s = tl.where((logical < context_len) & (raw >= 0), s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        safe_max = tl.where(m_new == -float('inf'), 0., m_new)
        p = tl.exp(s - safe_max)
        alpha = tl.exp(m_i - safe_max)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        vq = tl.load(vq_ptr + rows[:, None] * N_KV_HEADS * NW + head_off
                     + tl.arange(0, NW)[None, :])
        v_nib = ((vq[:, :, None] >> nib_shifts) & 0xF).to(tl.int32) - 8
        v_deq = tl.reshape(v_nib.to(tl.float32), (BLOCK_N, NG, V_GROUP))
        # one scale per (token, group); it covers all V_GROUP channels
        vs = tl.load(vs_ptr + rows[:, None] * N_KV_HEADS * NG + v_head_off
                     + tl.arange(0, NG)[None, :]).to(tl.float32)      # [BN, NG]
        v_deq = tl.reshape(v_deq * vs[:, :, None], (BLOCK_N, D))

        acc = acc * alpha + tl.sum(p[:, None] * v_deq, axis=0)
        m_i = m_new

    if SPLITS == 1:
        acc = acc / tl.maximum(l_i, 1e-20)
        tl.store(out_ptr + row * N_Q_HEADS * D + q_head * D + offs_d, acc.to(tl.float16))
    else:
        base = ((row * N_Q_HEADS + q_head) * SPLITS + split) * (D + 2)
        tl.store(out_ptr + base + offs_d, acc)
        tl.store(out_ptr + base + D, m_i)
        tl.store(out_ptr + base + D + 1, l_i)


@triton.jit
def merge_kv4_splits(parts, out, D: tl.constexpr, SPLITS: tl.constexpr, BLOCK_S: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    base = (row * SPLITS + s) * (D + 2)
    maxima = tl.load(parts + base + D, s < SPLITS, other=-float('inf'))
    sums = tl.load(parts + base + D + 1, s < SPLITS, other=0.)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float('inf'), 0., maximum)
    weights = tl.exp(maxima - maximum)
    accum = tl.load(parts + base[:, None] + d[None, :], (s < SPLITS)[:, None], other=0.)
    value = tl.sum(accum * weights[:, None], 0) / tl.maximum(tl.sum(sums * weights, 0), 1e-20)
    tl.store(out + row * D + d, value)


def paged_attention_decode(q: torch.Tensor, k_cache, v_cache,
                           block_tables: torch.Tensor, context_lens: torch.Tensor,
                           block_n: int = 128, num_kv_heads: int | None = None,
                           k_scale=None, v_group: int = 64,
                           verify_m: int = 1) -> torch.Tensor:
    """q [rows, N_Q_HEADS, D] fp16 -> out [rows, N_Q_HEADS, D] fp16."""
    rows, N_Q, D = q.shape
    n_kv = num_kv_heads if num_kv_heads is not None else N_Q
    out = torch.empty_like(q)
    kq, ks = k_cache
    vq, vs = v_cache
    static_k = ks.dim() == 2
    # Small tiles bound register pressure independently of the physical page size.
    tile = min(64, block_n)
    capacity = block_tables.shape[1] * block_n
    grouped = verify_m > 1
    if grouped:
        query_blocks = triton.cdiv(verify_m * (N_Q // n_kv), 16)
        programs = (rows // verify_m) * n_kv * query_blocks
        splits = min(8, triton.cdiv(capacity, tile), max(triton.cdiv(128, programs), triton.cdiv(capacity, 1024)))
    else:
        splits = min(8, triton.cdiv(capacity, max(256, capacity // 4))) if rows * N_Q < 128 else min(8, triton.cdiv(capacity, 1024))
    parts = torch.empty((rows, N_Q, splits, D + 2), device=q.device, dtype=torch.float32) if splits > 1 else out
    if grouped:
        kv4_grouped_verify_kernel[(n_kv, rows // verify_m, query_blocks * splits)](
            q, kq, ks, vq, vs, block_tables, context_lens, parts,
            HQ=N_Q, HK=n_kv, D=D, M=verify_m, PAGE=block_n, PAGES=block_tables.shape[1],
            GROUP=v_group, STATIC_K=static_k, SPLITS=splits, Q_STRIDE=q.stride(0), BM=16, BN=tile,
            num_stages=1)
    else:
        kv4_paged_decode_kernel[(N_Q, rows, splits)](
            q, kq, ks, vq, vs, block_tables, parts,
            context_lens.to(torch.int32),
            N_Q_HEADS=N_Q, N_KV_HEADS=n_kv, D=D, BLOCK_N=tile, PAGE=block_n,
            SPLITS=splits, Q_STRIDE=q.stride(0),
            MAX_BLOCKS=block_tables.shape[1], SCALE=1.0 / (D ** 0.5),
            STATIC_K=1 if static_k else 0, M=verify_m, V_GROUP=v_group,
        )
    if splits > 1:
        merge_kv4_splits[(rows * N_Q,)](parts, out, D, splits, triton.next_power_of_2(splits))
    return out
