"""L0: slot-mapped int4 paged KV — store (quantize+write) and decode kernels."""
# Pool layout, one slot per token with all heads inside: k_q [slots, H, D/8] uint32 + k_s
#   [H, D] fp16 static; v_q [slots, H, D/8] uint32 + v_s [slots, H, D/G] fp16 dynamic.
# Encoding is offset-binary: store writes nibble = qi + 8 and every reader applies (nib - 8)
#   * scale — a two's-complement reader on offset-binary data biases each value by 8*scale.
# K is quantized with a static per-channel scale because calibration flattens the fixed
#   outlier channels; a dynamic per-channel scale would make the write address data-
#   dependent and break CUDA Graph capture. V's outliers are token-local, so its scale is
#   dynamic and grouped along D.
# Verify reuses the decode kernel as an M-query batch: rows = bs*M, row r is sequence r // M
#   at query r % M with key bound L + m, so causality needs no explicit mask.
from __future__ import annotations

import torch
import triton
import triton.language as tl

_MIN_SCALE = 1e-4


@triton.jit
def store_kv_quant_kernel(
    k_ptr, v_ptr,               # [N, H, D] fp16
    kq_ptr, ks_ptr,             # kq [slots,H,D/8] u32 ; ks [H,D] f16 (or [slots,H] f16)
    vq_ptr, vs_ptr,             # vq [slots,H,D/8] u32 ; vs [slots,H,D/G] f16
    slot_ptr,                   # [N] int32
    STATIC_K: tl.constexpr,     # 1 -> ks is the frozen [H, D] table
    H: tl.constexpr, D: tl.constexpr,
    PACK_G: tl.constexpr = 8, V_GROUP: tl.constexpr = 64,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_ptr + idx)
    if slot == -1:
        return
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    NW: tl.constexpr = D // PACK_G
    NG: tl.constexpr = D // V_GROUP
    n_in_w: tl.constexpr = V_GROUP // PACK_G          # words per V group
    pack_shifts = tl.arange(0, PACK_G)[None, None, :] * 4

    # ---------------- K ----------------
    k = tl.load(k_ptr + idx * H * D + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
    if STATIC_K:
        scale = tl.load(ks_ptr + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
    else:
        amax = tl.max(tl.abs(k), axis=1)
        scale = tl.maximum(amax / 7.0, 1e-4)[:, None]
    qi = tl.floor(tl.minimum(tl.maximum(k / scale, -8.0), 7.0) + 0.5).to(tl.int32)
    nib = (qi + 8) & 0xF
    packed = tl.sum(tl.reshape(nib, (H, NW, PACK_G)) << pack_shifts, axis=2)
    tl.store(kq_ptr + slot * H * NW + offs_h[:, None] * NW + tl.arange(0, NW)[None, :],
             packed)
    if not STATIC_K:
        tl.store(ks_ptr + slot * H + offs_h, scale[:, 0].to(tl.float16))

    # ---------------- V: per-token, groups of V_GROUP along D ----------------
    v = tl.load(v_ptr + idx * H * D + offs_h[:, None] * D + offs_d[None, :]).to(tl.float32)
    v_g = tl.reshape(v, (H, NG, V_GROUP))                       # [H, NG, V_GROUP]
    v_s = tl.maximum(tl.max(tl.abs(v_g), axis=2) / 7.0, 1e-4)   # [H, NG]
    vq = tl.floor(tl.minimum(tl.maximum(v_g / v_s[:, :, None], -8.0), 7.0) + 0.5)
    v_nib = (vq.to(tl.int32) + 8) & 0xF                         # [H, NG, V_GROUP]
    # pack V_GROUP nibbles into V_GROUP/PACK_G words per group
    v_words = tl.sum(tl.reshape(v_nib, (H, NG * n_in_w, PACK_G)) << pack_shifts, axis=2)
    tl.store(vq_ptr + slot * H * NW + offs_h[:, None] * NW
             + tl.arange(0, NG * n_in_w)[None, :], v_words)
    tl.store(vs_ptr + slot * H * NG + offs_h[:, None] * NG + tl.arange(0, NG)[None, :], v_s.to(tl.float16))


@triton.jit
def kv4_paged_decode_kernel(
    q_ptr, kq_ptr, ks_ptr, vq_ptr, vs_ptr, bt_ptr, out_ptr, context_lens_ptr,
    N_Q_HEADS: tl.constexpr, N_KV_HEADS: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, MAX_BLOCKS: tl.constexpr, SCALE: tl.constexpr,
    STATIC_K: tl.constexpr, M: tl.constexpr = 1,
    PACK_G: tl.constexpr = 8, V_GROUP: tl.constexpr = 64,
):
    q_head = tl.program_id(0)
    row = tl.program_id(1)               # row = seq * M + m (M=1: ordinary decode)
    seq = row // M
    m = row - seq * M
    kv_head = q_head // (N_Q_HEADS // N_KV_HEADS)
    context_len = tl.load(context_lens_ptr + seq) + m   # reads keys [0, L+m)
    n_blocks = tl.minimum(tl.cdiv(context_len, BLOCK_N), MAX_BLOCKS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    NW: tl.constexpr = D // PACK_G
    NG: tl.constexpr = D // V_GROUP
    n_in_w: tl.constexpr = V_GROUP // PACK_G
    nib_shifts = tl.arange(0, PACK_G)[None, None, :] * 4

    q = tl.load(q_ptr + row * N_Q_HEADS * D + q_head * D + offs_d).to(tl.float32)
    head_off = kv_head * NW
    v_head_off = kv_head * NG

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for b in range(n_blocks):
        raw = tl.load(bt_ptr + seq * MAX_BLOCKS + b)
        # A verify row may run past its sequence's reserved blocks (a capped
        # draft window); the padded table reads -1 there. Clamp for address
        # safety and mask the scores instead — the row's output is discarded
        # by the acceptance loop, but it must not read wild memory.
        blk = tl.maximum(raw, 0)
        rows = blk * BLOCK_N + offs_n                     # token slots in block

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
        s = tl.where((b * BLOCK_N + offs_n < context_len) & (raw >= 0), s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
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

    acc = acc / l_i
    tl.store(out_ptr + row * N_Q_HEADS * D + q_head * D + offs_d, acc.to(tl.float16))


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
    kv4_paged_decode_kernel[(N_Q, rows)](
        q, kq, ks, vq, vs, block_tables, out,
        context_lens.to(torch.int32),
        N_Q_HEADS=N_Q, N_KV_HEADS=n_kv, D=D, BLOCK_N=block_n,
        MAX_BLOCKS=block_tables.shape[1], SCALE=1.0 / (D ** 0.5),
        STATIC_K=1 if static_k else 0, M=verify_m, V_GROUP=v_group,
    )
    return out


def store_kv_quant(k: torch.Tensor, v: torch.Tensor,
                   k_cache, v_cache, slot_mapping: torch.Tensor,
                   v_group: int = 64):
    """k, v: [N, H, D] fp16. slot_mapping: [N] int32, one slot per token, -1 skips."""
    N, H, D = k.shape
    kq, ks = k_cache
    vq, vs = v_cache
    store_kv_quant_kernel[(N,)](k, v, kq, ks, vq, vs,
                                slot_mapping.to(torch.int32),
                                STATIC_K=1 if ks.dim() == 2 else 0,
                                H=H, D=D, V_GROUP=v_group)


def materialize_kv(k_cache, v_cache, slots: torch.Tensor,
                   v_group: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize pool slots back to fp16 K/V: [T, H, D] each."""
    # The inverse of store_kv_quant, used by prefill when a prefix-cache hit hands flash-
    #   attn a length only the pool holds. K comes back post-lambda, i.e. exactly what a
    #   fresh prefill would compute up to the int4 error.
    kq, ks = k_cache
    vq, vs = v_cache
    slots = torch.as_tensor(slots, device=kq.device)
    H, NW = kq.shape[1], kq.shape[2]
    T = slots.numel()                     # NOT kq.shape[0] (the whole pool)
    D = NW * 8
    NG = vs.shape[-1]
    shifts = torch.arange(8, device=kq.device, dtype=torch.int32) * 4

    # uint32 has no tensor-indexing kernel: reinterpret as int32 first
    # (values are 4-bit payloads, sign is irrelevant)
    kq = kq.view(torch.int32)[slots]                           # [T, H, NW]
    vq = vq.view(torch.int32)[slots]

    def unpack(w: torch.Tensor) -> torch.Tensor:
        # [T, H, NW] i32 -> [T, H, D] signed nibble values
        return (((w.unsqueeze(-1) >> shifts) & 0xF) - 8).flatten(-2)

    k = unpack(kq).float()                                      # [T, H, D]
    if ks.shape[-1] == D:                                       # static [H, D]
        k = k * ks.float()[None]
    else:                                                       # dynamic [slots, H]
        k = k * ks[slots].float()[..., None]
    v = (unpack(vq).float().reshape(T, H, NG, v_group)
         * vs[slots].float()[:, :, :, None])
    return k.half(), v.reshape(T, H, D).half()
