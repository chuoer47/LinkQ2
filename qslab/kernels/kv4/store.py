"""Int4 paged KV Triton operators."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

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
