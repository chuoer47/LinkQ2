"""Torch layout operations for the packed int4 KV pool."""
from __future__ import annotations

import torch

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
