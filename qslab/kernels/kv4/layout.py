"""Fused materialization of selected packed KV slots for chunked prefill."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def materialize_kv_kernel(kq, ks, vq, vs, slots, ko, vo,
                          H: tl.constexpr, D: tl.constexpr, V_GROUP: tl.constexpr,
                          STATIC_K: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (H * D)
    head = offsets // D % H
    channel = offsets % D
    slot = tl.load(slots + token, offsets < COUNT, other=0)
    word = (slot * H + head) * (D // 8) + channel // 8
    shift = (channel % 8) * 4
    kw = tl.load(kq + word, offsets < COUNT, other=0)
    vw = tl.load(vq + word, offsets < COUNT, other=0)
    key = ((kw >> shift) & 15).to(tl.int32) - 8
    value = ((vw >> shift) & 15).to(tl.int32) - 8
    if STATIC_K:
        key_scale = tl.load(ks + head * D + channel, offsets < COUNT, other=0).to(tl.float32)
    else:
        key_scale = tl.load(ks + slot * H + head, offsets < COUNT, other=0).to(tl.float32)
    value_scale = tl.load(vs + (slot * H + head) * (D // V_GROUP) + channel // V_GROUP,
                          offsets < COUNT, other=0).to(tl.float32)
    tl.store(ko + offsets, key.to(tl.float32) * key_scale, offsets < COUNT)
    tl.store(vo + offsets, value.to(tl.float32) * value_scale, offsets < COUNT)

def materialize_kv(k_cache, v_cache, slots: torch.Tensor,
                   v_group: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize selected slots to FP16 [T,H,D], without intermediate gathers."""
    kq, ks = k_cache
    vq, vs = v_cache
    slots = torch.as_tensor(slots, device=kq.device)
    h, d = kq.shape[1], kq.shape[2] * 8
    shape = (slots.numel(), h, d)
    k = torch.empty(shape, device=kq.device, dtype=torch.float16)
    v = torch.empty_like(k)
    count = k.numel()
    if count:
        materialize_kv_kernel[(triton.cdiv(count, 512),)](kq, ks, vq, vs, slots, k, v,
            H=h, D=d, V_GROUP=v_group, STATIC_K=ks.shape[-1] == d, COUNT=count, BLOCK=512)
    return k, v
