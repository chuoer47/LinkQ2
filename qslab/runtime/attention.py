"""qslab runtime attention: flash-attn prefill + quantized paged decode.

Keeps nano-vllm's structure (global Context set per step by the model runner,
slot_mapping-driven KV writes) and replaces the decode read path with qslab's
int4 kernel. The PagedAttention layer owns per-layer views into the runtime's
quantized KV pool.

Ordering follows nano-vllm: store BEFORE attention. decode's context_lens
includes the token being processed, so its KV must already sit in its slot
when the kernel reads the cache.
"""
from __future__ import annotations

import torch
from torch import nn

from flash_attn import flash_attn_varlen_func

from qslab.runtime.context import get_context
from qslab.runtime.paged_decode import store_kv_quant, paged_attention_decode


class PagedAttention(nn.Module):
    """One attention layer's interface into the quantized paged KV pool.

    k_cache/v_cache are (packed_u32, scales_fp16) tuples for this layer,
    attached by the model runner's allocate_kv_cache.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float,
                 num_kv_heads: int, block_n: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_n = block_n
        self.k_cache = self.v_cache = None      # (packed, scales) tuples

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # store FIRST (nano-vllm ordering): the decode kernel reads the
        # current token's own KV from the pool (context_lens includes it).
        if k_cache is not None and v_cache is not None:
            n_tokens = k.shape[0]
            if n_tokens > 0:
                store_kv_quant(k.view(n_tokens, self.num_kv_heads, self.head_dim),
                               v.view(n_tokens, self.num_kv_heads, self.head_dim),
                               k_cache, v_cache,
                               ctx.slot_mapping.view(n_tokens))

        if ctx.is_prefill:
            # prefill: attention over the fp16 q/k/v directly (varlen, causal);
            # the KV has been quantized into slots above for later decode reads
            o = flash_attn_varlen_func(
                q.view(-1, self.num_heads, self.head_dim),
                k.view(-1, self.num_kv_heads, self.head_dim),
                v.view(-1, self.num_kv_heads, self.head_dim),
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            # decode: q is [bs, H, D]; the cache is int4 packed and the kernel
            # dequantizes tiles on the fly
            assert k_cache is not None and v_cache is not None
            bs = q.shape[0]
            if bs == 1:
                o = paged_attention_decode(
                    q[0], k_cache, v_cache, ctx.block_tables[0],
                    int(ctx.context_lens[0]), block_n=self.block_n,
                    num_kv_heads=self.num_kv_heads)
                o = o.unsqueeze(0)                # [1, H, D]
            else:
                o = paged_attention_decode_batched(
                    q, k_cache, v_cache, ctx.block_tables,
                    ctx.context_lens, block_n=self.block_n,
                    num_kv_heads=self.num_kv_heads)
        return o
