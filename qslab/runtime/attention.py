"""qslab runtime attention: flash-attn prefill + quantized paged decode.

Keeps nano-vllm's structure (global Context set per step by the model runner,
slot_mapping-driven KV writes) and replaces the decode read path with qslab's
int4 kernel. The PagedAttention layer owns per-layer views into the runtime's
quantized KV pool.

Ordering follows nano-vllm: store BEFORE attention. decode's context_lens
includes the token being processed, so its KV must already sit in its slot
when the kernel reads the cache.

SmoothAttention (optional, from a calibration file): q *= lambda and
k /= lambda, applied after RoPE. The attention logits are unchanged, but K's
per-channel outliers are flattened so a single static int4 scale per channel
suffices. lambda is per KV head; on the query side it is broadcast across each
head's GQA group.
"""
from __future__ import annotations

import torch
from torch import nn

from flash_attn import flash_attn_varlen_func

from qslab.runtime.context import get_context
from qslab.runtime.paged_decode import (store_kv_quant, paged_attention_decode,
                                        materialize_kv)


class PagedAttention(nn.Module):
    """One attention layer's interface into the quantized paged KV pool.

    k_cache/v_cache are (packed_u32, scales) tuples for this layer, attached
    by the model runner's allocate_kv_cache. When k_scale is the static table
    (shape [H_kv, D]) the K store/dedcode path uses a frozen per-channel
    scale; otherwise it falls back to a dynamic per-token scale.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float,
                 num_kv_heads: int, block_n: int = 128, v_group: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_n = block_n
        self.v_group = v_group
        self.k_cache = self.v_cache = None      # (packed, scales) tuples
        self.lam = None                         # [H_kv, D] SmoothAttention
        self.lam_q = None                       # [H_q, D] GQA-expanded

    def _materialize_prefix(self, ctx, k, v):
        """Dequantize the pool-resident prefix of each sequence and prepend
        it to this batch's freshly computed rows. k/v are [N, H_kv, D] where
        N covers only the scheduled (new) tokens, ordered by cu_seqlens_q;
        the returned tensors are ordered by cu_seqlens_k instead (prefix
        rows first, per sequence), which is what flash-attn consumes.

        The slot plan (prefix_slots + prefix_plan) is computed ONCE per step
        by prepare_prefill: 28 layers recomputing it with tolist() would pay
        a GPU sync each, which alone cost more than the saved prefill."""
        plan = ctx.prefix_plan
        kp, vp = materialize_kv(self.k_cache, self.v_cache, ctx.prefix_slots,
                                v_group=self.v_group)
        k_parts, v_parts = [], []
        for slot_off, cached, q_start, q_end in plan:
            if cached > 0:
                k_parts.append(kp[slot_off:slot_off + cached])
                v_parts.append(vp[slot_off:slot_off + cached])
            k_parts.append(k[q_start:q_end])
            v_parts.append(v[q_start:q_end])
        return torch.cat(k_parts), torch.cat(v_parts)

    def set_smooth_lambda(self, lam: torch.Tensor):
        """lam [H_kv, D] from the calibration file."""
        self.lam = lam
        n_rep = self.num_heads // self.num_kv_heads
        self.lam_q = lam.repeat_interleave(n_rep, dim=0).contiguous()

    # dynamo must not trace this: it reads the global Context (recreated
    # every step), which would churn guards; with it disabled, torch.compile
    # graph-breaks here and fuses everything in between — which is the point
    # of the compiled draft (notes/M9 §6: ~1500 unfused elementwise kernels)
    @torch._dynamo.disable
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        ctx = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if self.lam is not None:
            # (Q*L)(K/L)^T leaves the logits unchanged; only the quantization
            # difficulty of K moves. Done after RoPE, so the encoder sees the
            # smoothed key.
            q = (q * self.lam_q.to(q.dtype))
            k = (k / self.lam.to(k.dtype))

        # store FIRST (nano-vllm ordering): the decode kernel reads the
        # current token's own KV from the pool (context_lens includes it).
        if k_cache is not None and v_cache is not None:
            n_tokens = k.shape[0]
            if n_tokens > 0:
                # v arrives as a slice of the fused qkv projection and is NOT
                # contiguous (row stride = the full qkv width); the store
                # kernel indexes rows as idx*H*D, so it must be packed first.
                # flash-attn handles strides itself, which is why prefill was
                # unaffected while every stored V row was garbage.
                store_kv_quant(k.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous(),
                               v.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous(),
                               k_cache, v_cache,
                               ctx.slot_mapping.view(n_tokens),
                               v_group=self.v_group)

        if ctx.is_prefill:
            # prefill: attention over the fp16 q/k/v directly (varlen, causal);
            # the KV has been quantized into slots above for later decode reads.
            # A prefix-cache hit (or an earlier chunked-prefill block) means
            # cu_seqlens_k claims a length only the pool holds — flash-attn
            # would silently attend to misaligned rows (the M8 silent-wrong-
            # answer bug). Materialize the pool-resident prefix and prepend
            # it, so the declared lengths finally tell the truth.
            if ctx.block_tables is not None:
                k, v = self._materialize_prefix(ctx, k, v)
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
            # decode / verify: q is [rows, H, D] — bs rows for decode, bs*M
            # for a verify forward (M = ctx.verify_m). The cache is int4
            # packed and the kernel dequantizes tiles on the fly; row m of a
            # sequence reads keys [0, L+m), so drafts never leak into
            # earlier rows (their KV sits at higher slot indices).
            assert k_cache is not None and v_cache is not None
            o = paged_attention_decode(
                q, k_cache, v_cache, ctx.block_tables, ctx.context_lens,
                block_n=self.block_n, num_kv_heads=self.num_kv_heads,
                k_scale=self.k_cache[1] if k_cache[1].dim() == 2 else None,
                v_group=self.v_group, verify_m=ctx.verify_m)
        return o
