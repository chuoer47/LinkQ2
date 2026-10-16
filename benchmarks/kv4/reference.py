"""Benchmark-only FP16 paged cache using the official FlashAttention API."""
from flash_attn import flash_attn_with_kvcache

from qslab.runtime.model.attention import PagedAttention


class FP16PagedAttention(PagedAttention):
    def __init__(self, original, keys, values, block_table, cache_start):
        super().__init__(original.num_heads, original.head_dim, original.scale, original.num_kv_heads)
        self.keys, self.values = keys, values
        self.block_table, self.cache_start = block_table, cache_start

    def forward(self, q, k, v):
        if self.lam is not None:
            q = q * self.lam_q.to(q.dtype)
            k = k / self.lam.to(k.dtype)
        return flash_attn_with_kvcache(
            q[None], self.keys, self.values, k=k[None], v=v[None],
            cache_seqlens=self.cache_start, block_table=self.block_table,
            softmax_scale=self.scale, causal=True)[0]
