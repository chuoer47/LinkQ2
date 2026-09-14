"""Qwen3Attention replacement: swap torch ops for qslab-managed paths.

Strategy (docs/01 §1.2): subclass transformers' attention module and replace
the cache interaction with our own KV cache implementation. The q/k/v/o
projections stay as the parent's Linear modules for M0 — M1 will redirect
those matmuls to qslab_kernels.w4a16_gemm.

We do NOT monkey-patch: the engine builds the reference model, then walks
`model.model.layers[i].self_attn` and replaces each instance with
PatchedQwen3Attention (copying over the projection weights).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from qslab.cache.kv_cache import FP16KVCache


class PatchedQwen3Attention(Qwen3Attention):
    """Qwen3 attention with qslab KV cache and explicit call sites.

    Differences from the parent class:
    - uses an externally-owned FP16KVCache (passed via `qslab_cache`) instead
      of transformers' DynamicCache
    - attention matmuls written out explicitly (easy to swap for kernels in M1)
    """

    def __init__(self, parent_attn: Qwen3Attention, layer_idx: int, engine_cache: FP16KVCache):
        # Bypass parent's __init__ (it would allocate new projections); we
        # instead build a bare nn.Module and steal the parent's submodules.
        # (4.57: head counts live on config, not the module — pull from config)
        torch.nn.Module.__init__(self)
        self.config = parent_attn.config
        self.layer_idx = layer_idx
        self.head_dim = parent_attn.head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = parent_attn.num_key_value_groups
        self.max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
        self.rope_theta = getattr(self.config, "rope_theta", None)
        self.is_causal = parent_attn.is_causal
        self.attention_dropout = parent_attn.attention_dropout
        self.scaling = parent_attn.scaling

        # steal projection modules from the parent (no weight copy needed)
        self.q_proj = parent_attn.q_proj
        self.k_proj = parent_attn.k_proj
        self.v_proj = parent_attn.v_proj
        self.o_proj = parent_attn.o_proj
        self.q_norm = parent_attn.q_norm
        self.k_norm = parent_attn.k_norm
        self.sliding_window = parent_attn.sliding_window

        # engine-managed cache (one per layer, owned by the engine loop)
        self.engine_cache = engine_cache

        # we are already a nn.Module (Qwen3Attention is one); mark init done
        self._qslab_initialized = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        """Simplified forward for decode-only: no cross-attention, no FA2."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # rope: same math as transformers (rotate_half), explicit here
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        # --- engine-owned cache update ---
        k_full, v_full = self.engine_cache.update(key_states, value_states)

        # repeat KV for GQA, then explicit attention matmuls (swap points in M1/M2)
        k_rep = repeat_kv(k_full, self.num_key_value_groups)
        v_rep = repeat_kv(v_full, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, k_rep.transpose(2, 3)) * self.scaling
        # causal mask: query at position t attends to [0..t]
        T_q = query_states.shape[2]
        T_k = k_full.shape[2]
        kv_start = T_k - T_q
        positions = torch.arange(kv_start, T_k, device=attn_weights.device)
        causal = (positions[None, :] <= positions[:, None] + kv_start)  # [T_q, T_k]
        attn_weights = attn_weights + torch.where(
            causal, 0.0, torch.finfo(attn_weights.dtype).min
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, v_rep)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(attn_output), None


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Standard rotate-half RoPE (matches transformers Qwen3)."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv*n_rep, T, D] (same as transformers)."""
    if n_rep == 1:
        return hidden
    B, H, T, D = hidden.shape
    return hidden[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


def patch_model(model: torch.nn.Module, num_layers: int, num_kv_heads: int,
                head_dim: int, max_len: int, device: str) -> list[FP16KVCache]:
    """Replace every layer's self_attn with PatchedQwen3Attention.

    Returns the list of engine caches (one per layer) so the engine loop
    owns the cache lifecycle.
    """
    from types import SimpleNamespace

    caches: list[FP16KVCache] = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        parent = layer.self_attn
        cache = FP16KVCache(batch=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
                            max_len=max_len, device=device)
        patched = PatchedQwen3Attention(parent, layer_idx=i, engine_cache=cache)
        layer.self_attn = patched
        caches.append(cache)
    return caches
