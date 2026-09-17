"""qslab runtime Qwen3 — vendored from nano-vllm (MIT), adapted:

- TP parallelism removed (single GPU).
- Linear layers are qslab runtime primitives with the weight_loader protocol
  (so nano-vllm's loader fills them), and are W4-swappable after load.
- Attention is qslab's quantized paged attention: per-token int4 KV storage
  (slot-mapped), flash-attn varlen prefill, Triton decode kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3Config

from qslab.runtime.primitives import (
    Linear, MergedLinear, QKVLinear, RMSNorm, SiluAndMul, VocabEmbedding)
from qslab.runtime.attention import PagedAttention
from qslab.runtime.rotary import get_rope


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_position: int = 4096 * 32, head_dim: int | None = None,
                 rms_norm_eps: float = 1e-06, qkv_bias: bool = False,
                 rope_theta: float = 10000, rope_scaling: dict | None = None):
        super().__init__()
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVLinear(hidden_size, self.total_num_heads,
                                  self.total_num_kv_heads, self.head_dim,
                                  bias=qkv_bias)
        self.o_proj = Linear(self.total_num_heads * self.head_dim, hidden_size)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, max_position,
                                   rope_theta)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling,
                                   self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedLinear(hidden_size,
                                         [intermediate_size, intermediate_size])
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = Linear(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)
