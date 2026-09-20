"""qslab runtime Qwen3 — vendored from nano-vllm (MIT), adapted:

- TP parallelism removed (single GPU).
- Linear layers are qslab runtime primitives with the weight_loader protocol
  (so nano-vllm's loader fills them), and are W4-swappable after load.
- Attention is qslab's quantized paged attention: per-token int4 KV storage
  (slot-mapped), flash-attn varlen prefill, Triton decode kernel.

**Projections are deliberately NOT fused** (no qkv_proj / gate_up_proj):

  * the packed W4 checkpoints are keyed by HF module name (q_proj, k_proj,
    v_proj, gate_proj, up_proj) with the AWQ input scale stored per logical
    module — a fused qkv would discard them, or force per-shard scale factors
    to be stored on one tensor.
  * keeping them separate ends the "v is a strided view of a fused buffer"
    hazard that caused the M8 store bug.
  * the cost was measured at ~3% decode throughput (149.1 -> 144.8 tok/s),
    which does not justify the fused layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from qslab.runtime.model.primitives import Linear, LMHead, RMSNorm, VocabEmbedding
from qslab.runtime.model.attention import PagedAttention
from qslab.runtime.model.rotary import get_rope


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

        # separate projections: see the module docstring
        self.q_proj = Linear(hidden_size, self.q_size, bias=qkv_bias)
        self.k_proj = Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.v_proj = Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.o_proj = Linear(self.q_size, hidden_size)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, max_position,
                                   rope_theta, rope_scaling)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling,
                                   self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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
    # projections are stored separately, so no shard mapping is needed
    packed_modules_mapping: dict = {}

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)
