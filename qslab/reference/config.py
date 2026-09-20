"""Architecture params mirrored from a model's config.json (M0: Qwen3).

Runtime knobs (block size, W4 backend, speculation, memory budget) are NOT
here: they live in qslab/runtime/config.py. This mirror exists so the offline
reference path can talk about layer shapes without importing transformers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture params, loaded from a model's config.json (M0: Qwen3)."""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = False
