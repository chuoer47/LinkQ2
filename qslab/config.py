"""Model architecture config, mirrored from a model's config.json.

The engine/quantization knobs that used to sit next to ModelConfig
(W4QuantConfig / KVQuantConfig / SpecConfig / EngineConfig) belonged to the
M0-M7 engine, now on branch archive/legacy-engine. Runtime knobs live in
qslab/runtime/config.py.
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

