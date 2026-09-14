"""Engine-level configuration.

All knobs that change engine/quantization behavior live here as frozen
dataclasses. No I/O, no torch import needed at module level for typing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass(frozen=True)
class W4QuantConfig:
    """Weight quantization settings (M1). format_version matches packfmt."""
    enabled: bool = False
    algo: str = "rtn"            # "rtn" | "awq"
    group_size: int = 128
    symmetric: bool = True       # v1: zero field present but all zeros
    format_version: int = 1


@dataclass(frozen=True)
class KVQuantConfig:
    """KV cache quantization settings (M2). Plan comes from quantizer."""
    enabled: bool = False
    kv_fp16_layers: tuple[int, ...] = ()   # layers kept at fp16
    k_quant: str = "per_channel"
    v_quant: str = "per_token"


@dataclass(frozen=True)
class SpecConfig:
    """Speculative decoding settings (M3)."""
    enabled: bool = False
    draft_model_path: str = ""
    gamma: int = 4               # draft tokens per verify round
    seed: int = 0


@dataclass(frozen=True)
class EngineConfig:
    model_path: Path
    device: str = "cuda:0"
    dtype: str = "float16"       # M0: fp16 only
    max_new_tokens: int = 128
    temperature: float = 0.0     # 0.0 = greedy (M0 verification default)
    top_k: int = 0
    top_p: float = 1.0
    w4: W4QuantConfig = field(default_factory=W4QuantConfig)
    kv4: KVQuantConfig = field(default_factory=KVQuantConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
