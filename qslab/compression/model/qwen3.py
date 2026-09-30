"""Qwen3 checkpoint adapter used by the offline compressor."""
from __future__ import annotations

from qslab.reference.loader import load_model_config, load_reference_model


def load_qwen3(path: str, device: str):
    return load_reference_model(path, device=device), load_model_config(path)


def export_model_config(config) -> dict:
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
            "rms_norm_eps", "rope_theta", "max_position_embeddings",
            "tie_word_embeddings")
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}
