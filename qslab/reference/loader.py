"""HF reference loading: config mirror + the fp16 reference model."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from qslab.reference.config import ModelConfig


def load_model_config(model_path: Path | str) -> ModelConfig:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return ModelConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )


def load_reference_model(model_path: Path | str, device: str = "cuda:0") -> torch.nn.Module:
    """Load the HF Qwen3 model in fp16 (delegates to the adapters layer)."""
    # L2 must not import transformers; construction lives in adapters.model_builder.
    from adapters.model_builder import build_hf_causal_lm
    return build_hf_causal_lm(model_path, device=device)
