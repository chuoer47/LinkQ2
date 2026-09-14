"""Model loading: safetensors weights -> torch modules.

M0 path: build the reference transformers Qwen3 model, then keep its weights
as the oracle. The loader also exposes a plain state_dict reader that later
milestones (W4 packing / own format) will build upon.

NOTE: transformers import is tolerated here during M0 because the model
*definition* itself comes from transformers (decision 01 §1.1). The weights
reader below is the part that stays with qslab long-term.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from qslab.config import ModelConfig


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
    """Load the HF Qwen3 model in fp16 as reference/oracle."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    return model.to(device).eval()


def read_safetensors_state_dict(model_path: Path | str) -> dict[str, torch.Tensor]:
    """Plain safetensors reader (own-format loading will reuse this)."""
    from safetensors import safe_open

    model_path = Path(model_path)
    sd: dict[str, torch.Tensor] = {}
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {model_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    return sd
