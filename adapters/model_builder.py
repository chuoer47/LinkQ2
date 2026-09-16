"""HF model construction adapter — the ONLY place qslab builds a transformers model.

L2 (qslab.models) must not import transformers; it receives a built module from
here. This keeps the dependency rule: L2 ← L3 ← L4 ← adapters (HF side).
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def build_hf_causal_lm(model_path: Path | str, device: str = "cuda:0",
                       dtype: torch.dtype = torch.float16,
                       attn_impl: str = "eager") -> torch.nn.Module:
    """Build the HF Qwen3 model (fp16, eval) on `device`."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, attn_implementation=attn_impl)
    return model.to(device).eval()


def hf_config_dict(model_path: Path | str) -> dict:
    """Raw HF config as a dict (for ModelConfig construction in L2)."""
    cfg = AutoConfig.from_pretrained(str(model_path))
    return cfg.to_dict()
