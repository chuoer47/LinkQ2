"""HF model construction adapter — the ONLY place qslab builds a transformers model."""
# L2 (qslab.models) must not import transformers; it receives a built module from here.
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig


def build_hf_causal_lm(model_path: Path | str, device: str = "cuda:0",
                       dtype: torch.dtype | str = torch.float16,
                       attn_impl: str = "eager") -> torch.nn.Module:
    """Build the HF Qwen3 model (fp16, eval) on `device`."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=dtype, attn_implementation=attn_impl)
    return model.to(device).eval()


def build_quantized_causal_lm(model_path, device='cuda:0', dtype=torch.float16,
                            backend='gemlite', fake=False):
    from accelerate import init_empty_weights
    from qslab.runtime.quantization import load_checkpoint_weights
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation='eager')
    load_checkpoint_weights(model, model_path, backend=backend, device=device, dtype=dtype, fake=fake)
    model.tie_weights()
    if (Path(model_path) / 'generation_config.json').exists():
        model.generation_config = GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
    return model.to(device).eval()
