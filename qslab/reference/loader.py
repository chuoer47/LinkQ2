"""HF FP16 reference loading."""
from pathlib import Path
import torch


def load_reference_model(model_path: Path | str, device: str = "cuda:0") -> torch.nn.Module:
    """Load the HF Qwen3 model in fp16 (delegates to the adapters layer)."""
    # L2 must not import transformers; construction lives in adapters.model_builder.
    from adapters.model_builder import build_hf_causal_lm
    return build_hf_causal_lm(model_path, device=device)
