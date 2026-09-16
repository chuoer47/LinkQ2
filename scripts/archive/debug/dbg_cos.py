"""Debug: capture per-layer cos/sin and query rope positions in path A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")
ext = [12095, 13]
cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0")

eng = QslabEngine(cfg)
eng.reset_cache()
eng.prefill(ids)

# call attention layer0 directly with the SAME hidden states that path B
# would produce for tokens [12095, 13] at positions 5,6
from qslab.models.patched import apply_rope

attn0 = eng.model.model.layers[0].self_attn

# manually compute hidden_states for ext tokens through the backbone? too deep.
# Instead: capture what cos/sin the backbone passes during the append forward.
cos_sin = {}
orig_forward = attn0.forward.__wrapped__ if hasattr(attn0.forward, "__wrapped__") else None

import types
captured = {}
def spy_forward(hidden_states, position_embeddings, attention_mask=None, **kw):
    captured["cos_shape"] = position_embeddings[0].shape
    captured["cos_first"] = position_embeddings[0][0, :, 0, :3].tolist()
    captured["hs_shape"] = hidden_states.shape
    return orig_forward(hidden_states, position_embeddings, attention_mask, **kw) if orig_forward else None

attn0.forward = spy_forward
try:
    eng.model(input_ids=torch.tensor([ext], device="cuda:0"),
              cache_position=torch.arange(5, 7, device="cuda:0"),
              use_cache=False)
except Exception as e:
    print("spy forward err:", e)
print("captured:", captured)
