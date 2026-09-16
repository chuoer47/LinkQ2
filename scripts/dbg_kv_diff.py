"""Debug: compare K/V cache content: append path vs full-forward path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.model.patched import PatchedQwen3Attention
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")
ext = [12095, 13]

cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0")

# capture K/V at layer 0 in both paths
caps = {}
def cap_hook(name):
    def hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        caps.setdefault(name, []).append(hs.detach().clone())
    return hook

# --- path A: append (prefill 5, then forward 2) ---
engA = QslabEngine(cfg)
attn0 = engA.model.model.layers[0].self_attn
h = attn0.register_forward_pre_hook(cap_hook("A"), with_kwargs=True)
engA.reset_cache()
engA.prefill(ids)
outA = engA.model(input_ids=torch.tensor([ext], device="cuda:0"),
                  cache_position=torch.arange(len(ids), len(ids)+2, device="cuda:0"),
                  use_cache=False)
h.remove()
kA = engA.kv_caches[0].k[:, :, :7].clone()
print("A logits argmax:", outA.logits[0].argmax(-1).tolist())

# --- path B: one full forward of 7 tokens ---
engB = QslabEngine(cfg)
attn0B = engB.model.model.layers[0].self_attn
h = attn0B.register_forward_pre_hook(cap_hook("B"), with_kwargs=True)
engB.reset_cache()
outB = engB.model(input_ids=torch.tensor([ids + ext], device="cuda:0"), use_cache=False)
h.remove()
kB = engB.kv_caches[0].k[:, :, :7].clone()
print("B logits argmax:", outB.logits[0].argmax(-1).tolist())

print("K diff (layer0, first 7 pos):", (kA - kB).abs().max().item())
