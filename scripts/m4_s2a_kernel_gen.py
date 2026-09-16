"""M4-S2 (split A): kernel-path engine generation on 8B, save tokens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.models.w4linear import swap_w4_linears
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-8B")
prompts = [" The capital of France is", " large language model quantization is"]

cfg = EngineConfig(model_path="models/Qwen3-8B", device="cuda:0", max_new_tokens=32)
eng = QslabEngine(cfg)
n = swap_w4_linears(eng.model, "models/Qwen3-8B-qslab-w4-awq")
print(f"swapped {n} linears", flush=True)
out = {}
for p in prompts:
    ids = tok.encode(p)
    o = eng.generate(ids, max_new_tokens=32)
    out[p] = o
    print(f"gen ok: {tok.decode(o)[:50]!r}", flush=True)
torch.save(out, "results/m4_s2_kernel_tokens.pt")
print("saved results/m4_s2_kernel_tokens.pt")
