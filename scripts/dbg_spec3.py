"""Debug: per-round trace of speculative generation vs ref tokens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")
N = 64

target_cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=N)
draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0", max_new_tokens=N)

ref = QslabEngine(target_cfg)
out_ref = ref.generate(ids, max_new_tokens=N)
print("ref :", out_ref)

spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=4)

# instrument: wrap SpeculativeEngine.generate loop via manual replay
# simpler: patch to print inside — instead re-derive with small gamma and
# compare after each round using the stats + output diff
out_spec = spec.generate(ids, max_new_tokens=N)
print("spec:", out_spec)
# find first divergence
for i, (a, b) in enumerate(zip(out_ref, out_spec)):
    if a != b:
        print(f"first divergence at generated[{i}]: ref={a} spec={b}")
        print("ref context:", out_ref[max(0,i-3):i+3])
        print("spec context:", out_spec[max(0,i-3):i+3])
        break
