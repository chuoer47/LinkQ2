"""M3-S5 v2: lossless rate over many tokens (acceptance-line wording:
>=99% token-level agreement vs target-only greedy; float nondeterminism ok).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

GAMMA = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_NEW = 64

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
prompts = [" The capital of France is", " large language model quantization is",
           " def fibonacci(n):"]

target_cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0",
                          max_new_tokens=N_NEW)
draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0",
                         max_new_tokens=N_NEW)

ref = QslabEngine(target_cfg)
spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=GAMMA)

total = 0
match = 0
for p in prompts:
    ids = tok.encode(p)
    out_ref = ref.generate(ids, max_new_tokens=N_NEW)
    out_spec = spec.generate(ids, max_new_tokens=N_NEW)
    n = min(len(out_ref), len(out_spec))
    m = sum(1 for a, b in zip(out_ref, out_spec) if a == b)
    total += n
    match += m
    print(f"prompt {p[:26]!r}: {m}/{n} tokens agree")
rate = match / total
print(f"TOKEN_AGREEMENT_RATE: {rate:.4f} (line: >=0.99)")
print("spec stats:", {k: round(v, 2) if isinstance(v, float) else v
                      for k, v in spec.last_stats.items()})
