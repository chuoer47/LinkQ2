"""M5: speculative decoding retest on 8B with W4-autodetect target.

M3 found 0.74x on 1.7B (dispatch-bound); M4 saw 1.19x with fp16-8B target
+ kv4. Now: W4 target (39.8 tok/s solo) + 0.6B draft — does spec still add
on top of a slower target? Protocol: 512 in / 128 dec, gamma sweep.
"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.model.w4linear import swap_w4_linears
from qslab.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

N_IN, N_DEC = 512, 128
tok = QwenTokenizerAdapter("models/Qwen3-8B")
text = (" large language model inference requires careful memory management "
        "and quantization reduces the weight footprint significantly while ")
ids = tok.encode(text)
while len(ids) < N_IN:
    ids = ids + ids
ids = ids[:N_IN]

target_cfg = EngineConfig(model_path="models/Qwen3-8B", device="cuda:0",
                          max_new_tokens=N_DEC)
draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0",
                         max_new_tokens=N_DEC)

# baseline: W4-autodetect target-only (already measured 39.78, re-measure here)
tgt = QslabEngine(target_cfg)
swap_w4_linears(tgt.model, "models/Qwen3-8B-qslab-w4-awq")


def bench(fn, rounds=3):
    rates = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        rates.append(N_DEC / (time.perf_counter() - t0))
    return statistics.median(rates)


tgt.generate(ids, max_new_tokens=16)
tok_s_base = bench(lambda: tgt.generate(ids, max_new_tokens=N_DEC))
print(f"W4-autodetect target-only: {tok_s_base:.2f} tok/s", flush=True)

results = {"baseline_tok_s": round(tok_s_base, 2), "gamma": {}}
for gamma in [2, 4, 6]:
    # two-stage build: swap target first, then attach draft
    spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=gamma, target_engine=tgt)
    spec.generate(ids, max_new_tokens=16)
    tok_s = bench(lambda: spec.generate(ids, max_new_tokens=N_DEC))
    ar = spec.last_stats["ar"]
    results["gamma"][gamma] = {"tok_s": round(tok_s, 2), "ar": round(ar, 2),
                               "speedup": round(tok_s / tok_s_base, 3)}
    print(f"gamma={gamma}: {tok_s:.2f} tok/s | AR {ar:.2f} | speedup {tok_s/tok_s_base:.2f}x",
          flush=True)
    # NOTE: draft stays attached to tgt; rebuild spec each gamma reuses tgt
    del spec.draft
    torch.cuda.empty_cache()

results["timestamp"] = datetime.now().isoformat()
Path("results/m5_spec_retest.json").write_text(json.dumps(results, indent=2))
print("saved results/m5_spec_retest.json")
