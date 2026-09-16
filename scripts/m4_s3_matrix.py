"""M4-S3/S6: 8B engine matrix — one mode per invocation (memory).

Usage: python scripts/m4_s3_matrix.py <fp16|w4|w4kv4|w4kv4spec> [--ppl]
Saves generation outputs + memory report per mode.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODE = sys.argv[1] if len(sys.argv) > 1 else "w4kv4"
DO_PPL = "--ppl" in sys.argv

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.model.w4linear import swap_w4_linears
from qslab.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

MODEL = "models/Qwen3-8B"
W4 = "models/Qwen3-8B-qslab-w4-awq"
PLAN = "results/kv4_plan_8b.json"
N_IN, N_DEC = 512, 128

tok = QwenTokenizerAdapter(MODEL)
text = (" large language model inference requires careful memory management "
        "and quantization reduces the weight footprint significantly while ")
ids = tok.encode(text)
while len(ids) < N_IN:
    ids = ids + ids
ids = ids[:N_IN]

cfg = EngineConfig(model_path=MODEL, device="cuda:0", max_new_tokens=N_DEC)

kv_mode = "fp16"
if "kv4" in MODE:
    kv_mode = "kv4"

if "spec" in MODE:
    # two-stage build: swap target to W4 BEFORE loading the draft, so the
    # fp16 8B never coexists with the packed copy on one GPU
    tgt = QslabEngine(cfg, kv_mode=kv_mode, kv_plan_path=PLAN)
    if "w4" in MODE:
        n = swap_w4_linears(tgt.model, W4)
        print(f"swapped {n} target linears (W4)", flush=True)
    eng = SpeculativeEngine(cfg, EngineConfig(model_path="models/Qwen3-0.6B",
                                              device="cuda:0", max_new_tokens=N_DEC),
                            gamma=4, target_engine=tgt)
    runner = lambda: eng.generate(ids, max_new_tokens=N_DEC)
else:
    eng = QslabEngine(cfg, kv_mode=kv_mode, kv_plan_path=PLAN)
    if "w4" in MODE:
        n = swap_w4_linears(eng.model, W4)
        print(f"swapped {n} linears", flush=True)
    runner = lambda: eng.generate(ids, max_new_tokens=N_DEC)

# warmup
eng.generate(ids, max_new_tokens=16)

# throughput: 3 rounds median
import statistics
rates = []
for _ in range(3):
    t0 = time.perf_counter()
    runner()
    rates.append(N_DEC / (time.perf_counter() - t0))
tok_s = statistics.median(rates)

tgt = eng.target if "spec" in MODE else eng
kv_mem = sum(c.memory_bytes_valid() for c in tgt.kv_caches) / 1e6
weight_mem = sum(p.nelement() * p.element_size() for p in tgt.model.parameters()) / 1e9

report = {"mode": MODE, "tok_s": round(tok_s, 2),
          "kv_mem_mb@512": round(kv_mem, 2), "weights_gb": round(weight_mem, 2)}
print("REPORT:", report, flush=True)

if DO_PPL:
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:16]
    from benchmarks.bench_ppl import ppl_of_model
    ppl = ppl_of_model(tgt.model, tok, texts, "cuda:0",
                       reset_fn=lambda: [c.reset() for c in tgt.kv_caches])
    report["ppl"] = round(ppl, 4)
    print("PPL:", round(ppl, 4), flush=True)

out_p = Path(f"results/m4_matrix_{MODE}.json")
import json
out_p.write_text(json.dumps(report, indent=2))
print(f"saved {out_p}")
