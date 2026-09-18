"""M6: speculative mode comparison — chained / lookahead / dynamic.

Target: 8B W4-autodetect. Texts: repetitive (lookahead-friendly) + natural.
Validates losslessness per mode too (vs target-only greedy).
"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.models.w4linear import swap_w4_linears
from qslab.engine.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

N_IN, N_DEC = 512, 128
tok = QwenTokenizerAdapter("models/Qwen3-8B")

REPETITIVE = ("The quick brown fox jumps over the lazy dog. "
              "The quick brown fox jumps over the lazy dog. ") * 60
NATURAL = (" large language model inference requires careful memory management "
           "and quantization reduces the weight footprint significantly while ")


def make_ids(text, n):
    ids = tok.encode(text)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def bench(fn, rounds=3):
    rates = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        rates.append(N_DEC / (time.perf_counter() - t0))
    return statistics.median(rates)


target_cfg = EngineConfig(model_path="models/Qwen3-8B", device="cuda:0",
                          max_new_tokens=N_DEC)
draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0",
                         max_new_tokens=N_DEC)

results = {}
for text_name, text in [("repetitive", REPETITIVE), ("natural", NATURAL)]:
    ids = make_ids(text, N_IN)
    # baseline (W4 target-only)
    tgt = QslabEngine(target_cfg)
    swap_w4_linears(tgt.model, "models/Qwen3-8B-qslab-w4-awq")
    tgt.generate(ids, max_new_tokens=16)
    ref_tokens = tgt.generate(ids, max_new_tokens=N_DEC)
    base = bench(lambda: tgt.generate(ids, max_new_tokens=N_DEC))
    results[text_name] = {"baseline": round(base, 2)}
    print(f"[{text_name}] baseline: {base:.2f} tok/s", flush=True)

    for mode, gamma in [("chained", 4), ("lookahead", 4), ("dynamic", 4)]:
        spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=gamma,
                                 target_engine=tgt, mode=mode)
        out = spec.generate(ids, max_new_tokens=N_DEC)
        lossless = out == ref_tokens
        spec.generate(ids, max_new_tokens=16)
        tok_s = bench(lambda: spec.generate(ids, max_new_tokens=N_DEC))
        ar = spec.last_stats["ar"]
        results[text_name][mode] = {"tok_s": round(tok_s, 2), "ar": round(ar, 2),
                                    "speedup": round(tok_s / base, 3),
                                    "lossless": bool(lossless)}
        print(f"[{text_name}] {mode:9s}: {tok_s:6.2f} tok/s | AR {ar:4.2f} | "
              f"{tok_s/base:.2f}x | lossless={lossless}", flush=True)
        spec.draft = spec.draft if spec.draft is not None else None
        del spec
        torch.cuda.empty_cache()
    del tgt
    torch.cuda.empty_cache()

results["timestamp"] = datetime.now().isoformat()
Path("results/m6_spec_modes.json").write_text(json.dumps(results, indent=2))
print("saved results/m6_spec_modes.json")
