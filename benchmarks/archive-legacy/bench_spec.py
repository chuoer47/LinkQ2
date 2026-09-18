"""M3-S6: speculative decoding benchmark — gamma sweep, AR, e2e speedup.

Protocol (docs/03): batch=1 decode, 512 in / 128 dec, median of 5 rounds.
Compares target-only greedy vs speculative (gamma in 1..6).
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
from qslab.engine.spec import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

N_IN, N_DEC, ROUNDS = 512, 128, 5


def make_ids(tok, n):
    text = (" large language model inference requires careful memory management "
            "and quantization reduces the weight footprint significantly while ")
    ids = tok.encode(text)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def bench(fn, rounds=ROUNDS):
    rates = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        rates.append(N_DEC / (time.perf_counter() - t0))
    return statistics.median(rates), statistics.stdev(rates)


def main():
    tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
    ids = make_ids(tok, N_IN)

    target_cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0",
                              max_new_tokens=N_DEC)
    draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0",
                             max_new_tokens=N_DEC)

    ref = QslabEngine(target_cfg)
    tok_s_base, std_base = bench(lambda: ref.generate(ids, max_new_tokens=N_DEC))
    print(f"target-only : {tok_s_base:.2f} tok/s (std {std_base:.2f})")
    del ref
    torch.cuda.empty_cache()

    results = {"baseline_tok_s": round(tok_s_base, 2), "gamma": {}}
    for gamma in [1, 2, 3, 4, 5, 6]:
        spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=gamma)
        # warmup
        spec.generate(ids, max_new_tokens=16)
        tok_s, std = bench(lambda: spec.generate(ids, max_new_tokens=N_DEC))
        ar = spec.last_stats["ar"]
        results["gamma"][gamma] = {
            "tok_s": round(tok_s, 2), "std": round(std, 2), "ar": round(ar, 2),
            "speedup": round(tok_s / tok_s_base, 3),
        }
        print(f"gamma={gamma}: {tok_s:.2f} tok/s | AR {ar:.2f} | speedup {tok_s/tok_s_base:.2f}x")
        del spec
        torch.cuda.empty_cache()

    results["timestamp"] = datetime.now().isoformat()
    Path("results/bench_spec.json").write_text(json.dumps(results, indent=2))
    print("saved results/bench_spec.json")


if __name__ == "__main__":
    main()
