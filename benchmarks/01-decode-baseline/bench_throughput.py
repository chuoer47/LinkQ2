"""Throughput benchmark: batch=1 decode tokens/s (docs/03 §3.1 protocol).

- input 512 tokens (padded prompt from calibration-like text), decode 128
- report mean of the LAST 100 decode steps (skip warmup)
- 5 rounds -> median + std, plus GPU snapshot json in results/
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter


def gpu_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        gpus = [dict(zip(["index", "mem_used", "mem_total", "util"], line.split(", ")))
                for line in out.strip().splitlines()]
        return {"nvidia_smi": gpus}
    except Exception as e:
        return {"error": str(e)}


def make_input_ids(tok, n_tokens: int) -> list[int]:
    """Deterministic pseudo-text prompt of ~n_tokens."""
    text = (" large language model inference requires careful memory management "
            "and quantization reduces the weight footprint significantly while ")
    ids = tok.encode(text)
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


def bench_round(eng: QslabEngine, input_ids: list[int], n_decode: int) -> float:
    """One timed decode; returns tokens/s over the last 100 steps."""
    eng.reset_cache()
    logits = eng.prefill(input_ids)
    start_pos = len(input_ids)
    next_tok = int(logits.argmax(dim=-1))

    times: list[float] = []
    for i in range(n_decode - 1):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        logits = eng.decode_step(next_tok, start_pos=start_pos + i)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1) / 1000.0)  # ms -> s
        next_tok = int(logits.argmax(dim=-1))
    tail = times[-100:]
    return 1.0 / statistics.median(tail), sum(tail) / len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-1.7B")
    ap.add_argument("--n-input", type=int, default=512)
    ap.add_argument("--n-decode", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = QwenTokenizerAdapter(args.model)
    input_ids = make_input_ids(tok, args.n_input)
    cfg = EngineConfig(model_path=args.model, device="cuda:0")
    eng = QslabEngine(cfg)

    # warmup round (not recorded)
    eng.generate(input_ids, max_new_tokens=16)

    results = []
    for r in range(args.rounds):
        tok_per_s, mean_lat = bench_round(eng, input_ids, args.n_decode)
        results.append(tok_per_s)
        print(f"round {r}: {tok_per_s:.2f} tok/s (mean step {mean_lat*1000:.2f} ms)")

    med = statistics.median(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    kv_gb = eng.kv_memory_bytes() / 1e9

    report = {
        "model": args.model,
        "n_input": args.n_input,
        "n_decode": args.n_decode,
        "rounds": args.rounds,
        "tokens_per_s_median": round(med, 3),
        "tokens_per_s_all": [round(x, 3) for x in results],
        "tokens_per_s_std": round(std, 3),
        "kv_cache_valid_gb_after": round(kv_gb, 4),
        "gpu": gpu_snapshot(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    out = Path(args.out) if args.out else Path(
        f"results/bench_throughput_{Path(args.model).name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"median: {med:.2f} tok/s +- {std:.2f} | saved {out}")


if __name__ == "__main__":
    main()
