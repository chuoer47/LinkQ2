"""M2-S5: NIAH (needle in a haystack) recall at 32K context.

Synthetic haystack of repeated filler text; one random needle
("One of the special magic numbers for {key} is: {value}.") buried at a
depth ratio; question asks the value. Greedy decode, exact-match recall.

Compares fp16 vs kv4 cache at 32K (Qwen3-1.7B native window).
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

CTX = 32768
DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]
N_TRIALS = 3


def make_haystack(tok, target_tokens: int, rng: random.Random) -> str:
    rng.seed(1234)
    filler = ("The sun rises over the quiet hills and the village begins another "
              "ordinary day. Farmers walk along the road, birds cross the sky, "
              "and life moves at its familiar gentle pace. ")
    words = []
    n = 0
    while n < target_tokens:
        words.append(filler)
        n = len(tok.encode(" ".join(words[-8:]))) if words else 0
        if len(words) % 64 == 0:
            n = len(tok.encode(" ".join(words)))
    return " ".join(words)


def run():
    tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
    rng = random.Random(2026)
    results = {}
    for mode in ["fp16", "kv4"]:
        cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0",
                           max_new_tokens=16)
        eng = QslabEngine(cfg, kv_mode=mode, kv_plan_path="results/kv4_plan.json")
        recalls = []
        for depth in DEPTHS:
            hit = 0
            for t in range(N_TRIALS):
                key = f"magic-{rng.randint(1000,9999)}"
                value = str(rng.randint(100000, 999999))
                needle = f"One of the special magic numbers for {key} is: {value}. "
                hay = make_haystack(tok, CTX - 200, random.Random(1234))
                pos = int(len(hay) * depth)
                text = hay[:pos] + needle + hay[pos:]
                q = f"\n\nQuestion: What is the special magic number for {key}? Answer:"
                ids = tok.encode(text + q)[:CTX]
                # trim from front if over
                out = eng.generate_chunked(ids, max_new_tokens=12)
                ans = tok.decode(out)
                hit += int(value in ans)
                print(f"  [{mode} depth={depth} t={t}] ans={ans.strip()[:24]!r} "
                      f"expect={value} hit={value in ans}")
            recalls.append({"depth": depth, "hits": hit, "trials": N_TRIALS})
        rec = sum(r["hits"] for r in recalls) / sum(r["trials"] for r in recalls)
        results[mode] = {"overall_recall": round(rec, 3), "by_depth": recalls}
        print(f"[{mode}] overall recall {rec:.2%}")
        del eng
        torch.cuda.empty_cache()

    out = {"ctx": CTX, "model": "Qwen3-1.7B", "results": results,
           "timestamp": datetime.now().isoformat()}
    Path("results/m2_niah.json").write_text(json.dumps(out, indent=2))
    print("saved results/m2_niah.json")


if __name__ == "__main__":
    run()
