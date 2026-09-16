"""M2-S5 v2: NIAH at 4K/8K context (1.7B-model-scaled), cleaner harness.

32K NIAH is beyond Qwen3-1.7B's effective retrieval even at fp16 (0% both
modes — model limitation, not quantization). We test at sizes where the
fp16 baseline can actually retrieve, so the KV4 vs FP16 *delta* is meaningful.
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

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
DEPTHS = [0.25, 0.5, 0.75]
N_TRIALS = 3
FILLER = ("The sun rises over the quiet hills and the village begins another "
          "ordinary day. Farmers walk along the road, birds cross the sky, "
          "and life moves at its familiar gentle pace. ")


def build_prompt(tok, key: str, value: str, depth: float, ctx: int) -> list[int]:
    needle = f"One of the special magic numbers for {key} is: {value}. "
    fill_ids = tok.encode(FILLER)
    hay = (fill_ids * (ctx // len(fill_ids) + 1))[:ctx - 200]
    pos = int(len(hay) * depth)
    body = tok.decode(hay[:pos]) + needle + tok.decode(hay[pos:])
    user_msg = f"{body}\n\nQuestion: What is the special magic number for {key}?"
    # chat template wrapper: Qwen3 needs it for instruction following;
    # disable thinking mode (its reasoning eats the token budget)
    chat = tok._tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return chat[:ctx]


def run():
    tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
    rng = random.Random(2026)
    results = {}
    for mode in ["fp16", "kv4"]:
        cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0",
                           max_new_tokens=32)
        eng = QslabEngine(cfg, kv_mode=mode, kv_plan_path="results/kv4_plan.json")
        per_depth = []
        for depth in DEPTHS:
            hit = 0
            for t in range(N_TRIALS):
                key = f"magic-{rng.randint(1000, 9999)}"
                value = str(rng.randint(100000, 999999))
                ids = build_prompt(tok, key, value, depth, CTX)
                out = eng.generate_chunked(ids, max_new_tokens=32, chunk=1024)
                ans = tok.decode(out)
                h = value in ans
                hit += int(h)
                print(f"  [{mode} d={depth} t={t}] ans={ans.strip()[:20]!r} "
                      f"exp={value} hit={h}", flush=True)
            per_depth.append({"depth": depth, "hits": hit, "trials": N_TRIALS})
        rec = sum(r["hits"] for r in per_depth) / sum(r["trials"] for r in per_depth)
        results[mode] = {"overall_recall": round(rec, 3), "by_depth": per_depth}
        print(f"[{mode}] overall recall {rec:.2%}", flush=True)
        del eng
        torch.cuda.empty_cache()

    out = {"ctx": CTX, "model": "Qwen3-1.7B", "results": results,
           "timestamp": datetime.now().isoformat()}
    Path("results/m2_niah.json").write_text(json.dumps(out, indent=2))
    print("saved results/m2_niah.json")


if __name__ == "__main__":
    run()
