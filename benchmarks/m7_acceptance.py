"""M7-S3 acceptance: 8B paged-vs-dense KV4 comparison.

1. PPL (WikiText-2, 16 docs): paged must be within 0.05 of dense KV4.
2. Memory at 8K context: dense KV4's transient fp16 copy vs paged peak.
   (The whole point of M7: the O(context) fp16 copy disappears.)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

MODEL = "/home/<user>/<workdir>/qserve-lab/models/Qwen3-8B"
W4 = "/home/<user>/<workdir>/qserve-lab/models/Qwen3-8B-qslab-w4-awq"
PLAN = "/home/<user>/<workdir>/qserve-lab/results/kv4_plan_8b.json"


def run_mode(mode: str, ctx_tokens: int | None = None):
    from qslab.config import EngineConfig
    from qslab.engine import QslabEngine
    from qslab.models.w4linear import swap_w4_linears
    from adapters.tokenizer import QwenTokenizerAdapter

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = EngineConfig(model_path=MODEL, device="cuda:0")
    eng = QslabEngine(cfg, kv_mode=mode, kv_plan_path=PLAN)
    swap_w4_linears(eng.model, W4)
    tok = QwenTokenizerAdapter(MODEL)

    res = {"mode": mode}
    if ctx_tokens:
        ids = tok.encode(" long context memory test ")
        while len(ids) < ctx_tokens:
            ids = ids + ids
        ids = ids[:ctx_tokens]
        eng.reset_cache()
        eng.prefill(ids)
        res["kv_packed_mb"] = round(
            sum(c.memory_bytes_valid() for c in eng.kv_caches) / 1e6, 2)
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    else:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 200][:16]
        from benchmarks.bench_ppl import ppl_of_model
        ppl = ppl_of_model(eng.model, tok, texts, "cuda:0",
                           reset_fn=lambda: [c.reset() for c in eng.kv_caches])
        res["ppl"] = round(ppl, 4)
    del eng
    torch.cuda.empty_cache()
    return res


print("== PPL: dense vs paged (8B W4) ==", flush=True)
r1 = run_mode("kv4")
r2 = run_mode("kv4.paged")
print(f"dense kv4     PPL {r1['ppl']}")
print(f"paged kv4     PPL {r2['ppl']}")
print(f"difference    {r2['ppl'] - r1['ppl']:+.4f} (line: <0.05)")

print("\n== Memory @8K context ==", flush=True)
CTX = 8192
m1 = run_mode("kv4", ctx_tokens=CTX)
m2 = run_mode("kv4.paged", ctx_tokens=CTX)
print(f"dense kv4: packed {m1['kv_packed_mb']} MB | peak {m1['peak_mem_gb']} GB")
print(f"paged kv4: packed {m2['kv_packed_mb']} MB | peak {m2['peak_mem_gb']} GB")

report = {"ppl": {"dense": r1, "paged": r2}, "mem": {"dense": m1, "paged": m2},
          "timestamp": datetime.now().isoformat()}
Path("/home/<user>/<workdir>/qserve-lab/results/m7_acceptance.json").write_text(
    json.dumps(report, indent=2))
print("saved results/m7_acceptance.json")
