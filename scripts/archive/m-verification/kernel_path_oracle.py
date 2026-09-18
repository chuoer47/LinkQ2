"""M1-S9d: engine with kernel-path W4 (W4Linear) — oracle alignment + e2e bench.

Correctness: kernel-path engine vs load_w4_model (dequant) reference, token match.
Perf: tokens/s FP16 vs W4-kernel path.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

PACKED = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-qslab-w4-awq2"

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.models.loader import load_reference_model, load_model_config
from qslab.models.w4linear import swap_w4_linears
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
prompts = [" The capital of France is", " large language model quantization is"]
n_new = 32

# --- correctness: kernel path engine vs dequant reference ---
mc = load_model_config("models/Qwen3-1.7B")
cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=n_new)
eng = QslabEngine(cfg)
n_swapped = swap_w4_linears(eng.model, PACKED)
print(f"swapped {n_swapped} linears -> W4Linear (kernel path)")

from qslab.models.loader import load_w4_model
ref = load_w4_model(PACKED)
all_match = True
for p in prompts:
    ids = tok.encode(p)
    out_eng = eng.generate(ids, max_new_tokens=n_new)
    x = torch.tensor([ids], device="cuda:0")
    with torch.inference_mode():
        out_ref = ref.generate(x, max_new_tokens=n_new, do_sample=False)
    gen_ref = out_ref[0][len(ids):].tolist()
    m = out_eng == gen_ref
    all_match &= m
    print(f"match={m} | {tok.decode(out_eng)[:50]!r}")
print("KERNEL_PATH_ALL_MATCH:", all_match)
del ref
torch.cuda.empty_cache()

# --- perf: fp16 vs w4-kernel, same protocol as bench_throughput ---
def make_ids(n):
    text = (" large language model inference requires careful memory management "
            "and quantization reduces the weight footprint significantly while ")
    ids = tok.encode(text)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]

def bench_engine(eng, input_ids, n_decode=128):
    lat = []
    eng.reset_cache()
    logits = eng.prefill(input_ids)
    start = len(input_ids)
    nxt = int(logits.argmax(dim=-1))
    for i in range(n_decode - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = eng.decode_step(nxt, start + i)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        nxt = int(logits.argmax(dim=-1))
    lat = lat[-100:]
    lat.sort()
    return 1.0 / lat[len(lat) // 2]

input_ids = make_ids(512)
# fp16 baseline (fresh engine, unswapped — reuse ref model instead)
eng_fp16 = QslabEngine(cfg)
tp_fp16 = bench_engine(eng_fp16, input_ids)
del eng_fp16
torch.cuda.empty_cache()
# w4 kernel engine (already built above)
tp_w4 = bench_engine(eng, input_ids)
print(f"FP16: {tp_fp16:.2f} tok/s | W4-kernel: {tp_w4:.2f} tok/s | speedup {tp_w4/tp_fp16:.2f}x")
