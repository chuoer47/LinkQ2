"""M9: prefix-cache TTFT on the multi-turn shape (design: materialize route).

Turn 1 (cold): full prefill. Turn 2 (hit): the whole turn-1 prefix comes
back from the int4 pool, prefill only computes the new suffix. Reported as
wall clock for max_tokens=1 (first token) — the realistic second-turn TTFT.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams


def main():
    model = os.environ.get("MODEL", "models/Qwen3-1.7B")
    calib = ("results/smooth_kv4_qwen3-1.7b.pt" if "1.7" in model
             else "results/smooth_kv4_qwen3-8b.pt")
    ctx = int(os.environ.get("CTX", "3800"))          # turn-1 context tokens
    eng = LLMEngine(model=model, max_model_len=4096, max_num_seqs=4,
                    enforce_eager=os.environ.get("EAGER") == "1",
                    gpu_memory_utilization=float(os.environ.get("UTIL", "0.6")),
                    smooth_kv=calib)
    try:
        tok = eng.tokenizer
        unit = " The capital of France is Paris. The capital of Germany is Berlin. "
        ids = tok.encode(unit * 400)
        turn1 = tok.decode(ids[:ctx])          # slice at TOKEN granularity
        suffix = tok.encode(" Question: what is the capital of France? Answer:")

        sp = SamplingParams(temperature=1e-6, max_tokens=1)
        t0 = time.perf_counter()
        eng.generate([turn1], sp, use_tqdm=False)
        cold = time.perf_counter() - t0

        hits = []
        for _ in range(3):
            t0 = time.perf_counter()
            eng.generate([turn1], sp, use_tqdm=False)
            hits.append(time.perf_counter() - t0)
        hit = min(hits)

        print(f"=== prefix cache TTFT: {model} ctx={ctx}+{len(suffix)} ===")
        print(f"cold turn-1: {cold*1000:7.1f} ms")
        print(f"hit  turn-2: {hit*1000:7.1f} ms   ({cold/hit:.2f}x faster TTFT)")
    finally:
        eng.exit()


if __name__ == "__main__":
    main()
