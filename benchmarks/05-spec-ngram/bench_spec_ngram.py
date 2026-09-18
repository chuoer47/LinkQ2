"""M9 bench: n-gram speculative decoding vs plain decode (design-m9 §6).

Three prompt families, reported honestly:
  copy    — text that repeats its own structure: the n-gram proposer's home
            turf (summarise/edit/code workload shape)
  natural — ordinary prose: low match rate, the overhead shows
  random  — random token ids: never matches, pure verify overhead

Metrics: decode-window tok/s (prefill steps excluded), average tokens
committed per verify step (1 = plain decode), verify step count.

Env: MODEL (default 1.7B), GAMMA (default 4), TOKENS (default 256),
W4 (packed dir, default none), UTIL, TAG. Greedy sampling throughout —
the acceptance rule is argmax equality (design-m9 §2).
"""
import os
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

N_TOKENS = int(os.environ.get("TOKENS", "256"))
GAMMA = int(os.environ.get("GAMMA", "4"))

COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of Spain is Madrid. "
               "The capital of France is")
NATURAL_PROMPT = ("The history of computing spans several centuries, from early "
                  "mechanical aids to modern electronic machines. Early devices "
                  "like the abacus assisted with calculation, and the "
                  "Difference Engine of the nineteenth century was designed for")


def random_prompt(tok, n=160):
    ids = tok.encode(NATURAL_PROMPT)[:32]
    rng = random.Random(7)
    vocab = tok.vocab_size
    return ids + [rng.randint(1000, 30000) for _ in range(n)]


def run_one(model, prompt, spec, gamma, w4=None, w4_backend=None, util=0.5):
    kw = {}
    calib = "results/smooth_kv4_qwen3-1.7b.pt" if "1.7" in model else \
            "results/smooth_kv4_qwen3-8b.pt"
    if os.path.exists(calib):
        kw["smooth_kv"] = calib
    if w4:
        kw["w4"] = w4
        if w4_backend:
            kw["w4_backend"] = w4_backend
    if spec:
        kw.update(spec_method="ngram", spec_num_drafts=gamma)
    eng = LLMEngine(model=model, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=False, gpu_memory_utilization=util, **kw)
    try:
        dec_time = 0.0
        committed = 0
        steps = 0
        eng.add_request(prompt, SamplingParams(temperature=1e-6,
                                                max_tokens=N_TOKENS,
                                                ignore_eos=True))
        while not eng.is_finished():
            t = perf_counter()
            out, num = eng.step()
            dt = perf_counter() - t
            if num < 0:                       # decode / verify step
                dec_time += dt
                committed += -num
                steps += 1
        return dict(tok_s=committed / dec_time, avg=committed / max(steps, 1),
                    steps=steps, dec_time=dec_time)
    finally:
        eng.exit()


def main():
    model = os.environ.get("MODEL", "models/Qwen3-1.7B")
    w4 = os.environ.get("W4")
    util = float(os.environ.get("UTIL", "0.5"))
    tag = os.environ.get("TAG", Path(model).name)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompts = {
        "copy": tok.encode(COPY_PROMPT),
        "natural": tok.encode(NATURAL_PROMPT),
        "random": random_prompt(tok),
    }

    print(f"=== n-gram spec bench: {tag} gamma={GAMMA} tokens={N_TOKENS} "
          f"w4={bool(w4)} ===")
    print(f"{'type':8} {'spec':5} {'tok/s':>8} {'tok/step':>9} {'steps':>6}")
    for name, prompt in prompts.items():
        base = run_one(model, prompt, spec=False, gamma=GAMMA, w4=w4, util=util)
        print(f"{name:8} {'off':5} {base['tok_s']:8.1f} {1.0:9.2f} {base['steps']:6d}")
        if w4:
            # a W4 spec step runs Marlin (M>=2 crossover), so the honest
            # speculation-only comparison needs a Marlin baseline too
            mbase = run_one(model, prompt, spec=False, gamma=GAMMA, w4=w4,
                            w4_backend="w4.marlin", util=util)
            print(f"{name:8} {'offM':5} {mbase['tok_s']:8.1f} {1.0:9.2f} {mbase['steps']:6d}")
            torch.cuda.empty_cache()
        else:
            mbase = base
        spec = run_one(model, prompt, spec=True, gamma=GAMMA, w4=w4, util=util)
        print(f"{name:8} {'on':5} {spec['tok_s']:8.1f} {spec['avg']:9.2f} {spec['steps']:6d}"
              f"   x{spec['tok_s'] / base['tok_s']:.2f}"
              f"{' (x%.2f vs marlin base)' % (spec['tok_s'] / mbase['tok_s']) if w4 else ''}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
