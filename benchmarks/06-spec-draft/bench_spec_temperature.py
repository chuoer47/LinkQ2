"""Does the Leviathan ratio path cost the speculation gain?

MODEL      target model dir (default models/Qwen3-1.7B)
METHOD     proposers to sweep (default ngram; "lookahead" "draft")
GAMMA      draft window (default 4)
TOKENS     tokens generated per run (default 256)
TEMPS      temperatures to sweep (default "1e-6 0.7")
W4         packed W4 dir for the target (default none)
DRAFT      draft model dir when METHOD includes draft
UTIL       gpu_memory_utilization (default 0.5)
PROMPTS    prompt families               Timing covers decode/verify steps only (LLMEngine.step
             returns a negative count for those), so prefill length does not move the numbers.
             (default "copy natural random")"""
# At T>0 a verify step can pay a [bs*(gamma+1), V] float32 proposal distribution plus a
#   vocab-wide softmax, a ratio draw and a residual resample; one-hot proposers (n-gram,
#   lookahead, a greedy draft) skip the matrix.
# Peak allocated bytes are reported per run so the fp32 proposal matrix shows up as a number
#   instead of an argument.
# Timing covers decode/verify steps only (step() returns a negative count for those), so
#   prefill length does not move the numbers.
import os
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

GAMMA = int(os.environ.get("GAMMA", "4"))
N_TOKENS = int(os.environ.get("TOKENS", "256"))
TEMPS = [float(t) for t in os.environ.get("TEMPS", "1e-6 0.7").split()]
METHODS = os.environ.get("METHOD", "ngram").split()
PROMPTS = os.environ.get("PROMPTS", "copy natural random").split()
UTIL = float(os.environ.get("UTIL", "0.5"))

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
    return ids + [rng.randint(1000, 30000) for _ in range(n)]


def _release(eng):
    """Drop the engine and everything holding the card."""
    # A sweep here builds and tears down a dozen engines in one process, so freeing the int4
    #   pools between runs matters more than in a single-shot bench.
    prop = getattr(eng.scheduler, "proposer", None) if hasattr(eng, "scheduler") else None
    if prop is not None and hasattr(prop, "exit"):
        prop.exit()
    if hasattr(eng.scheduler, "proposer"):
        eng.scheduler.proposer = None
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    eng.exit()
    torch.cuda.empty_cache()


def run_one(model, prompt, *, method, gamma, temp, w4, draft):
    kw = {}
    calib = ("results/smooth_kv4_qwen3-1.7b.pt" if "1.7" in model
             else "results/smooth_kv4_qwen3-8b.pt")
    if os.path.exists(calib):
        kw["smooth_kv"] = calib
    if w4:
        kw["w4"] = w4
    # one memory budget for both sides of every comparison — the off/on peak
    # delta is only meaningful if the KV pool behind it is the same size
    kw.update(max_model_len=4096, max_num_seqs=8, enforce_eager=False,
              gpu_memory_utilization=UTIL)
    if method:
        kw["spec_method"] = method
        kw["spec_num_drafts"] = gamma
        if method == "draft":
            # two int4 pools on one card — the split test_8b_acceptance.py uses
            kw.update(draft_model=draft, max_num_seqs=4,
                      gpu_memory_utilization=0.62,
                      draft_gpu_memory_utilization=0.9)
    eng = LLMEngine(model=model, **kw)
    try:
        dec = committed = steps = 0.0
        torch.cuda.reset_peak_memory_stats()
        eng.add_request(prompt, SamplingParams(temperature=temp,
                                               max_tokens=N_TOKENS,
                                               ignore_eos=True))
        while not eng.is_finished():
            t = perf_counter()
            _, num = eng.step()
            dt = perf_counter() - t
            if num < 0:                       # decode / verify step
                dec += dt
                committed += -num
                steps += 1
        st = dict(eng.scheduler.spec_stats)
        return dict(tok_s=committed / max(dec, 1e-9),
                    avg=committed / max(steps, 1),
                    steps=int(steps),
                    acc=st.get("acceptance_rate", 0.0),
                    peak=torch.cuda.max_memory_allocated() / 1e9)
    finally:
        _release(eng)


def main():
    model = os.environ.get("MODEL", "models/Qwen3-1.7B")
    w4 = os.environ.get("W4")
    draft = os.environ.get("DRAFT", "models/Qwen3-0.6B")
    tag = os.environ.get("TAG", Path(model).name)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    fams = {"copy": tok.encode(COPY_PROMPT),
            "natural": tok.encode(NATURAL_PROMPT),
            "random": random_prompt(tok)}

    print(f"=== ratio-path spec bench: {tag} gamma={GAMMA} tokens={N_TOKENS} "
          f"w4={bool(w4)} temps={TEMPS} ===")
    for method in METHODS:
        for name in PROMPTS:
            prompt = fams[name]
            rows = []
            for temp in TEMPS:
                base = run_one(model, prompt, method=None, gamma=GAMMA,
                               temp=temp, w4=w4, draft=draft)
                spec = run_one(model, prompt, method=method, gamma=GAMMA,
                               temp=temp, w4=w4, draft=draft)
                rows.append((base, spec))
                print(f"{method:9} {name:8} T={temp:<6g} off {base['tok_s']:7.1f} "
                      f"on {spec['tok_s']:7.1f}  x{spec['tok_s'] / base['tok_s']:.2f}"
                      f"  tok/step {spec['avg']:.2f} acc {spec['acc']:.2f} "
                      f"peak {base['peak']:.2f}->{spec['peak']:.2f} GB")
            if len(rows) > 1:
                # the lossless tax is the difference between these two speedups,
                # both measured against their own same-temperature baseline
                first, last = rows[0], rows[-1]
                g = first[1]["tok_s"] / first[0]["tok_s"]
                t = last[1]["tok_s"] / last[0]["tok_s"]
                print(f"{'':19}lossless tax: x{g:.2f} at T={TEMPS[0]:g} -> "
                      f"x{t:.2f} at T={TEMPS[-1]:g}   "
                      f"(tok/s cost {(1 - t / g) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
