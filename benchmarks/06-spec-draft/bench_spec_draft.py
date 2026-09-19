"""M9 draft-model speculation bench, now as a real script (TODO 缺环1).

This closes the gap the M9 notes admitted: results/m9_spec_draft.txt,
m9_gamma_sweep_draft.txt and m9_draft_fused.txt were produced by inline
snippets, so nothing in the repo could re-run them. Shape follows
benchmarks/05-spec-ngram/bench_spec_ngram.py (same prompts, same decode-window
timing, same "never compare by token equality" discipline).

What it measures
  baseline   plain decode, no speculation, per prompt family
  sweep      draft-model speculation (Qwen3-0.6B) across gamma, tok/s +
             tokens committed per verify step + acceptance rate

Two honesty notes about reproducing the M9 底稿:
  * DraftProposer hardcodes compile=True for the draft runtime
    (qslab/runtime/draft.py), i.e. the inductor-fused variant is the only one
    reachable from config now. Rows here are comparable against
    results/m9_draft_fused.txt, NOT against the unfused table in
    results/m9_gamma_sweep_draft.txt (that comparison needs the M9 code).
  * the 2241-kernels/step and CUDA-event breakdowns in the M9 notes came from
    a profiler run, which this script does not do.

Env:
  MODEL    target            (default models/Qwen3-8B)
  W4        packed target    (default models/Qwen3-8B-qslab-w4-awq; "" = fp16)
  DRAFT    draft model       (default models/Qwen3-0.6B)
  DRAFT_W4 packed draft dir  (default "" — M9 falsified W4 for the draft:
             kernel count, not bytes, was the tax)
  GAMMAS   gamma list        (default "1 2 3 4 6 8")
  TOKENS   tokens per run    (default 192)
  TEMPERATURE (default 1e-6 = greedy; see bench_spec_temperature.py for a sweep)
  ADAPTIVE on/off           (default off; spec_adaptive_gamma only bites the
             draft proposer, which is the one that pays a forward per proposal)
  UTIL     target gpu_memory_utilization (default 0.62)
  DRAFT_UTIL (default 0.9)
"""
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

MODEL = os.environ.get("MODEL", "models/Qwen3-8B")
W4 = os.environ.get("W4", "models/Qwen3-8B-qslab-w4-awq")
DRAFT = os.environ.get("DRAFT", "models/Qwen3-0.6B")
DRAFT_W4 = os.environ.get("DRAFT_W4") or None
GAMMAS = [int(g) for g in os.environ.get("GAMMAS", "1 2 3 4 6 8").split()]
N_TOKENS = int(os.environ.get("TOKENS", "192"))
TEMP = float(os.environ.get("TEMPERATURE", "1e-6"))
ADAPTIVE = os.environ.get("ADAPTIVE", "off").lower() in ("1", "on", "true")
UTIL = float(os.environ.get("UTIL", "0.62"))
DRAFT_UTIL = float(os.environ.get("DRAFT_UTIL", "0.9"))

COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of Spain is Madrid. "
               "The capital of France is")
NATURAL_PROMPT = ("The history of computing spans several centuries, from early "
                  "mechanical aids to modern electronic machines. Early devices "
                  "like the abacus assisted with calculation, and the "
                  "Difference Engine of the nineteenth century was designed for")


def _release(eng):
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


def run_one(prompt, *, gamma):
    """One generation, timed over the decode/verify window only."""
    kw = dict(max_model_len=4096, max_num_seqs=4, enforce_eager=False,
              gpu_memory_utilization=UTIL)
    calib = "results/smooth_kv4_" + Path(MODEL).name.lower() + ".pt"
    if os.path.exists(calib):
        kw["smooth_kv"] = calib
    if W4:
        kw["w4"] = W4
    if gamma:
        kw.update(spec_method="draft", spec_num_drafts=gamma,
                  draft_model=DRAFT, draft_gpu_memory_utilization=DRAFT_UTIL,
                  spec_adaptive_gamma=ADAPTIVE)
        if DRAFT_W4:
            kw["draft_w4"] = DRAFT_W4
    eng = LLMEngine(model=MODEL, **kw)
    try:
        dec = committed = 0.0
        steps = 0
        window = []
        eng.add_request(prompt, SamplingParams(temperature=TEMP,
                                               max_tokens=N_TOKENS,
                                               ignore_eos=True))
        while not eng.is_finished():
            t = perf_counter()
            _, num = eng.step()
            dt = perf_counter() - t
            if num < 0:
                dec += dt
                committed += -num
                steps += 1
                if gamma and ADAPTIVE:
                    window.append(eng.scheduler.proposal_gamma)
        st = dict(eng.scheduler.spec_stats)
        return dict(tok_s=committed / max(dec, 1e-9),
                    avg=committed / max(steps, 1),
                    ms=1e3 * dec / max(steps, 1),
                    acc=st.get("acceptance_rate", 0.0),
                    win=window)
    finally:
        _release(eng)


def main():
    fams = {"natural": NATURAL_PROMPT, "copy": COPY_PROMPT}
    print(f"=== draft spec: {Path(MODEL).name}"
          f"{'-W4A16KV4' if W4 else '-fp16'} + {Path(DRAFT).name}"
          f"{'-W4' if DRAFT_W4 else ''}, graph, T={TEMP:g}, bs=1, "
          f"{N_TOKENS} tok, adaptive={'on' if ADAPTIVE else 'off'} ===")
    base = {n: run_one(p, gamma=0) for n, p in fams.items()}
    print("baseline plain: " + " | ".join(
        f"{n} {b['tok_s']:.1f} tok/s ({b['ms']:.2f} ms/step)"
        for n, b in base.items()))
    print(f"{'gamma':>5} | {'natural tok/s (x) tok/step':>28} | "
          f"{'copy tok/s (x) tok/step':>28} | acc nat/copy")
    for g in GAMMAS:
        cells, accs, wins = [], [], {}
        for n, p in fams.items():
            r = run_one(p, gamma=g)
            x = r["tok_s"] / base[n]["tok_s"]
            cells.append(f"{r['tok_s']:8.1f} ({x:.2f}x) {r['avg']:5.2f}   ")
            accs.append(f"{r['acc']:.2f}")
            if r["win"]:
                lo, hi = min(r["win"]), max(r["win"])
                wins[n] = (lo, hi, r["win"][-1])
        print(f"{g:>5} | {cells[0]} | {cells[1]} | {'/'.join(accs)}")
        for n, (lo, hi, last) in wins.items():
            print(f"{'':>5}   adaptive window on {n}: in [{lo}, {hi}], "
                  f"ends at {last} (ceiling = spec_num_drafts={g})")


if __name__ == "__main__":
    main()
