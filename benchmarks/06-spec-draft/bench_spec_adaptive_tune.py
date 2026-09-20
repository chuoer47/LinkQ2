"""Sweep the two unswept priors of the adaptive draft window.

`Scheduler._adapt_gamma` ships as a shrink-only ratchet: average the last
WINDOW=3 per-sequence accept lengths, and if that mean is <= 1.0 cut the window
to gamma//2. M10 settled *whether* to grow back (no — results/m10_draft_sweep.txt
reading 4) but reading 7 states plainly that the trigger window and the size of
the cut were never swept; they are M6 priors carried into the new runtime. This
script sweeps both axes.

Family (FAMILY env, whitespace-separated): natural is the sweep, because it is the
only family where the controller ever fires; copy is the guard — M10 read 6 found
the shipped ratchet never intervenes there, and any window this sweep finds worth
shipping has to leave that true, which is what FAMILY="natural copy" checks.

Cells (POLICIES, whitespace-separated; `w3h` IS the shipped rule, so reproducing
1.00-1.18x on it is the precondition for trusting the rest):
  off          adaptive off, window pinned at the ceiling — the 0.91x control
  wNh          moving average over N verify steps, cut to ceiling//2, N in 1 2 3 4 6
  w3rel        shipped window, cut relative: proposal_gamma//2 (can cascade to 1)
  w3m1         shipped window, give up one draft slot per trigger
  w3to1        shipped window, cut straight to no speculation
  w3thr15      shipped window + cut, trigger moved to avg <= 1.5

Timing follows benchmarks/06-spec-draft/bench_spec_draft.py: decode/verify steps
only, prefill excluded, and the same reporting discipline — W4 greedy trajectories
are chaotic, so tok/s is a band (+/-6-9% run to run on natural), not a per-token
claim. The steady signals are tok/step and *where the cut lands*, which is what
the summary rows lead with.

Env:
  MODEL     target            (default models/Qwen3-8B)
  W4        packed target dir (default models/Qwen3-8B-qslab-w4-awq; "" = fp16)
  DRAFT     draft model       (default models/Qwen3-0.6B)
  GAMMA     window ceiling    (default 4; spec_num_drafts, also the graph-family M)
  TOKENS    tokens per run    (default 192)
  REPEATS   runs per cell     (default 3; baselines are measured the same times)
  POLICIES  cell list         (default "off w1h w2h w3h w4h w6h w3rel w3m1 w3to1 w3thr15")
  FAMILY    prompt families   (default "natural"; add "copy" for the guard pass)
  TEMPERATURE (default 1e-6 = greedy; T>0 collapses acceptance — see
              bench_spec_temperature.py, do not expect the controller to bite)
  UTIL      target gpu_memory_utilization (default 0.62)
  DRAFT_UTIL (default 0.9)
"""
import os
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.sampling_params import SamplingParams

MODEL = os.environ.get("MODEL", "models/Qwen3-8B")
W4 = os.environ.get("W4", "models/Qwen3-8B-qslab-w4-awq")
DRAFT = os.environ.get("DRAFT", "models/Qwen3-0.6B")
GAMMA = int(os.environ.get("GAMMA", "4"))
N_TOKENS = int(os.environ.get("TOKENS", "192"))
REPEATS = int(os.environ.get("REPEATS", "3"))
TEMP = float(os.environ.get("TEMPERATURE", "1e-6"))
UTIL = float(os.environ.get("UTIL", "0.62"))
DRAFT_UTIL = float(os.environ.get("DRAFT_UTIL", "0.9"))
POLICIES = os.environ.get(
    "POLICIES", "off w1h w2h w3h w4h w6h w3rel w3m1 w3to1 w3thr15").split()

NATURAL_PROMPT = ("The history of computing spans several centuries, from early "
                  "mechanical aids to modern electronic machines. Early devices "
                  "like the abacus assisted with calculation, and the "
                  "Difference Engine of the nineteenth century was designed for")
COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of Spain is Madrid. "
               "The capital of France is")

#: default is natural-only (the family where the controller ever fires); copy is
#: the guard that a twitchy window must not cost anything on high-acceptance text
PROMPTS = {"natural": NATURAL_PROMPT, "copy": COPY_PROMPT}
FAMILIES = [f for f in os.environ.get("FAMILY", "natural").split() if f in PROMPTS]

#: cell -> (window, cut rule, threshold). w3h restates the shipped rule verbatim.
RULES = {
    "w1h": (1, "abs-halve", 1.0),
    "w2h": (2, "abs-halve", 1.0),
    "w3h": (3, "abs-halve", 1.0),
    "w4h": (4, "abs-halve", 1.0),
    "w6h": (6, "abs-halve", 1.0),
    "w3rel": (3, "rel-halve", 1.0),
    "w3m1": (3, "minus1", 1.0),
    "w3to1": (3, "to-1", 1.0),
    "w3thr15": (3, "abs-halve", 1.5),
}

_CUTS = {
    "abs-halve": lambda cur, ceil: max(1, ceil // 2),
    "rel-halve": lambda cur, ceil: max(1, cur // 2),
    "minus1": lambda cur, ceil: max(1, cur - 1),
    "to-1": lambda cur, ceil: 1,
}

_SHIP = Scheduler._adapt_gamma  # restored verbatim between cells


def _install(rule):
    """Point Scheduler._adapt_gamma at (window, cut, threshold) for one cell."""
    if rule is None:
        Scheduler._adapt_gamma = _SHIP
        return

    window, cut, thr = rule

    def adapt(self):
        recent = list(self._recent)[-window:]
        if not recent:
            return
        if sum(recent) / len(recent) > thr:
            return
        self.proposal_gamma = _CUTS[cut](self.proposal_gamma, self.gamma)

    Scheduler._adapt_gamma = adapt


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


def run_one(rule, *, adaptive, gamma, prompt, n_tokens=N_TOKENS):
    """One generation, timed over the decode/verify window only.

    `windows[i]` is the window the scheduler reports *after* verify step i+1,
    i.e. what the next proposal round will spend — the same convention as
    bench_spec_draft.py, so the two files' trajectories are comparable.
    """
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
                  spec_adaptive_gamma=adaptive)
    _install(rule)
    eng = LLMEngine(model=MODEL, **kw)
    try:
        sch = eng.scheduler
        if adaptive and rule is not None:
            # the deque ships at maxlen=4; a wider window needs widening first,
            # otherwise list(_recent)[-6:] silently reads 4 entries
            sch._recent = deque(sch._recent, maxlen=max(4, rule[0]))
        dec = committed = 0.0
        steps = 0
        windows = []
        eng.add_request(prompt, SamplingParams(
            temperature=TEMP, max_tokens=n_tokens, ignore_eos=True))
        while not eng.is_finished():
            t = perf_counter()
            _, num = eng.step()
            dt = perf_counter() - t
            if num < 0:
                dec += dt
                committed += -num
                steps += 1
                windows.append(sch.proposal_gamma)
        acc = dict(sch.spec_stats).get("acceptance_rate", 0.0)
        cut = next((i for i, g in enumerate(windows, 1) if g < gamma), None)
        ladder, prev = [gamma], gamma
        for g in windows:
            if g != prev:
                ladder.append(g)
                prev = g
        return dict(tok_s=committed / max(dec, 1e-9),
                    avg=committed / max(steps, 1),
                    ms=1e3 * dec / max(steps, 1),
                    acc=acc, steps=steps, cut=cut,
                    ladder="->".join(str(g) for g in ladder),
                    held=sum(1 for g in windows if g == gamma))
    finally:
        _release(eng)


def run_family(fam):
    """Sweep every cell against this family's own plain-decode baseline."""
    prompt = PROMPTS[fam]
    print(f"\n===== family: {fam} =====")
    base = []
    for i in range(REPEATS):
        b = run_one(None, adaptive=False, gamma=0, prompt=prompt)
        base.append(b["tok_s"])
        print(f"[baseline] rep {i + 1}/{REPEATS}: {b['tok_s']:.1f} tok/s "
              f"({b['ms']:.2f} ms/step, {b['steps']} steps)")
    bmean = sum(base) / len(base)
    print(f"[baseline] mean {bmean:.1f} tok/s, spread "
          f"{min(base):.1f}-{max(base):.1f} ({100 * (max(base) / min(base) - 1):.1f}%)\n")

    print(f"{'cell':>8} | {'rep':>3} | {'tok/s':>6} {'x':>5} | {'tok/step':>8} "
          f"{'ms/step':>7} | {'acc':>5} | {'cut@step':>8} | ladder | held@ceiling")
    summary = {}
    for name in POLICIES:
        rule = None if name == "off" else RULES[name]
        rows = []
        for i in range(REPEATS):
            r = run_one(rule, adaptive=rule is not None, gamma=GAMMA, prompt=prompt)
            r["x"] = r["tok_s"] / bmean
            rows.append(r)
            print(f"{name:>8} | {i + 1:>3} | {r['tok_s']:6.1f} {r['x']:5.2f} | "
                  f"{r['avg']:8.2f} {r['ms']:7.2f} | {r['acc']:5.2f} | "
                  f"{str(r['cut'] or 'never'):>8} | {r['ladder']:>8} | "
                  f"{r['held']}/{r['steps']}")
        xs = [r["x"] for r in rows]
        cuts = [r["cut"] for r in rows if r["cut"]]
        summary[name] = dict(mean=sum(xs) / len(xs), lo=min(xs), hi=max(xs),
                             avg=sum(r["avg"] for r in rows) / len(rows),
                             cuts=cuts, ladders=sorted({r["ladder"] for r in rows}),
                             held=sum(r["held"] / max(1, r["steps"]) for r in rows) / len(rows))
        s = summary[name]
        print(f"{name:>8} | n={REPEATS} | mean {s['mean']:.2f}x band "
              f"[{s['lo']:.2f}, {s['hi']:.2f}] | tok/step {s['avg']:.2f} | "
              f"cut@{(min(s['cuts']) if s['cuts'] else 'never')}"
              f"{'-' + str(max(s['cuts'])) if len(s['cuts']) > 1 else ''} | "
              f"{'/'.join(s['ladders'])} | held {100 * s['held']:.0f}%\n")

    ship = summary.get("w3h")
    print(f"--- 排序 [{fam}]（mean x，越高越好；对照 = w3h 即出厂棘轮） ---")
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        d = "" if ship is None else f"  ({'+' if s['mean'] >= ship['mean'] else ''}" \
            f"{100 * (s['mean'] / ship['mean'] - 1):.1f}% vs w3h)"
        print(f"{name:>8}: {s['mean']:.2f}x band [{s['lo']:.2f}, {s['hi']:.2f}] "
              f"tok/step {s['avg']:.2f}{d}")
    return summary


def main():
    print(f"=== adaptive draft window sweep: {Path(MODEL).name}"
          f"{'-W4A16KV4' if W4 else '-fp16'} + {Path(DRAFT).name}, graph, "
          f"T={TEMP:g}, bs=1, gamma<={GAMMA}, {N_TOKENS} tok, "
          f"families={'+'.join(FAMILIES)} ===")
    for fam in FAMILIES:
        run_family(fam)


if __name__ == "__main__":
    main()
