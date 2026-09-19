"""M10 evidence: what each W4 backend really keeps resident per linear (TODO gap 5).

`w4.auto` dispatches at M > CROSSOVER_M = 0, so the v1 GEMV path is unreachable
on the main line — yet the v1 pack (`qfp`/`scale` buffers, the source of the
Marlin repack) used to stay resident next to Marlin's `_B`/`_s`. ARCHITECTURE §5
recorded that as a fact and left the size as arithmetic. This measures it:
per-backend byte totals over a genuinely swapped model, so the reclaimable figure
is a number instead of a guess.

After the release landed (`uses_v1_pack` in qslab/quant/w4_backends.py) the same
script is the before/after check: `dead v1` must read 0.000 on the Marlin paths
and their `resident` column must drop to the repack alone.

Env: MODEL, W4 (packed dir), BACKENDS (space-separated), UTIL.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.api import LLM

MODEL = os.environ.get("MODEL", "models/Qwen3-1.7B")
W4 = os.environ.get("W4", "models/Qwen3-1.7B-qslab-w4-awq2")
BACKENDS = os.environ.get("BACKENDS", "w4.auto w4.marlin w4.v1").split()
UTIL = float(os.environ.get("UTIL", "0.5"))
CALIB = os.environ.get("SMOOTH_KV", "results/smooth_kv4_qwen3-1.7b.pt"
                       if "1.7" in MODEL else "results/smooth_kv4_qwen3-8b.pt")


def _bytes(*ts):
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


def measure(llm):
    tot = dict(n=0, qfp=0, scale=0, act=0, repack=0, fp16=0)
    model = llm._engine.model_runner.model
    for m in model.modules():
        if type(m).__name__ != "W4Linear":
            continue
        b = m._backend
        mr = b._marlin if b.name == "w4.auto" else (b if b.name == "w4.marlin" else None)
        tot["n"] += 1
        # a Marlin layer releases the pack once repacked, so it has no qfp at all
        tot["qfp"] += _bytes(getattr(m, "qfp", None))
        tot["scale"] += _bytes(getattr(m, "scale", None))
        tot["act"] += _bytes(getattr(m, "act_scale", None))
        if mr is not None:
            tot["repack"] += _bytes(mr._B, mr._s, mr._ws)
        tot["fp16"] += m.in_features * m.out_features * 2
    tot["resident"] = tot["qfp"] + tot["scale"] + tot["act"] + tot["repack"]
    # the v1 pack is dead weight once nothing dispatches to it: w4.marlin never
    # does, and w4.auto only would below CROSSOVER_M (0 on the main line)
    tot["dead_v1"] = tot["qfp"] + tot["scale"] if tot["repack"] else 0
    return tot


def release(llm):
    import gc
    eng = llm._engine
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner, eng, llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    gb = 1024 ** 3
    calib = CALIB if Path(CALIB).exists() else None
    print(f"=== W4 weight residency: {MODEL} ({W4}) ===")
    print(f"{'backend':11} {'lines':>6} {'v1 pack':>10} {'repack':>10} {'act':>7} "
          f"{'resident':>9} {'vs fp16':>8} {'dead v1':>9}")
    for be in BACKENDS:
        llm = LLM(MODEL, w4=W4, w4_backend=be, smooth_kv=calib,
                  enforce_eager=True, gpu_memory_utilization=UTIL,
                  max_model_len=4096, max_num_seqs=4)
        t = measure(llm)
        pool = llm.kv_memory_bytes()
        print(f"{be:11} {t['n']:6d} {(t['qfp'] + t['scale']) / gb:>10.3f} "
              f"{t['repack'] / gb:>10.3f} {t['act'] / gb:>7.3f} "
              f"{t['resident'] / gb:>9.3f} "
              f"{t['resident'] / t['fp16']:>7.2f}x {t['dead_v1'] / gb:>8.3f}")
        print(f"{'':11} (GB) KV pool {calib or 'unsmoothed'} @util={UTIL}: "
              f"{pool / gb:.2f} GB -> {(pool + t['dead_v1']) / gb:.2f} GB if the "
              f"dead v1 pack were freed (+{100 * t['dead_v1'] / max(pool, 1):.0f}%)")
        release(llm)


if __name__ == "__main__":
    main()
