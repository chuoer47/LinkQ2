"""M11: NIAH at 32K/64K/128K through the paged runtime (W4A16 + KV4) with YaRN.

Protocol: needle-in-a-haystack with the filler / needle / question wording of
benchmarks/01-decode-baseline/bench_niah.py (M2), so the 32K row stays
comparable to that 底稿 — but this one drives the new runtime and the 8B W4
pack, and it reaches past the 40960-position ceiling Qwen3-8B ships with.

What each default column means:

  32768:native    the model's own window. The anchor: retrieval with rope untouched.
  32768:yarn:3.2  the same prompt, the 128K scaling applied anyway. YaRN compresses
                  every frequency, so this is a real perturbation and not a no-op —
                  it prices the in-window cost of the extension.
  65536:yarn      past the native ceiling. Without YaRN these two cannot even be
  131072:yarn     admitted: Config clamps max_model_len to hf_config's 40960.

Trials are paired, not resampled: the same (depth, key, value, prompt length) list
is replayed for every column, and needle RNG is fixed. Only rope differs. Each
trial's haystack starts at a different phase of the filler, so no column gets a
prefix-cache-warm prefill and every prefill number is a cold one.

Not measured, deliberately:
  * an unscaled-rope run at 64K/128K as the negative control. The ceiling clamp is
    the only gate, so building it would mean hand-patching hf_config inside a bench;
    the results file records that as untested instead.
  * long-context PPL. The sliding-window protocol needs per-position logits, and at
    131072 positions that is a 40 GB [N, vocab] tensor — the protocol does not reach.

Budget (why UTIL defaults to 0.8): the int4 pool charges 36 layers x 8 KV heads x
256 B = 73.7 KB per token, so a 128K sequence needs ~9.4 GiB of pool. On top of that,
a chunked prefill dequantizes the whole resident prefix once per layer
(2 x 131072 x 8 x 128 B ~= 0.5 GiB, plus the concat that adds the new rows), and
that transient has to live in the (1 - UTIL) remainder.

Env:
  TIERS     whitespace list of CTX[:mode[:factor]]  (default as above)
  MODEL     target                    (default models/Qwen3-8B)
  W4        packed target dir         (default models/Qwen3-8B-qslab-w4-awq; "" = fp16)
  CALIB     SmoothAttention file      (default results/smooth_kv4_qwen3-8b.pt; "" = off)
  TRIALS    needle trials per depth   (default 2)
  DEPTHS    whitespace list           (default 0.10 0.50 0.90)
  GEN       tokens to sample          (default 32)
  UTIL      gpu_memory_utilization    (default 0.8)
  SEED      needle RNG                (default 2026)
"""
import gc
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

MODEL = os.environ.get("MODEL", "models/Qwen3-8B")
W4 = os.environ.get("W4", "models/Qwen3-8B-qslab-w4-awq")
CALIB = os.environ.get("CALIB", "results/smooth_kv4_qwen3-8b.pt")
TIERS = os.environ.get("TIERS", "32768:native 32768:yarn:3.2 65536:yarn 131072:yarn")
DEPTHS = [float(d) for d in os.environ.get("DEPTHS", "0.10 0.50 0.90").split()]
N_TRIALS = int(os.environ.get("TRIALS", "2"))
GEN = int(os.environ.get("GEN", "32"))
UTIL = float(os.environ.get("UTIL", "0.8"))
SEED = int(os.environ.get("SEED", "2026"))

# M2's filler, verbatim, so haystacks are the same text at the same token cost
FILLER = ("The sun rises over the quiet hills and the village begins another "
          "ordinary day. Farmers walk along the road, birds cross the sky, "
          "and life moves at its familiar gentle pace. ")


def native_ceiling() -> int:
    from transformers import AutoConfig
    return int(AutoConfig.from_pretrained(MODEL).max_position_embeddings)


def parse_tiers(native: int):
    """CTX[:mode[:factor]] -> (ctx, mode, factor, ceiling).

    `ceiling` is what the rope cache and Config's clamp will actually allow, so
    the bench can refuse an impossible prompt before spending a prefill on it.
    """
    out = []
    for spec in TIERS.split():
        parts = spec.split(":")
        ctx = int(parts[0])
        mode = parts[1] if len(parts) > 1 else "native"
        if mode == "native":
            out.append((ctx, "native", None, native))
            continue
        if len(parts) > 2:
            factor = float(parts[2])
        else:
            assert ctx > native, f"{spec}: no factor given and {ctx} is not past the native ceiling {native}"
            factor = math.ceil(ctx / native * 100) / 100
            assert int(native * factor) >= ctx, f"{spec}: factor {factor} still cannot reach {ctx}"
        out.append((ctx, "yarn", factor, int(native * factor)))
    return out


def build_prompt(tok, key: str, value: str, depth: float, hay_len: int,
                 phase: int = 0) -> list[int]:
    """Haystack of exactly hay_len filler tokens with the needle spliced in.

    `phase` shifts where the haystack starts inside the repeated filler, so
    consecutive trials share no prefix: without it the engine's prefix cache
    serves trials 2..n a warm haystack and the column's prefill times are
    meaningless (the first version of this bench measured 6.7 -> 1.0 s across
    one 32K column). Retrieval itself is unaffected — a cached prefix holds
    the same KV.
    """
    needle = f"One of the special magic numbers for {key} is: {value}. "
    fill_ids = tok.encode(FILLER)
    rep = fill_ids * (hay_len // len(fill_ids) + 3)
    hay = rep[phase:phase + hay_len]
    pos = int(len(hay) * depth)
    body = tok.decode(hay[:pos]) + needle + tok.decode(hay[pos:])
    user_msg = f"{body}\n\nQuestion: What is the special magic number for {key}?"
    # Qwen3 needs the chat wrapper to follow the question at all, and its
    # thinking mode would spend the whole answer budget on reasoning
    chat = tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return chat[:hay_len]


def run_one(eng, ids):
    """One greedy request, with prefill and decode wall time kept apart."""
    eng.add_request(ids, SamplingParams(temperature=1e-6, max_tokens=GEN))
    prefill = decode = 0.0
    completion: list[int] = []
    while not eng.is_finished():
        t = time.perf_counter()
        outputs, n = eng.step()
        dt = time.perf_counter() - t
        if n > 0:
            prefill += dt
        else:
            decode += dt
        for _, toks in outputs:
            completion = toks
    return completion, prefill, decode


def release(eng):
    """Give the card back before the next column builds its engine.

    The attention layers hold views into the int4 pool, and exit() only drops
    the runner, so a plain `del eng` leaves the whole pool allocated and the
    second column OOMs during its weight load (observed). This is the same
    teardown tests/test_draft_spec.py::_release does.
    """
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.exit()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    native = native_ceiling()
    print(f"model={MODEL} w4={W4 or 'fp16'} calib={CALIB or 'off'} native_ceiling={native}")
    print(f"tiers={TIERS} depths={DEPTHS} trials/depth={N_TRIALS} gen={GEN} util={UTIL}")
    rng = random.Random(SEED)
    cases = [(f"magic-{rng.randint(1000, 9999)}", str(rng.randint(100000, 999999)), d)
             for d in DEPTHS for _ in range(N_TRIALS)]

    for ctx, mode, factor, ceiling in parse_tiers(native):
        tag = f"{ctx // 1024}K:{mode}" + (f"{factor}" if factor else "")
        if ceiling < ctx:
            print(f"\n### {tag}: SKIPPED — {ctx} is past this rope config's ceiling {ceiling}")
            continue
        hay_len = min(ctx, ceiling - GEN - 1)
        print(f"\n### {tag}: prompt<= {hay_len} tok"
              + (f", yarn factor {factor}" if factor else ", rope untouched"))

        kw = dict(max_model_len=ctx, max_num_seqs=2, enforce_eager=False,
                  gpu_memory_utilization=UTIL)
        if factor:
            kw["rope_scaling"] = {"rope_type": "yarn", "factor": factor,
                                  "original_max_position_embeddings": native}
        if W4:
            kw["w4"] = W4
        if CALIB:
            kw["smooth_kv"] = CALIB
        t0 = time.perf_counter()
        eng = LLMEngine(MODEL, **kw)
        cfg = eng.model_runner.config
        cap = cfg.num_kvcache_blocks * eng.model_runner.block_size
        kb = cfg.hf_config.num_hidden_layers * cfg.hf_config.num_key_value_heads * 2 * cfg.hf_config.head_dim
        print(f"  [engine] {time.perf_counter() - t0:.1f}s  pool {cap} slots "
              f"({cap * kb / 2**30:.2f} GiB, {cap / (ctx + GEN):.1f}x this tier)  "
              f"startup peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats()
        if cap < hay_len + GEN:
            print(f"  {tag}: SKIPPED — pool holds {cap} slots, needs {hay_len + GEN}. "
                  f"Raise UTIL or lower the tier.")
            release(eng)
            continue

        hits = {d: [0, 0] for d in DEPTHS}
        tok_s = []
        for i, (key, value, depth) in enumerate(cases):
            ids = build_prompt(eng.tokenizer, key, value, depth, hay_len, phase=7 * i)
            comp, pre, dec = run_one(eng, ids)
            ans = eng.tokenizer.decode(comp)
            hit = int(value in ans)
            got = re.search(r"\d{6}", ans)
            hits[depth][0] += hit
            hits[depth][1] += 1
            rate = len(comp) / dec if dec > 0 else 0.0
            tok_s.append((pre, len(ids), rate))
            print(f"  [{tag} d={depth:.2f} len={len(ids)}] hit={hit} "
                  f"got={got.group() if got else '-'} exp={value} "
                  f"prefill={pre:.1f}s decode={rate:.1f}tok/s ans={ans.strip()[:60]!r}",
                  flush=True)
        by_depth = " ".join(f"d{d:.2f}={h}/{n}" for d, (h, n) in hits.items())
        tot_h = sum(h for h, _ in hits.values())
        tot_n = sum(n for _, n in hits.values())
        print(f"  [{tag}] recall {tot_h}/{tot_n} = {tot_h / tot_n:.0%}   {by_depth}")
        if tok_s:
            print(f"  [{tag}] prefill {sum(p for p, _, _ in tok_s) / len(tok_s):.1f}s/tier-run, "
                  f"decode {sum(r for _, _, r in tok_s) / len(tok_s):.1f} tok/s "
                  f"(at {hay_len} ctx)")
        print(f"  [peak] {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        release(eng)


if __name__ == "__main__":
    main()
