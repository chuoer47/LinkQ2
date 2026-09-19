"""Batch-size throughput sweep on the current runtime.

Why this file exists
-------------------
The repo's only decode-throughput bench, benchmarks/01-decode-baseline/
bench_throughput.py, is "batch=1 decode tokens/s" by its own docstring and
drives the *frozen* M0-M7 QslabEngine. Every live runtime bench configures
concurrency but never uses it (benchmarks/05-spec-ngram/bench_spec_ngram.py:60
sets max_num_seqs=8 and then calls add_request exactly once). So no reading in
results/ prices bs>1 on the current main line.

Admissibility gate
------------------
--selfcheck re-runs the shipped bs=1 readings before any bs>1 number is
trusted, the same rule this project applies to every sweep harness:
  1.7B fp16 weights + int4 KV, copy prompt, 256 tokens -> off 145.5 / ngram 455.9
      (results/m9_spec_1.7b.txt)
  8B W4A16KV4, copy prompt, 192 tokens -> off 97.2 tok/s
      (results/m9_spec_8b.txt)
A cell is only reported if the matching selfcheck line lands within TOL.

Metrics
-------
decode-window tok/s only (steps where step() returns negative; prefill is
excluded) — the identical formula to bench_spec_ngram.py:63-78, so the two are
comparable cell by cell. tok/s(wall) is the same run's generated-tokens / whole
wall time (prefill included); it exists because vLLM's offline API can only
report that protocol, and it is the column benchmarks/10-vllm-compare reads. Prompts are made distinct per request by prepending a
random token id: BlockManager.compute_hash chains the parent hash
(block_manager.py:43-50), so a differing block 0 disables prefix reuse for the
whole sequence. Without that, N identical prompts would share most of their
KV blocks and the sweep would measure caching, not batching.

Env: MODEL W4 UTIL TOKENS BATCHES GAMMA SPEC TOL SELF TAG
"""
import os
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from qslab.runtime.llm_engine import LLMEngine  # noqa: E402
from qslab.runtime.sampling_params import SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", "models/Qwen3-1.7B")
W4 = os.environ.get("W4")
UTIL = float(os.environ.get("UTIL", "0.5"))
N_TOKENS = int(os.environ.get("TOKENS", "256"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,2,4,8,16").split(",")]
GAMMA = int(os.environ.get("GAMMA", "4"))
SPEC = os.environ.get("SPEC", "")            # "" | ngram
TOL = float(os.environ.get("TOL", "0.05"))
TAG = os.environ.get("TAG", Path(MODEL).name)

COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of Spain is Madrid. "
               "The capital of France is")

# shipped readings this harness must reproduce (see docstring)
SHIPPED = {
    "Qwen3-1.7B": {"off": 145.5, "ngram": 455.9},
    "Qwen3-8B": {"off": 97.2, "ngram": 310.4},
}


def _engine_kwargs(spec_on):
    kw = {}
    calib = "results/smooth_kv4_" + Path(MODEL).name.lower() + ".pt"
    if os.path.exists(calib):
        kw["smooth_kv"] = calib
    if W4:
        kw["w4"] = W4
    if spec_on:
        kw.update(spec_method="ngram", spec_num_drafts=GAMMA)
    return kw


def make_engine(max_bs, spec_on=False):
    return LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=max_bs,
                     enforce_eager=False, gpu_memory_utilization=UTIL,
                     **_engine_kwargs(spec_on))


def measure(eng, prompts, n_tokens):
    """Run one cell to completion; return the decode-window reading."""
    for p in prompts:
        eng.add_request(p, SamplingParams(temperature=1e-6, max_tokens=n_tokens,
                                          ignore_eos=True))
    dec_time = committed = steps = 0
    per_step = []
    t0 = perf_counter()
    prefill_time = 0.0
    while not eng.is_finished():
        t = perf_counter()
        _, num = eng.step()
        dt = perf_counter() - t
        if num < 0:
            dec_time += dt
            committed += -num
            steps += 1
            per_step.append(-num)
        else:
            prefill_time += dt
    agg = committed / dec_time
    wall = perf_counter() - t0
    return dict(n=len(prompts), agg=agg, per_req=agg / len(prompts),
                wall_agg=committed / max(wall, 1e-9),
                tok_per_step=committed / max(steps, 1),
                min_step=min(per_step) if per_step else 0,
                max_step=max(per_step) if per_step else 0,
                steps=steps, dec_time=dec_time, prefill=prefill_time,
                wall=wall)


def distinct_prompts(tok, body, n, rng):
    """n prompts of identical length, pairwise distinct from token 0 on."""
    used = set()
    out = []
    for _ in range(n):
        while True:
            head = rng.randint(1000, 200000) % tok.vocab_size
            if head not in used:
                used.add(head)
                break
        out.append([head] + body)
    return out


def kv_schema(eng):
    """Compare what allocate_kv_cache BILLS per slot (model_runner.py:123-124,
    slot_bytes = 2*head_dim) with what it actually allocates (:139-147:
    kq/vq int4 at head_dim/2 bytes each + vs at 2*head_dim/v_group bytes).
    """
    mr = eng.model_runner
    cfg = mr.config
    hc = cfg.hf_config
    head_dim = getattr(hc, "head_dim", hc.hidden_size // hc.num_attention_heads)
    slots = cfg.num_kvcache_blocks * cfg.kvcache_block_size
    per_token = static_bytes = 0        # both summed over the whole pool
    for (kq, ks), (vq, vs) in mr.kv_cache:
        per_token += (kq.numel() * kq.element_size()
                      + vq.numel() * vq.element_size()
                      + vs.numel() * vs.element_size()) // slots
        static_bytes += ks.numel() * ks.element_size()
    billed = 2 * head_dim * hc.num_key_value_heads * hc.num_hidden_layers
    return dict(layers=hc.num_hidden_layers, kv_heads=hc.num_key_value_heads,
                head_dim=head_dim, blocks=cfg.num_kvcache_blocks,
                capacity_tokens=slots,
                billed_kib=billed / 1024, actual_kib=per_token / 1024,
                ratio=billed / per_token, static_mib=static_bytes / 2**20)


def selfcheck(eng, tok):
    """bs=1, shipped prompt, shipped token count, identical formula.

    A miss means this harness is NOT measuring what the shipped readings
    measured, so the bs>1 cells below would be meaningless: exit non-zero.
    """
    key = "ngram" if SPEC else "off"
    want = SHIPPED.get(Path(MODEL).name, {}).get(key)
    r = measure(eng, [tok.encode(COPY_PROMPT)], N_TOKENS)
    name = "off" if not SPEC else f"ngram g={GAMMA}"
    print(f"--- selfcheck: {TAG} {name} tokens={N_TOKENS} tol={TOL:.0%} ---")
    if want is None:
        print(f"{name:12} {r['agg']:8.1f} tok/s   (no shipped reference)")
        return True, r
    d = abs(r["agg"] - want) / want
    ok = d <= TOL
    print(f"{name:12} {r['agg']:8.1f} tok/s   shipped {want:8.1f}  "
          f"Δ{d:6.1%}  [{'OK' if ok else 'MISMATCH'}]")
    return ok, r


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    body = tok.encode(COPY_PROMPT)
    print(f"=== batch-throughput: {TAG} w4={bool(W4)} util={UTIL} tokens={N_TOKENS} "
          f"prompt_len={len(body)} spec={SPEC or 'off'} ===")
    eng = make_engine(max(BATCHES), spec_on=bool(SPEC))
    try:
        s = kv_schema(eng)
        print(f"[kvschema] layers={s['layers']} kv_heads={s['kv_heads']} head_dim={s['head_dim']} "
              f"blocks={s['blocks']} capacity={s['capacity_tokens']} tok "
              f"billed={s['billed_kib']:.1f}KiB/tok actual={s['actual_kib']:.1f}KiB/tok "
              f"ratio={s['ratio']:.2f}x static_scale={s['static_mib']:.2f}MiB")
        if os.environ.get("SELF", "1") == "1":
            ok, _ = selfcheck(eng, tok)
            if not ok:
                print("GATE FAILED: harness does not reproduce the shipped bs=1 reading; "
                      "bs>1 cells are not reported.")
                sys.exit(1)
        print(f"{'bs':>4} {'tok/s(dec)':>11} {'tok/s(wall)':>12} {'tok/s/req':>10} "
              f"{'tok/step':>9} {'min/max':>9} {'steps':>6} {'prefill_s':>10} {'wall_s':>7}")
        rng = random.Random(11)
        rows = []
        for bs in BATCHES:
            prompts = distinct_prompts(tok, body, bs, rng)
            r = measure(eng, prompts, N_TOKENS)
            rows.append((bs, r))
            print(f"{bs:>4} {r['agg']:11.1f} {r['wall_agg']:12.1f} {r['per_req']:10.1f} "
                  f"{r['tok_per_step']:9.2f} {str(r['min_step'])+'/'+str(r['max_step']):>9} "
                  f"{r['steps']:6d} {r['prefill']:10.2f} {r['wall']:7.2f}")
            torch.cuda.empty_cache()
        base = rows[0][1]["agg"]
        print("speedup vs bs=1: " +
              " ".join(f"bs{b}={r['agg'] / base:.2f}x" for b, r in rows[1:]))
    finally:
        eng.exit()


if __name__ == "__main__":
    main()
