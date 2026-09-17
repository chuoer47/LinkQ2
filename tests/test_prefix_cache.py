"""Prefix caching must not silently corrupt results.

nano-vllm's block manager reuses blocks whose token hash matches an earlier
sequence. When that happens the reused prefix exists ONLY in the int4 pool:
`prepare_prefill` sets context.block_tables and starts the new chunk at
num_cached_tokens, so the prefix is never recomputed.

Our prefill runs flash-attn on the freshly computed fp16 q/k/v — it never
reads the pool — so a cache hit means attention sees only the new suffix and
the sequence silently loses its prefix.

These tests are written so they FAIL while that is true. Fixing it means
either reading the cached prefix out of the pool during prefill, or refusing
prefix reuse until that path exists.
"""
import pytest

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
# long enough to fill a block (block_size 128) — a shorter prompt never
# reaches the cache lookup, which only inspects full blocks
LONG = " The history of computing began long before electronic machines. " * 12


def _release(eng):
    import gc
    import torch
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def _engine():
    return LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                     enforce_eager=True, gpu_memory_utilization=0.4,
                     smooth_kv=CALIB)


def test_identical_prompt_is_deterministic():
    """Submitting the same prompt twice must give the same answer."""
    eng = _engine()
    try:
        sp = SamplingParams(temperature=1e-6, max_tokens=8)
        first = eng.generate([LONG], sp, use_tqdm=False)[0]["token_ids"]
        second = eng.generate([LONG], sp, use_tqdm=False)[0]["token_ids"]
        assert first == second, (
            "repeat of the same prompt changed: "
            f"{first[:8]} vs {second[:8]}\n"
            "A prefix-cache hit drops the cached prefix from prefill.")
    finally:
        _release(eng)


def test_repeat_request_matches_a_fresh_engine():
    """A repeat request must land on the same answer as a clean run.

    Only one engine fits on the card at a time, so the baseline is taken
    first and its engine released before the repeat is run.
    """
    sp = SamplingParams(temperature=1e-6, max_tokens=8)
    prompt = LONG + " Different suffix one."

    fresh = _engine()
    try:
        baseline = fresh.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
    finally:
        _release(fresh)

    eng = _engine()
    try:
        eng.generate([prompt], sp, use_tqdm=False)          # populates blocks
        repeat = eng.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
        assert repeat == baseline, (
            f"repeat diverged from a fresh run: {repeat[:8]} vs {baseline[:8]}")
    finally:
        _release(eng)
