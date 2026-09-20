"""M8 acceptance on the main-line 8B model: W4 + KV4 + paged runtime.

Throughput only. Correctness is covered elsewhere and deliberately not by
greedy token equality, which is a chaotic criterion at 4-bit KV (see
tests/test_e2e_engine.py); the 8B generation accuracy is measured as PPL in
benchmarks/bench_ppl_kv.py (+0.369, 16.06 -> 16.43).
"""
import gc
import time

import pytest
import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-8B"
W4 = "models/Qwen3-8B-qslab-w4-awq"
CALIB = "results/smooth_kv4_qwen3-8b.pt"
PROMPT = " The capital of France is"
MAX_TOKENS = 24


def _release(eng):
    prop = getattr(eng.scheduler, "proposer", None)
    if prop is not None and hasattr(prop, "exit"):
        prop.exit()
    eng.scheduler.proposer = None
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def test_8b_decode_throughput():
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=False, gpu_memory_utilization=0.82,
                    smooth_kv=CALIB, w4=W4)
    try:
        eng.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=8),
                     use_tqdm=False)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            eng.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=128),
                         use_tqdm=False)
            ts.append(128 / (time.perf_counter() - t0))
        tps = sorted(ts)[1]
        print(f"\n  8B W4+KV4+graph decode: {tps:.1f} tok/s (median of 3)")
        assert tps > 5
    finally:
        _release(eng)


def test_8b_speculative_verify():
    """M9 on the main-line 8B stack (W4 + KV4 + graph + n-gram spec): the
    verify path must produce the copy-continuation and accept > 1 token/step
    (Marlin handles the M=5 rows; v1 GEMV would re-read weights per row)."""
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=False, gpu_memory_utilization=0.82,
                    smooth_kv=CALIB, w4=W4,
                    spec_method="ngram", spec_num_drafts=4)
    try:
        prompt = (" The capital of France is Paris. The capital of Germany is Berlin. "
                  "The capital of Italy is Rome. The capital of France is")
        committed = 0
        steps = 0
        eng.add_request(prompt, SamplingParams(temperature=1e-6, max_tokens=32,
                                                ignore_eos=True))
        while not eng.is_finished():
            _, num = eng.step()
            if num < 0:
                committed += -num
                steps += 1
        assert steps < 31, f"steps={steps}: no multi-token acceptance"
        # first token comes from the prefill step (plain postprocess path)
        assert committed == 32 - 1
        avg = committed / steps
        assert avg > 2.0, f"avg committed/step = {avg:.2f}"
    finally:
        _release(eng)


def test_8b_draft_spec_acceptance():
    """The main-line combination no test covered (TODO M10): 8B W4+KV4 driven by
    a drafting model — two runtimes and two int4 pools on one 24 GB card, which
    is why the memory fractions differ from the tests above. Acceptance is
    asserted structurally, exactly as the n-gram case."""
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=4,
                    enforce_eager=False, gpu_memory_utilization=0.62,
                    smooth_kv=CALIB, w4=W4,
                    spec_method="draft", spec_num_drafts=4,
                    draft_model="models/Qwen3-0.6B",
                    draft_gpu_memory_utilization=0.9)
    try:
        prompt = (" The capital of France is Paris. The capital of Germany is Berlin. "
                  "The capital of Italy is Rome. The capital of France is")
        committed = steps = 0
        eng.add_request(prompt, SamplingParams(temperature=1e-6, max_tokens=32,
                                               ignore_eos=True))
        while not eng.is_finished():
            _, num = eng.step()
            if num < 0:
                committed += -num
                steps += 1
        assert steps < 31, f"steps={steps}: no multi-token acceptance"
        assert committed == 32 - 1        # the prefill step owns the first one
        avg = committed / steps
        assert avg > 2.0, f"avg committed/step = {avg:.2f}"
        st = eng.scheduler.spec_stats
        assert st["steps"] == steps and st["committed"] == committed
    finally:
        _release(eng)
