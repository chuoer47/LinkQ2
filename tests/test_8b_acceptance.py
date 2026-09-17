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

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-8B"
W4 = "models/Qwen3-8B-qslab-w4-awq"
CALIB = "results/smooth_kv4_qwen3-8b.pt"
PROMPT = " The capital of France is"
MAX_TOKENS = 24


def _release(eng):
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
