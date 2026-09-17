"""M8 acceptance on the main-line 8B model: W4 + KV4 + paged runtime.

Verifies the full stack the project is built around — packed W4 weights, the
SmoothAttention-calibrated int4 KV cache, CUDA Graph — against a greedy HF
reference on the same prompt.
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


ORACLE_JSON = "results/m8_8b_oracle.json"


@pytest.fixture(scope="module")
def oracle():
    """Greedy HF reference, computed in a subprocess.

    The fp16 8B model needs ~16 GB; running it in-process leaves the card
    fragmented and the engine below cannot allocate its own copy.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    path = Path(ORACLE_JSON)
    if not path.exists():
        subprocess.run([sys.executable, "tests/_8b_oracle.py",
                        str(MAX_TOKENS), str(path)],
                       cwd=str(Path(__file__).resolve().parents[1]), check=True)
    return json.loads(path.read_text())["tokens"]


def test_8b_w4_kv4_graph_is_coherent(oracle):
    """The full stack runs and stays coherent on 8B.

    Not a token-exact check: the 8B static K scale does not hold as well as
    the 1.7B one — the runtime K exceeds its calibrated per-channel limit on
    18-66 channels (1.7B: 1), giving ~0.15 relative KV error, and greedy
    decoding flips at a near-tie around token 5. The 1.7B tests assert
    exactness; here we assert the output is fluent and on-topic, which
    catches real breakage (garbage, repetition collapse, zeros) without
    depending on the calibration's margin.

    See notes/M8-整合.md for the per-layer measurements.
    """
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=False, gpu_memory_utilization=0.82,
                    smooth_kv=CALIB, w4=W4)
    try:
        assert eng.model_runner.graphs, "graph capture failed"
        out = eng.generate([PROMPT], SamplingParams(temperature=1e-6,
                                                    max_tokens=MAX_TOKENS),
                           use_tqdm=False)
        got = out[0]["token_ids"]
        print(f"\n  hf      : {oracle}")
        print(f"  runtime : {got}")
        print(f"  text    : {out[0]['text'][:70]!r}")
        assert got[0] == oracle[0], "first token (pure prefill) must match"
        assert len(set(got)) > len(got) // 3, f"degenerate repetition: {got}"
        assert 0 not in got, f"zero tokens indicate a broken kernel: {got}"
        # prefill is the exact boundary; decode drifts with 4-bit rounding
        assert "Paris" in out[0]["text"]
    finally:
        _release(eng)


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
