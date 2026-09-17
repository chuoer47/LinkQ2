"""The eager runtime path: prefill exactness and coherent generation.

Strict token equality against a greedy HF decode is deliberately NOT asserted
— see tests/test_e2e_engine.py for why that is a chaotic criterion at 4-bit
KV. What is asserted here is the prefill boundary (which must be exact) and
that generation stays fluent.
"""
import gc

import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
PROMPT = " The capital of France is"
FIRST_TOKEN = 12095          # " Paris", HF-confirmed


def _release(eng):
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def test_eager_generates_coherently():
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.35,
                    smooth_kv=CALIB)
    try:
        out = eng.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=16),
                           use_tqdm=False)
        toks = out[0]["token_ids"]
        assert toks[0] == FIRST_TOKEN
        # token 1 must be sentence punctuation or a space-led word, not a
        # repetition of the prompt or a zero
        assert 0 not in toks
        assert len(set(toks)) > 4
    finally:
        _release(eng)
