"""The eager runtime path against the HF oracle.

Kept in its own module because only one engine fits on the card at a time —
each e2e module builds its engine, uses it, and releases it.
"""
import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
PROMPT = " The capital of France is"
ORACLE = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315,
          15344, 374, 21718, 13]


def test_eager_matches_oracle():
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.35,
                    smooth_kv=CALIB)
    out = eng.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=16),
                       use_tqdm=False)
    assert out[0]["token_ids"] == ORACLE
    import gc
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()
