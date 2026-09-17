"""M8: end-to-end generation through the runtime, W4-free with KV4.

The oracle is a greedy HF decode of the same prompt. With SmoothAttention
calibration applied the KV4 path reproduces it token for token, so the test
is a strict equality check rather than a fuzzy one — any drift in the cache
layout (encoding, row addressing, group scaling) shows up as a mismatch.
"""
import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams
from adapters.tokenizer import QwenTokenizerAdapter

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
PROMPT = " The capital of France is"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"


@pytest.fixture(scope="module")
def oracle():
    from qslab.models.loader import load_reference_model
    tok = QwenTokenizerAdapter(MODEL)
    ids = tok.encode(PROMPT)
    ref = load_reference_model(MODEL)
    with torch.inference_mode():
        out = ref.generate(torch.tensor([ids], device="cuda"),
                           max_new_tokens=16, do_sample=False)
    del ref
    torch.cuda.empty_cache()
    return ids, out[0][len(ids):].tolist()


@pytest.fixture(scope="module")
def engine():
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.5,
                    smooth_kv=CALIB)
    yield eng


def test_greedy_matches_hf(engine, oracle):
    _, expected = oracle
    out = engine.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=16),
                          use_tqdm=False)
    assert out[0]["token_ids"] == expected


def test_kv4_pool_is_int4(engine):
    """The pool must actually be packed — a silent fallback to fp16 would
    make the correctness test above pass for the wrong reason."""
    a = engine.model_runner.model.model.layers[0].self_attn.attn
    kq, ks = a.k_cache
    vq, vs = a.v_cache
    H, D = a.num_kv_heads, a.head_dim
    assert kq.dtype == torch.uint32 and kq.shape[-1] == D // 8      # 4 bits/channel
    assert vq.dtype == torch.uint32 and vq.shape[-1] == D // 8
    assert ks.shape == (H, D), "K scale should be the static per-channel table"
    assert vs.shape[-1] == D // a.v_group, "V scale should be per token group"


def test_second_request_reuses_freed_blocks(engine, oracle):
    """Blocks released by a finished sequence must be reusable without
    carrying stale KV into the next one."""
    _, expected = oracle
    for _ in range(2):
        out = engine.generate([PROMPT], SamplingParams(temperature=1e-6, max_tokens=16),
                              use_tqdm=False)
        assert out[0]["token_ids"] == expected
