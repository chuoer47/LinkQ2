"""The runtime reproduces the HF reference at the prefill boundary."""
import pytest
import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
PROMPTS = [" The capital of France is", " Water freezes at",
           " The color of the sky is", " The author of Hamlet is"]


def _release(eng):
    import gc
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def _hf_logits():
    from qslab.reference.loader import load_reference_model
    from adapters.tokenizer import QwenTokenizerAdapter
    tok = QwenTokenizerAdapter(MODEL)
    ref = load_reference_model(MODEL)
    with torch.inference_mode():
        rows = [ref(torch.tensor([tok.encode(p)], device="cuda")).logits[0, -1].float()
                for p in PROMPTS]
    del ref
    torch.cuda.empty_cache()
    return torch.stack(rows).cpu()


def test_prefill_logits_match_hf():
    """The int4-calibrated runtime must reproduce HF's next-token logits after a full
    prefill."""
    expected = _hf_logits()
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.45,
                    smooth_kv=CALIB)
    try:
        mr = eng.model_runner
        cap = {}
        orig = mr.run_model

        def spy(input_ids, positions, is_prefill):
            logits = orig(input_ids, positions, is_prefill)
            cap["logits"] = logits.detach().float().clone()
            return logits

        mr.run_model = spy
        for p in PROMPTS:
            eng.add_request(p, SamplingParams(temperature=1e-6, max_tokens=2))
        seqs, is_prefill = eng.scheduler.schedule()
        mr.call("run", seqs, is_prefill)
        got = cap["logits"].cpu()

        assert got.shape == expected.shape
        top_got = got.argmax(-1)
        top_exp = expected.argmax(-1)
        gap = (got - expected).abs()
        assert torch.equal(top_got, top_exp), \
            f"top-1 differs: {top_got.tolist()} vs {top_exp.tolist()}"
        assert gap.max().item() < 0.3, f"max|Δ| = {gap.max().item():.4f}"
        assert gap.mean().item() < 0.05, f"mean|Δ| = {gap.mean().item():.5f}"
    finally:
        _release(eng)


def test_kv4_pool_is_int4():
    """No silent fallback to fp16: the pool must be packed, K's scale must be the static
    per-channel table."""
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.45,
                    smooth_kv=CALIB)
    try:
        a = eng.model_runner.model.model.layers[0].self_attn.attn
        kq, ks = a.k_cache
        vq, vs = a.v_cache
        H, D = a.num_kv_heads, a.head_dim
        assert kq.dtype == torch.uint32 and kq.shape[-1] == D // 8
        assert vq.dtype == torch.uint32 and vq.shape[-1] == D // 8
        assert ks.shape == (H, D), "K scale should be the static per-channel table"
        assert vs.shape[-1] == D // a.v_group
    finally:
        _release(eng)


def test_generation_is_coherent():
    """Catches real breakage (garbage, repetition collapse, zero tokens) without depending
    on a 4-bit cache reproducing a chaotic greedy path."""
    eng = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                    enforce_eager=True, gpu_memory_utilization=0.45,
                    smooth_kv=CALIB)
    try:
        out = eng.generate([PROMPTS[0]], SamplingParams(temperature=1e-6,
                                                        max_tokens=24),
                           use_tqdm=False)
        got = out[0]["token_ids"]
        assert got[0] == 12095, "first token (pure prefill) must be ' Paris'"
        assert 0 not in got, f"zero tokens indicate a broken kernel: {got}"
        assert len(set(got)) > len(got) // 3, f"degenerate repetition: {got}"
    finally:
        _release(eng)
