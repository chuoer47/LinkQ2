"""L4 facade: qslab.api.LLM over the mainline runtime."""
import pytest
import torch

from qslab.api import LLM, SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
W4 = "models/Qwen3-1.7B-qslab-w4-awq2"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of France is")


def _llm(**kw):
    kw.setdefault("max_model_len", 4096)
    kw.setdefault("max_num_seqs", 4)
    kw.setdefault("gpu_memory_utilization", 0.5)
    kw.setdefault("enforce_eager", True)
    return LLM(MODEL, smooth_kv=CALIB, **kw)


def _release(llm):
    """Hand the card back; two 1.7B engines do not co-reside (see conftest)."""
    import gc
    eng = llm._engine
    for layer in eng.model_runner.model.model.layers:
        a = layer.self_attn.attn
        a.k_cache = a.v_cache = None
    eng.model_runner.model = None
    eng.model_runner.kv_cache = None
    del eng.model_runner
    gc.collect()
    torch.cuda.empty_cache()


def test_generate_with_w4_and_int4_kv():
    llm = _llm(w4=W4)
    try:
        res = llm.generate(COPY_PROMPT, SamplingParams(max_tokens=16))
        assert len(res["token_ids"]) == 16
        assert res["text"].strip()            # decodes to something printable
        assert llm.kv_memory_bytes() > 0
        model = llm._engine.model_runner.model
        w4 = [m for m in model.modules() if type(m).__name__ == "W4Linear"]
        assert len(w4) > 100, f"only {len(w4)} linears were swapped"
        m = w4[0]
        # one packed copy = int4 nibbles plus per-group fp16 scales
        packed = (m.in_features * m.out_features // 2
                  + m.in_features // m.group_size * m.out_features * 2)
        # Marlin's repack is the same size as the v1 pack and replaces it, so the budget is
        #   ONE packed copy — holding both would double it
        assert m.weight_memory_bytes() <= packed * 1.02
        assert m.weight_memory_bytes() < m.in_features * m.out_features * 2
        # the embeddings and lm_head stay fp16, so the whole model is not 1/4
        assert llm.weight_memory_bytes() < 2_400_000_000
    finally:
        _release(llm)


def test_greedy_is_the_default_and_zero_is_not_a_shortcut():
    p = SamplingParams()
    assert p.temperature == 1e-6
    assert SamplingParams(temperature=0.0).temperature == 1e-6
    assert SamplingParams(temperature=0.7).temperature == 0.7


def test_speculative_stats_are_reported_and_reset():
    llm = _llm(w4=W4, spec="ngram", spec_gamma=4)
    try:
        res = llm.generate(COPY_PROMPT, SamplingParams(max_tokens=48))
        st = res["stats"]
        assert st["steps"] > 0 and st["proposals"] > 0
        assert 0.0 <= st["acceptance_rate"] <= 1.0
        # a copy prompt must beat plain decoding: > 1 token per verify step
        assert st["mean_len"] > 1.2, f"no multi-token acceptance: {st}"
        # one prompt -> the prefill step owns exactly one token; the rest of
        # the output was committed by verify steps (test_spec_runtime's rule)
        assert st["committed"] + 1 == len(res["token_ids"])
        again = llm.generate(COPY_PROMPT, SamplingParams(max_tokens=48))
        assert again["stats"]["steps"] == st["steps"], "counters leaked across calls"
    finally:
        _release(llm)


def test_no_speculation_reports_no_stats():
    llm = _llm(w4=W4)
    try:
        assert llm.generate(" Water freezes at", SamplingParams(max_tokens=8))["stats"] == {}
    finally:
        _release(llm)


def test_batch_and_argument_guards():
    llm = _llm(w4=W4)
    try:
        out = llm.generate_batch([COPY_PROMPT, " Water freezes at"],
                                 SamplingParams(max_tokens=8))
        assert len(out) == 2 and all(len(o["token_ids"]) == 8 for o in out)
    finally:
        _release(llm)
    with pytest.raises(AssertionError):
        _llm(spec="draft", spec_gamma=2)          # draft mode needs a draft model
    with pytest.raises(ValueError):
        _llm(draft="models/Qwen3-0.6B")           # a draft model without draft mode


def test_lookahead_mode_is_reachable_from_the_facade():
    llm = _llm(w4=W4, spec="lookahead", spec_gamma=4, spec_lookahead_span=8)
    try:
        assert llm._engine.scheduler.proposer.__class__.__name__ == "LookaheadProposer"
        res = llm.generate(COPY_PROMPT, SamplingParams(max_tokens=32))
        assert len(res["token_ids"]) == 32
        assert res["stats"]["mean_len"] > 1.0
    finally:
        _release(llm)
