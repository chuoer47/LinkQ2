"""M9: draft-model (Qwen3-0.6B) speculative decoding through the engine.

The draft is a second full runtime (own paged int4 pool + CUDA graphs,
runtime/draft.py). These tests assert the lockstep protocol's structure:
real acceptance on both repetitive and natural text (the draft model's
advantage over n-gram: it proposes on prose, not just copies), deterministic
replay, canonical block tables, and draft-sequence cleanup on finish.

Not asserted: spec vs non-spec token equality (GEMM M-dim tiling chaos, the
M8 criterion).
"""
import pytest
import torch

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

pytestmark = pytest.mark.e2e

MODEL = "models/Qwen3-1.7B"
DRAFT = "models/Qwen3-0.6B"
CALIB = "results/smooth_kv4_qwen3-1.7b.pt"
DRAFT_CALIB = "results/smooth_kv4_qwen3-0.6b.pt"
COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of France is")
NATURAL_PROMPT = ("The history of computing spans several centuries, from early "
                  "mechanical aids to modern electronic machines. Early devices like "
                  "the abacus assisted calculation, and the")


def _eng():
    return LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=4,
                     enforce_eager=False, gpu_memory_utilization=0.5,
                     smooth_kv=CALIB, spec_method="draft", spec_num_drafts=4,
                     draft_model=DRAFT, draft_gpu_memory_utilization=0.95)


def _release(eng):
    import gc
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


def _run(eng, prompt, n=48, ignore_eos=True):
    committed = steps = 0
    eng.add_request(prompt, SamplingParams(temperature=1e-6, max_tokens=n,
                                           ignore_eos=ignore_eos))
    while not eng.is_finished():
        _, num = eng.step()
        if num < 0:
            committed += -num
            steps += 1
    return committed, steps


def test_draft_accepts_on_copy_and_natural():
    eng = _eng()
    try:
        committed, steps = _run(eng, COPY_PROMPT, n=48)
        avg = committed / steps
        assert avg > 3.0, f"copy acceptance too low: {avg:.2f}"
        committed, steps = _run(eng, NATURAL_PROMPT, n=48)
        avg = committed / steps
        assert avg > 1.5, f"natural acceptance too low: {avg:.2f}"
    finally:
        _release(eng)


def test_draft_output_is_coherent_and_deterministic():
    eng = _eng()
    try:
        out1 = eng.generate([COPY_PROMPT], SamplingParams(temperature=1e-6,
                                                          max_tokens=32),
                            use_tqdm=False)
        out2 = eng.generate([COPY_PROMPT], SamplingParams(temperature=1e-6,
                                                          max_tokens=32),
                            use_tqdm=False)
        ids = out1[0]["token_ids"]
        assert ids[0] == 12095, "first token (pure prefill) must be ' Paris'"
        assert ids == out2[0]["token_ids"]
        assert 0 not in ids
        assert len(set(ids)) > 8
    finally:
        _release(eng)


def test_draft_sequences_are_cleaned_up_and_tables_canonical():
    eng = _eng()
    try:
        _run(eng, COPY_PROMPT, n=64, ignore_eos=False)
        prop = eng.scheduler.proposer
        assert len(prop.drafts) == 0, "finished target left a draft sequence"
        for seq in list(eng.scheduler.running):
            assert len(seq.block_table) == seq.num_blocks
    finally:
        _release(eng)


def test_draft_multiple_sequences():
    eng = _eng()
    try:
        for prompt in [COPY_PROMPT, NATURAL_PROMPT]:
            eng.add_request(prompt, SamplingParams(temperature=1e-6,
                                                   max_tokens=32, ignore_eos=True))
        committed = steps = 0
        while not eng.is_finished():
            _, num = eng.step()
            if num < 0:
                committed += -num
                steps += 1
        # both first tokens come from prefill steps (plain postprocess)
        assert committed == 64 - 2
        assert (committed / steps) > 1.0
    finally:
        _release(eng)
