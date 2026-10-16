"""Admission, termination and page ownership without model weights."""
from types import SimpleNamespace

import pytest

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.sequence import Sequence


def engine():
    Sequence.block_size = 4
    cfg = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16,
        max_model_len=12, eos=99, kvcache_block_size=4,
        num_kvcache_blocks=4, spec_method=None, spec_num_drafts=0,
        spec_adaptive_gamma=False)
    eng = LLMEngine.__new__(LLMEngine)
    eng.scheduler = Scheduler(cfg)
    return eng


def assert_released(eng):
    bm = eng.scheduler.block_manager
    assert not bm.used_block_ids
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert all(block.ref_count == 0 for block in bm.blocks)


def test_context_budget_is_clamped_without_mutating_params():
    eng = engine()
    params = SamplingParams(max_tokens=20)
    seq = eng.add_request([1] * 10, params)
    assert seq.max_tokens == 2 and params.max_tokens == 20
    for token in [2, 3]:
        seqs, prefill = eng.scheduler.schedule()
        eng.scheduler.postprocess(seqs, [token], prefill)
    assert len(seq) == 12 and seq.finish_reason == 'length'
    assert eng.is_finished()
    assert_released(eng)


@pytest.mark.parametrize('prompt', [[], [1] * 12])
def test_invalid_prompt_is_not_queued(prompt):
    eng = engine()
    with pytest.raises(ValueError, match='prompt'):
        eng.add_request(prompt, SamplingParams())
    assert eng.is_finished()


def test_request_that_cannot_fit_pool_is_rejected():
    eng = engine()
    eng.scheduler.max_model_len = 32
    with pytest.raises(ValueError, match='KV pool'):
        eng.add_request([1] * 16, SamplingParams(max_tokens=4))
    assert eng.is_finished()


def test_cancel_waiting_running_and_shared_prefix():
    eng = engine()
    first = eng.add_request([1, 2, 3, 4, 5], SamplingParams(max_tokens=4))
    seqs, prefill = eng.scheduler.schedule()
    eng.scheduler.postprocess(seqs, [6], prefill)
    second = eng.add_request([1, 2, 3, 4, 7], SamplingParams(max_tokens=4))
    seqs, prefill = eng.scheduler.schedule()
    eng.scheduler.postprocess(seqs, [8], prefill)
    shared = first.block_table[0]
    assert second.block_table[0] == shared
    assert eng.scheduler.block_manager.blocks[shared].ref_count == 2
    eng.abort_request(first.seq_id)
    assert eng.scheduler.block_manager.blocks[shared].ref_count == 1
    waiting = eng.add_request([9], SamplingParams(max_tokens=2))
    eng.abort_request(waiting.seq_id)
    eng.abort_request(second.seq_id)
    eng.abort_request(second.seq_id)
    assert eng.is_finished()
    assert_released(eng)


def test_eos_and_execution_exception_release_pages():
    eng = engine()
    seq = eng.add_request([1, 2], SamplingParams(max_tokens=4))
    seqs, prefill = eng.scheduler.schedule()
    eng.scheduler.postprocess(seqs, [99], prefill)
    assert seq.finish_reason == 'stop'
    assert_released(eng)
    seq = eng.add_request([3, 4], SamplingParams(max_tokens=4))

    def fail(*_):
        raise RuntimeError('forward failed')
    eng.model_runner = SimpleNamespace(call=fail)
    with pytest.raises(RuntimeError, match='forward failed'):
        eng.step()
    assert seq.finish_reason == 'error' and eng.is_finished()
    assert_released(eng)
    del eng.model_runner

