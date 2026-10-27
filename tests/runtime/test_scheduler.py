"""nano-vLLM PR #276 scheduling invariants with the project's page manager."""
from types import SimpleNamespace

from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.sequence import Sequence


def scheduler(reserve=0):
    Sequence.block_size = 4
    return Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=4,
        max_model_len=32, eos=99, kvcache_block_size=4, num_kvcache_blocks=8,
        spec_method=None, spec_num_drafts=0, spec_adaptive_gamma=False,
        enable_chunked_prefill=True, prefill_reserve_blocks=reserve))


def test_decode_shares_budget_with_partial_prefill():
    sched = scheduler()
    decode = Sequence([1, 2], SamplingParams(max_tokens=8, ignore_eos=True))
    sched.add(decode)
    seqs, has_prefill = sched.schedule()
    sched.postprocess(seqs, [3], has_prefill)
    long = Sequence(list(range(10)), SamplingParams(max_tokens=2, ignore_eos=True))
    sched.add(long)
    seqs, has_prefill = sched.schedule()
    assert has_prefill and seqs == [decode, long]
    assert [s.num_scheduled_tokens for s in seqs] == [1, 3]
    assert decode.should_sample and not long.should_sample
    sched.postprocess(seqs, [4, 0], has_prefill)
    assert decode.num_completion_tokens == 2 and long.num_completion_tokens == 0
    while not sched.is_finished():
        seqs, has_prefill = sched.schedule()
        assert sum(s.num_scheduled_tokens for s in seqs) <= 4
        sched.postprocess(seqs, [5] * len(seqs), has_prefill)
    assert decode.num_completion_tokens == 8 and long.num_completion_tokens == 2
    assert not sched.block_manager.used_block_ids


def test_shared_prefix_counts_only_new_blocks_for_watermark():
    sched = scheduler(reserve=5)
    # First request uses two blocks and leaves six; one decode row is retained.
    sched.prefill_reserve_blocks = 0
    first = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=4, ignore_eos=True))
    sched.add(first)
    while first.num_completion_tokens == 0:
        seqs, flag = sched.schedule()
        sched.postprocess(seqs, [6] * len(seqs), flag)
    sched.prefill_reserve_blocks = 5
    second = Sequence([1, 2, 3, 4, 7], SamplingParams(max_tokens=2))
    sched.add(second)
    seqs, flag = sched.schedule()
    assert flag and seqs == [first, second]
    assert first.block_table[0] == second.block_table[0]
    assert sched.block_manager.blocks[first.block_table[0]].ref_count == 2


def test_verify_queries_share_the_token_budget_with_prefill():
    sched = scheduler()
    sched.gamma = sched.proposal_gamma = 4
    decode = Sequence([1], SamplingParams(max_tokens=12, ignore_eos=True))
    sched.add(decode)
    batch, flag = sched.schedule()
    sched.postprocess(batch, [2], flag)
    decode.spec_drafts = [3, 4]
    long = Sequence(list(range(10)), SamplingParams(max_tokens=2))
    sched.add(long)
    batch, flag = sched.schedule()
    assert flag and batch == [decode, long]
    assert [s.num_scheduled_tokens for s in batch] == [3, 1]
    assert decode.spec_verify and not long.spec_verify
    sched.commit_verified(decode, [9], accepted=0)
    assert decode.num_cached_tokens == 2 and len(decode) == 3
    sched.postprocess([long], [0], True)
    batch, _ = sched.schedule()
    assert not decode.spec_verify and decode.num_scheduled_tokens == 1
    assert sum(s.num_scheduled_tokens for s in batch) == 4


def test_rejection_releases_tail_and_eos_releases_every_page():
    sched = scheduler()
    sched.gamma = sched.proposal_gamma = 4
    sched.max_num_batched_tokens = 8
    seq = Sequence([1], SamplingParams(max_tokens=12))
    sched.add(seq)
    batch, flag = sched.schedule()
    sched.postprocess(batch, [2], flag)
    seq.spec_drafts = [3, 4, 5, 6]
    sched.schedule()
    assert len(seq.block_table) == 2
    sched.commit_verified(seq, [7], accepted=0)
    assert len(seq.block_table) == 1 and seq.num_cached_tokens == 2
    seq.spec_drafts = [8, 99, 10]
    sched.schedule()
    sched.commit_verified(seq, [8, 99, 10, 11], accepted=3)
    assert seq.completion_token_ids == [2, 7, 8, 99]
    assert seq.finish_reason == 'stop' and not sched.block_manager.used_block_ids
    assert sched.spec_stats['accepted'] == 2


def test_verify_window_shrinks_to_budget_and_output_limit():
    sched = scheduler()
    sched.gamma = sched.proposal_gamma = 4
    seq = Sequence([1], SamplingParams(max_tokens=3, ignore_eos=True))
    sched.add(seq)
    batch, flag = sched.schedule()
    sched.postprocess(batch, [2], flag)
    seq.spec_drafts = [3, 4, 5, 6]
    batch, flag = sched.schedule()
    assert seq.spec_drafts == [3] and seq.num_scheduled_tokens == 2
    sched.commit_verified(seq, [3, 4], accepted=1)
    assert seq.num_completion_tokens == 3 and seq.finish_reason == 'length'
    assert not sched.block_manager.used_block_ids


def test_memory_pressure_preempts_then_resumes_from_committed_history():
    sched = scheduler()
    sched.max_num_batched_tokens = 8
    from qslab.runtime.state.block_manager import BlockManager
    from qslab.runtime.state.sequence import SequenceStatus
    sched.block_manager = BlockManager(2, 4)
    first = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=2, ignore_eos=True))
    second = Sequence([5, 6, 7, 8], SamplingParams(max_tokens=2, ignore_eos=True))
    sched.add(first)
    sched.add(second)
    batch, prefill = sched.schedule()
    sched.postprocess(batch, [9, 10], prefill)
    history = list(second.token_ids)
    batch, prefill = sched.schedule()
    assert batch == [first] and not prefill
    assert second.status == SequenceStatus.WAITING and not second.block_table
    assert second.num_cached_tokens == 0 and second.token_ids == history
    sched.postprocess(batch, [11], prefill)
    batch, prefill = sched.schedule()
    assert prefill and batch == [second]
    sched.postprocess(batch, [12], prefill)
    assert first.is_finished and second.is_finished and sched.is_finished()
    assert not sched.block_manager.used_block_ids
