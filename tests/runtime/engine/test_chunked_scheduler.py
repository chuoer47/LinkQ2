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

