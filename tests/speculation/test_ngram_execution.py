"""Small checkpoints exercise n-gram + graph + mixed chunked prefill together."""
import pytest
import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams



def make_engine(checkpoint, mode, gamma, eager, max_len=64):
    source, quantized, calibration = checkpoint
    return LLMEngine(str(quantized if mode.startswith('w4') else source),
        kv_cache_dtype='kv4' if mode.endswith('kv4') else 'fp16',
        smooth_kv=str(calibration) if mode.endswith('kv4') else None,
        spec_method='ngram' if gamma else None, spec_num_drafts=gamma or 4,
        spec_ngram_size=1, enable_chunked_prefill=True, enforce_eager=eager,
        max_num_seqs=4, max_model_len=max_len, max_num_batched_tokens=8,
        num_kvcache_blocks=4, kvcache_block_size=256)


def drive(engine):
    jobs = []
    trace = []
    original = engine.scheduler.schedule

    def scheduled():
        seqs, flag = original()
        trace.append({'mixed_verify': flag and any(s.spec_verify for s in seqs),
                      'queries': sum(s.num_scheduled_tokens for s in seqs)})
        return seqs, flag

    def add(ids, tokens):
        jobs.append(engine.add_request(ids, SamplingParams(max_tokens=tokens,
                                                           temperature=0, ignore_eos=True)))
    engine.scheduler.schedule = scheduled
    engine.scheduler.block_manager.ENABLE_PREFIX_CACHE = False
    add([3, 8, 9, 4, 6, 2, 7], 18)
    step = 0
    try:
        while not engine.is_finished():
            if step == 2:
                add([5] * 17, 8)
                add([10] * 4, 8)
            engine.step()
            step += 1
            assert step < 80
    finally:
        del engine.scheduler.schedule
    assert [s.num_completion_tokens for s in jobs] == [18, 8, 8]
    assert all(t['queries'] <= 8 for t in trace)
    assert not engine.scheduler.block_manager.used_block_ids
    assert all(b.ref_count == 0 for b in engine.scheduler.block_manager.blocks)
    return [s.completion_token_ids for s in jobs], trace


@pytest.mark.gpu
@pytest.mark.parametrize('mode,gamma', [('fp16-fp16', 1), ('fp16-fp16', 2),
                                       ('fp16-fp16', 4), ('fp16-kv4', 2), ('w4-kv4', 2)])
@pytest.mark.parametrize('eager', [True, False])
@torch.inference_mode()
def test_ngram_matches_baseline_and_uses_the_real_verify_window(checkpoint, mode, gamma, eager):
    baseline = make_engine(checkpoint, mode, 0, True)
    try:
        reference, _ = drive(baseline)
    finally:
        baseline.exit()
    engine = make_engine(checkpoint, mode, gamma, eager)
    try:
        actual, trace = drive(engine)
        assert actual == reference
        assert any(t['mixed_verify'] for t in trace)
        stats = engine.scheduler.spec_stats
        assert stats['accepted'] > 0 and stats['mean_len'] > 1
        assert engine.model_runner.varlen_verify_calls > 0
        if not eager:
            runner = engine.model_runner
            assert runner.cuda_graphs.verify_rows == gamma + 1
            assert runner.cuda_graphs.families[runner.cuda_graphs.verify_rows].buffers['input_ids'].numel() == 4 * (gamma + 1)
            assert runner.verify_graph_replays > 0
        seq = engine.add_request([7] * 5, SamplingParams(max_tokens=12, ignore_eos=True))
        engine.step()
        engine.abort_request(seq.seq_id)
        assert not engine.scheduler.block_manager.used_block_ids
    finally:
        engine.exit()


@pytest.mark.gpu
@torch.inference_mode()
def test_verify_padding_near_context_limit_uses_actual_varlen_queries(checkpoint):
    engine = make_engine(checkpoint, 'fp16-fp16', 4, False, max_len=256)
    try:
        seq = engine.add_request([7] * 252, SamplingParams(max_tokens=4, temperature=0,
                                                        ignore_eos=True))
        while not seq.is_finished:
            engine.step()
        assert len(seq) == 256 and seq.num_completion_tokens == 4
        assert engine.model_runner.varlen_verify_calls > 0
        assert not engine.scheduler.block_manager.used_block_ids
    finally:
        engine.exit()
