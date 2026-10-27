"""Draft cache rollback, actual probabilities and mixed target execution."""
import pytest
import torch
from transformers import Qwen3ForCausalLM

from qslab.runtime.config import Config
from qslab.runtime.speculation.draft import DraftProposer, sample_proposals
from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.sequence import Sequence


def test_draft_sampling_reports_the_actual_distribution():
    logits = torch.tensor([[0., 0.0001, -0.0002], [1., 2., 3.], [3., 2., 1.]])
    tokens, probs = sample_proposals(logits, [1e-4, 0., 0.7])
    assert tokens[1] == 2
    torch.testing.assert_close(probs[0], torch.softmax(logits[0] / 1e-4, -1))
    torch.testing.assert_close(probs[2], torch.softmax(logits[2] / 0.7, -1))
    assert sample_proposals(logits, [0., 0., 0.])[1] is None


@pytest.mark.gpu
@pytest.mark.parametrize('eager', [True, False])
@pytest.mark.parametrize('accepted', [0, 2, 4])
@torch.inference_mode()
def test_draft_cache_after_rejection_or_full_acceptance_matches_fresh_model(checkpoint, eager, accepted):
    source, _, _ = checkpoint
    config = Config(str(source), max_model_len=512, max_num_seqs=1,
                    max_num_batched_tokens=512, num_kvcache_blocks=3, enforce_eager=eager)
    proposer = DraftProposer(config)
    target = Sequence([7] * 254, SamplingParams(temperature=0.7, max_tokens=32))
    try:
        first = proposer.propose_batch([target], {target.seq_id: 4})[target.seq_id]
        assert len(first.tokens) == 4
        target.token_ids.extend(first.tokens[:accepted] + [19])
        target.num_tokens = len(target.token_ids)
        target.last_token = 19
        old_cached = proposer.drafts[target.seq_id].num_cached_tokens
        seq = proposer._sync(target)
        assert seq.num_cached_tokens == min(254 + accepted, old_cached)
        assert len(target) - seq.num_cached_tokens == (2 if accepted == 4 else 1)
        second = proposer.propose_batch([target], {target.seq_id: 1})[target.seq_id]
        hf = Qwen3ForCausalLM.from_pretrained(source, torch_dtype=torch.float16).cuda().eval()
        logits = hf(torch.tensor([target.token_ids], device='cuda')).logits[0, -1]
        expected = torch.softmax(logits.float() / target.temperature, -1)
        torch.testing.assert_close(second.probabilities[0], expected, atol=0.004, rtol=0.03)
        assert len(proposer.drafts[target.seq_id].block_table) <= 2
        del hf
    finally:
        proposer.exit()
    assert not proposer.block_manager.used_block_ids


def engine(checkpoint, gamma, eager, w4=False, quantized_draft=False):
    source, quantized, calibration = checkpoint
    return LLMEngine(str(quantized if w4 else source),
        kv_cache_dtype='kv4' if w4 else 'fp16', smooth_kv=str(calibration) if w4 else None,
        spec_method='draft' if gamma else None, spec_num_drafts=gamma or 4,
        draft_model=str(quantized if quantized_draft else source) if gamma else None,
        draft_kv_cache_dtype='kv4' if quantized_draft else 'fp16',
        draft_smooth_kv=str(calibration) if quantized_draft else None,
        draft_num_kvcache_blocks=4,
        enable_chunked_prefill=True, enforce_eager=eager,
        max_num_seqs=4, max_model_len=64, max_num_batched_tokens=8,
        num_kvcache_blocks=4, kvcache_block_size=256)


def drive(eng):
    jobs = [eng.add_request([3, 8, 9, 4, 6, 2, 7],
            SamplingParams(max_tokens=18, temperature=0, ignore_eos=True))]
    seen_mixed = False
    original = eng.scheduler.schedule
    def schedule():
        nonlocal seen_mixed
        seqs, prefill = original()
        assert sum(s.num_scheduled_tokens for s in seqs) <= 8
        seen_mixed |= prefill and any(s.spec_verify for s in seqs)
        return seqs, prefill
    eng.scheduler.schedule = schedule
    try:
        for step in range(80):
            if step == 2:
                for ids in ([5] * 17, [10] * 4):
                    jobs.append(eng.add_request(ids, SamplingParams(max_tokens=8,
                                                        temperature=0, ignore_eos=True)))
            if eng.is_finished():
                break
            eng.step()
        else:
            raise AssertionError('engine made no progress')
    finally:
        del eng.scheduler.schedule
    assert [s.num_completion_tokens for s in jobs] == [18, 8, 8]
    assert not eng.scheduler.block_manager.used_block_ids
    return [s.completion_token_ids for s in jobs], seen_mixed


@pytest.mark.gpu
@pytest.mark.parametrize('eager,gamma,w4,quantized_draft', [(True, 2, False, False),
    (False, 1, False, False), (False, 2, False, False), (False, 4, False, False),
    (False, 2, True, False), (True, 4, True, True), (False, 4, True, True)])
@torch.inference_mode()
def test_draft_matches_greedy_baseline_with_graph_and_chunked_prefill(checkpoint, eager, gamma, w4, quantized_draft):
    baseline = engine(checkpoint, 0, True, w4)
    try:
        expected, _ = drive(baseline)
    finally:
        baseline.exit()
    eng = engine(checkpoint, gamma, eager, w4, quantized_draft)
    try:
        if quantized_draft:
            runner = eng.proposer.runner
            assert runner.config.kv_cache_dtype == 'kv4'
            assert runner.kv_cache[0][0][0].dtype == torch.uint32
            assert any(hasattr(m, 'weight_memory_bytes') for m in runner.model.modules())
        actual, mixed = drive(eng)
        assert actual == expected
        assert mixed and eng.scheduler.spec_stats['accepted'] > 0
        assert not eng.proposer.drafts
        assert not eng.proposer.block_manager.used_block_ids
        if not eager:
            assert eng.model_runner.cuda_graphs.verify_rows == gamma + 1
            assert eng.model_runner.verify_graph_replays > 0
            assert eng.proposer.runner.decode_graph_replays > 0
        seq = eng.add_request([9] * 5, SamplingParams(max_tokens=16, ignore_eos=True))
        eng.step()
        eng.step()
        assert seq.seq_id in eng.proposer.drafts
        eng.abort_request(seq.seq_id)
        assert not eng.proposer.drafts and not eng.proposer.block_manager.used_block_ids
    finally:
        eng.exit()


@pytest.mark.gpu
@pytest.mark.parametrize('quantized_draft', [False, True])
@torch.inference_mode()
def test_draft_sampling_mixed_temperatures_and_exception_cleanup(checkpoint, quantized_draft):
    eng = engine(checkpoint, 2, False, w4=quantized_draft, quantized_draft=quantized_draft)
    try:
        for t in (0., 0.7, 1e-4):
            eng.add_request([2, 4, 7], SamplingParams(temperature=t, max_tokens=10, ignore_eos=True))
        while not eng.is_finished():
            eng.step()
        assert not eng.proposer.drafts and not eng.proposer.block_manager.used_block_ids
        eng.add_request([8] * 5, SamplingParams(max_tokens=12, ignore_eos=True))
        eng.step()
        def fail(*_):
            raise RuntimeError('draft failed')
        eng.proposer.runner.run_model = fail
        with pytest.raises(RuntimeError, match='draft failed'):
            eng.step()
        assert eng.is_finished() and not eng.scheduler.block_manager.used_block_ids
        assert not eng.proposer.drafts and not eng.proposer.block_manager.used_block_ids
    finally:
        eng.exit()


@pytest.mark.gpu
@pytest.mark.parametrize('checkpoint', [True], indirect=True)
@torch.inference_mode()
def test_quantized_tied_embedding_draft_loads_and_releases(checkpoint):
    eng = engine(checkpoint, 4, False, w4=True, quantized_draft=True)
    try:
        for runner in (eng.model_runner, eng.proposer.runner):
            assert runner.model.lm_head.weight is runner.model.model.embed_tokens.weight
            assert not any(p.is_meta for p in runner.model.parameters())
        _, mixed = drive(eng)
        assert mixed and eng.proposer.runner.decode_graph_replays > 0
        assert not eng.proposer.drafts and not eng.proposer.block_manager.used_block_ids
    finally:
        eng.exit()
