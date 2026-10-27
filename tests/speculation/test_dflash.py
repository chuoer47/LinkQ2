"""Official DFlash code at tiny dimensions: lifecycle and cache correctness."""
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, Qwen3ForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.sequence import Sequence


@pytest.fixture
def dflash_checkpoint(checkpoint, tmp_path):
    official = Path(os.environ.get('DFLASH_CHECKPOINT', 'models/Qwen3-8B-DFlash-b16'))
    if not official.is_dir():
        pytest.skip('requires the official DFlash checkpoint code')
    cls = get_class_from_dynamic_module('dflash.DFlashDraftModel', str(official), local_files_only=True)
    config = AutoConfig.from_pretrained(checkpoint[0])
    config.num_target_layers = config.num_hidden_layers
    config.num_hidden_layers = 1
    config.layer_types = ['full_attention']
    config.block_size = 4
    config.dflash_config = {'mask_token_id': 125, 'target_layer_ids': [0]}
    cls.register_for_auto_class('AutoModel')
    torch.manual_seed(13)
    model = cls(config).bfloat16().eval()
    output = tmp_path/'dflash'
    model.save_pretrained(output)
    return output


def engine(checkpoint, draft, *, eager=False, w4=False, max_len=128, draft_graph=True):
    source, quantized, calibration = checkpoint
    return LLMEngine(str(quantized if w4 else source), draft_model=str(draft),
        spec_method='dflash', kv_cache_dtype='kv4' if w4 else 'fp16',
        smooth_kv=str(calibration) if w4 else None, enable_chunked_prefill=True,
        max_num_seqs=4, max_model_len=max_len, max_num_batched_tokens=8,
        enforce_eager=eager, num_kvcache_blocks=4, dflash_graph=draft_graph)


@pytest.mark.gpu
@pytest.mark.parametrize('eager,w4', [(True, False), (False, False), (False, True)])
@torch.inference_mode()
def test_dflash_greedy_mixed_prefill_graph_and_cleanup(checkpoint, dflash_checkpoint, eager, w4):
    def drive(eng):
        jobs = [eng.add_request([3]*5, SamplingParams(max_tokens=18, temperature=0, ignore_eos=True))]
        mixed = False
        for step in range(100):
            if step == 2:
                jobs.append(eng.add_request([8]*17, SamplingParams(max_tokens=9,
                    temperature=0, ignore_eos=True)))
            if eng.is_finished():
                break
            eng.step()
            mixed |= eng.model_runner.varlen_verify_calls > 0
        assert eng.is_finished() and [s.num_completion_tokens for s in jobs] == [18, 9]
        return [s.completion_token_ids for s in jobs], mixed
    source, quantized, calibration = checkpoint
    base = LLMEngine(str(quantized if w4 else source), enforce_eager=True,
        kv_cache_dtype='kv4' if w4 else 'fp16', smooth_kv=str(calibration) if w4 else None,
        enable_chunked_prefill=True, max_num_seqs=4, max_model_len=128,
        max_num_batched_tokens=8, num_kvcache_blocks=4)
    try:
        expected, _ = drive(base)
    finally:
        base.exit()
    eng = engine(checkpoint, dflash_checkpoint, eager=eager, w4=w4)
    try:
        actual, mixed = drive(eng)
        assert actual == expected and mixed and eng.proposer.calls > 0
        assert not eng.proposer.drafts and not eng.scheduler.block_manager.used_block_ids
        if not eager:
            assert eng.model_runner.cuda_graphs.verify_rows == 4 and eng.model_runner.verify_graph_replays > 0
            assert eng.proposer.graph_runner.replays > 0
        with pytest.raises(ValueError, match='greedy'):
            eng.add_request([3], SamplingParams(temperature=0.7))
        seq = eng.add_request([7]*3, SamplingParams(max_tokens=16, temperature=0, ignore_eos=True))
        eng.step()
        assert seq.seq_id in eng.proposer.drafts
        eng.abort_request(seq.seq_id)
        assert not eng.proposer.drafts
        # Re-prefill after preemption replaces the old feature/cache prefix.
        seq = eng.add_request([7]*17, SamplingParams(max_tokens=12, temperature=0, ignore_eos=True))
        while not seq.num_completion_tokens:
            eng.step()
        eng.scheduler.running.remove(seq)
        eng.scheduler.preempt(seq)
        eng.step()
        assert eng.proposer.drafts[seq.seq_id].length == seq.num_cached_tokens
        while not eng.is_finished():
            eng.step()
        assert not eng.proposer.drafts
        eng.add_request([7]*3, SamplingParams(max_tokens=16, temperature=0, ignore_eos=True))
        eng.step()
        def fail(*args, **kwargs):
            raise RuntimeError('dflash failed')
        eng.proposer.propose_batch = fail
        with pytest.raises(RuntimeError, match='dflash failed'):
            eng.step()
        assert not eng.proposer.drafts and eng.is_finished()
    finally:
        eng.exit()


@pytest.mark.gpu
@pytest.mark.parametrize('accepted', [0, 2, 3])
@torch.inference_mode()
def test_dflash_cached_features_after_acceptance_match_fresh_forward(checkpoint, dflash_checkpoint, accepted):
    eng = engine(checkpoint, dflash_checkpoint, max_len=512, draft_graph=False)
    hf = Qwen3ForCausalLM.from_pretrained(checkpoint[0], torch_dtype=torch.float16,
        attn_implementation='eager').cuda().eval()
    seq = Sequence([7]*254, SamplingParams(temperature=0, max_tokens=32))
    proposer = eng.proposer
    def features():
        return hf(torch.tensor([seq.token_ids[:-1]], device='cuda'),
                  output_hidden_states=True).hidden_states[1][0]
    try:
        proposer.observe(seq.seq_id, 0, features())
        first = proposer.propose_batch([seq], {seq.seq_id: 3})[seq.seq_id]
        for token in first.tokens[:accepted] + [19]:
            seq.append_token(token)
        full_features = features()
        proposer.observe(seq.seq_id, 253, full_features[253:])
        captured = []
        hook = eng.model_runner.model.lm_head.register_forward_hook(
            lambda _m, _args, output: captured.append(output.detach().clone()))
        try:
            cached = proposer.propose_batch([seq], {seq.seq_id: 3})[seq.seq_id]
            assert proposer.drafts[seq.seq_id].cache.get_seq_length() == len(seq)-1
            proposer.drop_seqs([seq.seq_id])
            proposer.observe(seq.seq_id, 0, full_features)
            fresh = proposer.propose_batch([seq], {seq.seq_id: 3})[seq.seq_id]
        finally:
            hook.remove()
        torch.testing.assert_close(captured[0], captured[1], atol=0.02, rtol=0.02)
        assert cached.tokens == fresh.tokens
    finally:
        del hf
        eng.exit()


@pytest.mark.gpu
@pytest.mark.parametrize('accepted', [0, 2, 3])
@torch.inference_mode()
def test_dflash_static_graph_matches_dynamic_cache_and_reuses_slots(checkpoint, dflash_checkpoint, accepted):
    graph = engine(checkpoint, dflash_checkpoint, max_len=512)
    dynamic = engine(checkpoint, dflash_checkpoint, max_len=512, draft_graph=False)
    hf = Qwen3ForCausalLM.from_pretrained(checkpoint[0], torch_dtype=torch.float16,
        attn_implementation='eager').cuda().eval()
    seq = Sequence([7]*254, SamplingParams(temperature=0, max_tokens=32))
    def features():
        return hf(torch.tensor([seq.token_ids[:-1]], device='cuda'),
                  output_hidden_states=True).hidden_states[1][0]
    def propose():
        a = graph.proposer.propose_batch([seq], {seq.seq_id: 3})[seq.seq_id]
        b = dynamic.proposer.propose_batch([seq], {seq.seq_id: 3})[seq.seq_id]
        assert a.tokens == b.tokens
        gs, ds = graph.proposer.drafts[seq.seq_id], dynamic.proposer.drafts[seq.seq_id]
        assert gs.cached_length == ds.cached_length == len(seq)-1
        for fixed, growing in zip(gs.cache.layers, ds.cache.layers):
            torch.testing.assert_close(fixed.keys[:, :, :gs.cached_length], growing.keys,
                                       atol=0.02, rtol=0.02)
            torch.testing.assert_close(fixed.values[:, :, :gs.cached_length], growing.values,
                                       atol=0.02, rtol=0.02)
        return a.tokens
    try:
        for eng in (graph, dynamic):
            eng.proposer.observe(seq.seq_id, 0, features())
        tokens = propose()
        for token in tokens[:accepted]+[19]:
            seq.append_token(token)
        for eng in (graph, dynamic):
            eng.proposer.observe(seq.seq_id, 253, features()[253:])
        propose()
        assert graph.proposer.graph_runner.replays == 1
        for eng in (graph, dynamic):
            eng.proposer.drop_seqs([seq.seq_id])
        seq = Sequence([8]*3, SamplingParams(temperature=0, max_tokens=16))
        for eng in (graph, dynamic):
            eng.proposer.observe(seq.seq_id, 0, features())
        propose()
        assert len(graph.proposer.graph_runner.slots) == 1
        assert graph.proposer.graph_runner.replays == 2
    finally:
        del hf
        graph.exit()
        dynamic.exit()
