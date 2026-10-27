"""Cached-prefix execution against a cold run of the same tiny checkpoint."""
import pytest
import torch

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams


@pytest.mark.gpu
@pytest.mark.parametrize('kv', ['fp16', 'kv4'])
@torch.inference_mode()
def test_prefix_hit_reuses_pages_and_preserves_final_chunk_logits(checkpoint, kv):
    source, _, calibration = checkpoint
    eng = LLMEngine(str(source), kv_cache_dtype=kv,
        smooth_kv=str(calibration) if kv == 'kv4' else None,
        enable_chunked_prefill=True, enforce_eager=False,
        max_num_seqs=1, max_model_len=512, max_num_batched_tokens=64,
        num_kvcache_blocks=4)
    logits = []
    hook = eng.model_runner.model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: logits.append(output.detach().clone()))
    prompt = [i % 32 + 1 for i in range(273)]
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    try:
        cold_seq = eng.add_request(prompt, params)
        while not eng.is_finished():
            eng.step()
        cold = list(cold_seq.completion_token_ids)
        reference = list(logits)
        logits.clear()
        seq = eng.add_request(prompt, params)
        eng.step()
        assert seq.num_cached_tokens == len(prompt)
        assert seq.num_completion_tokens == 1
        while not eng.is_finished():
            eng.step()
        assert seq.completion_token_ids == cold
        assert len(logits) == len(reference) == 4
        for actual, expected in zip(logits, reference):
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.01)
        assert not eng.scheduler.block_manager.used_block_ids
    finally:
        hook.remove()
        eng.exit()
