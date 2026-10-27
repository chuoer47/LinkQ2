"""HF block outputs, including residuals, survive decode/verify graph replay."""
import pytest
import torch
from transformers import Qwen3ForCausalLM

from qslab.runtime.config import Config
from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.model.context import reset_context
from qslab.runtime.state.sequence import Sequence


@pytest.mark.gpu
@pytest.mark.parametrize('eager', [True, False])
@torch.inference_mode()
def test_target_features_match_hf_before_and_after_graph_replay(checkpoint, eager):
    source, _, _ = checkpoint
    runner = ModelRunner(Config(str(source), max_model_len=128, max_num_seqs=1,
        max_num_batched_tokens=64, num_kvcache_blocks=2, enforce_eager=eager,
        spec_method='ngram', spec_num_drafts=15), feature_layers=(0,))
    hf = Qwen3ForCausalLM.from_pretrained(source, torch_dtype=torch.float16,
                                         attn_implementation='eager').cuda().eval()
    seq = Sequence([7, 8, 11, 4] * 7)
    seq.block_table = [0]
    def check(ids, expected_rows):
        reference = hf(torch.tensor([ids], device='cuda'), output_hidden_states=True).hidden_states[1]
        torch.testing.assert_close(runner.context_features, reference[0, -expected_rows:],
                                   atol=0.003, rtol=0.01)
    try:
        seq.num_scheduled_tokens = len(seq)
        runner.run([seq], True)
        check(seq.token_ids, len(seq))
        for token in (19, 21):
            seq.num_cached_tokens = len(seq)
            seq.append_token(token)
            seq.is_prefill = False
            seq.num_scheduled_tokens = 1
            runner.run([seq], False)
            check(seq.token_ids, 1)
        seq.spec_drafts = [5, 6, 9] * 5
        seq.num_scheduled_tokens = 16
        ids, positions = runner.prepare_verify([seq])
        runner.run_model(ids, positions, False, rows=16)
        check(seq.token_ids + seq.spec_drafts, 16)
        assert runner.feature_ranges == [(seq.seq_id, len(seq)-1, 0, 16)]
        if not eager:
            assert runner.decode_graph_replays == 2 and runner.verify_graph_replays == 1
    finally:
        reset_context()
        del hf
        runner.exit()
