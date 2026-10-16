"""Small real checkpoint: FP16 paged execution versus Transformers."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qslab.runtime.config import Config
from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.state.sequence import Sequence
from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.sampling_params import SamplingParams


@pytest.mark.gpu
def test_fp16_prefill_decode_and_page_reuse_match_hf(tmp_path):
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, max_position_embeddings=64, tie_word_embeddings=False)
    hf = Qwen3ForCausalLM(cfg).half().eval().cuda()
    hf.save_pretrained(tmp_path)
    prompt = [3, 8, 9, 4, 6, 2, 7]
    with torch.inference_mode():
        reference = hf.generate(torch.tensor([prompt], device='cuda'),
            max_new_tokens=8, do_sample=False, eos_token_id=None,
            pad_token_id=0)[0, len(prompt):].tolist()
    baseline = torch.cuda.memory_allocated()
    config = Config(str(tmp_path), max_model_len=32, max_num_seqs=2,
                    max_num_batched_tokens=4, num_kvcache_blocks=2)
    Sequence.block_size = config.kvcache_block_size
    runner = ModelRunner(config)
    scheduler = Scheduler(config)
    try:
        # Run twice against the same pool; each prompt is chunked into 4+3 rows.
        for _ in range(2):
            seq = Sequence(prompt, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
            scheduler.add(seq)
            while not scheduler.is_finished():
                batch, prefill = scheduler.schedule()
                scheduler.postprocess(batch, runner.run(batch, prefill), prefill)
            assert seq.completion_token_ids == reference
            assert not scheduler.block_manager.used_block_ids
    finally:
        runner.exit()
    assert torch.cuda.memory_allocated() <= baseline + 64 * 1024
