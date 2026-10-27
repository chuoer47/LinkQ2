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
@pytest.mark.parametrize('enforce_eager', [True, False])
@pytest.mark.parametrize('chunked', [False, True])
def test_fp16_prefill_decode_and_page_reuse_match_hf(tmp_path, enforce_eager, chunked):
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
        if not enforce_eager:
            # cuBLAS caches one workspace per stream. Initialize the shared
            # capture stream before measuring model-lifetime allocations.
            warm = torch.zeros(16, 16, device='cuda', dtype=torch.float16)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                product = warm @ warm
            del graph, warm, product
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    config = Config(str(tmp_path), max_model_len=32, max_num_seqs=4,
                    max_num_batched_tokens=4, num_kvcache_blocks=4,
                    enforce_eager=enforce_eager)
    config.enable_chunked_prefill = chunked
    Sequence.block_size = config.kvcache_block_size
    runner = ModelRunner(config)
    scheduler = Scheduler(config)
    head_rows = []
    hook = runner.model.lm_head.register_forward_pre_hook(lambda module, inputs: head_rows.append(len(inputs[0])))
    mixed_steps = 0
    try:
        # Run twice against the same pool; each prompt is chunked into 4+3 rows.
        for _ in range(2):
            seq = Sequence(prompt, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
            scheduler.add(seq)
            while not scheduler.is_finished():
                batch, prefill = scheduler.schedule()
                before = len(head_rows)
                expected = sum(seq.should_sample for seq in batch)
                scheduler.postprocess(batch, runner.run(batch, prefill), prefill)
                assert len(head_rows) - before == int(expected > 0)
                if expected:
                    assert head_rows[-1] == expected
            assert seq.completion_token_ids == reference
            assert not scheduler.block_manager.used_block_ids
        # Three real rows replay the four-row bucket; padding must not write KV.
        seqs = [Sequence(prompt, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
                for _ in range(3)]
        for seq in seqs:
            scheduler.add(seq)
        while not scheduler.is_finished():
            batch, prefill = scheduler.schedule()
            mixed_steps += int(prefill and any(not s.is_prefill for s in batch))
            scheduler.postprocess(batch, runner.run(batch, prefill), prefill)
        assert all(seq.completion_token_ids == reference for seq in seqs)
        assert not scheduler.block_manager.used_block_ids
        if chunked:
            assert mixed_steps > 0
        if not enforce_eager:
            assert set(runner.cuda_graphs.families[1].graphs) == {1, 2, 4} and runner.graph_replays > 0
    finally:
        hook.remove()
        runner.exit()
    assert torch.cuda.memory_allocated() <= baseline + 64 * 1024
