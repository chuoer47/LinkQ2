"""KV4 calibration requirements and cache-layout binding regressions."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from qslab.runtime.config import Config


def test_fp16_baseline_needs_no_calibration(tmp_path):
    Qwen3Config(hidden_size=256, num_attention_heads=2, head_dim=128).save_pretrained(tmp_path)
    cfg = Config(str(tmp_path))
    assert cfg.kv_cache_dtype == 'fp16' and cfg.enforce_eager
    with pytest.raises(ValueError, match='enforce_eager'):
        Config(str(tmp_path), enforce_eager=False)


def test_calibration_is_required_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match='smooth_kv'):
        Config(str(tmp_path), kv_cache_dtype='kv4')
    with pytest.raises(FileNotFoundError):
        Config(str(tmp_path), kv_cache_dtype='kv4', smooth_kv=str(tmp_path / 'missing.pt'))


@pytest.mark.parametrize('group', [0, 96])
def test_invalid_v_group_is_refused(tmp_path, group):
    Qwen3Config(hidden_size=256, num_attention_heads=2, head_dim=128).save_pretrained(tmp_path)
    calibration = tmp_path / 'calibration.pt'
    calibration.touch()
    with pytest.raises(ValueError, match='v_group'):
        Config(str(tmp_path), kv_cache_dtype='kv4', smooth_kv=str(calibration), v_group=group)


@pytest.mark.gpu
@pytest.mark.parametrize('group', [32, 128])
@pytest.mark.parametrize('block_size', [128, 256])
def test_pool_binding_uses_configured_group_and_page(tmp_path, monkeypatch, group, block_size):
    from torch import nn
    from qslab.runtime.execute.model_runner import ModelRunner
    from qslab.runtime.model.attention import PagedAttention
    from qslab.runtime.model.context import set_context, reset_context
    from qslab.kernels.kv4 import materialize_kv

    cfg = Qwen3Config(hidden_size=256, num_hidden_layers=1, num_attention_heads=2,
                     num_key_value_heads=2, head_dim=128)
    calibration = tmp_path / 'calibration.pt'
    torch.save({'lambda': torch.ones(1, 2, 128), 'kscale': torch.full((1, 2, 128), 0.1)}, calibration)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=cfg, v_group=group, smooth_kv=str(calibration),
                                    gpu_memory_utilization=0.9, num_kvcache_blocks=-1,
                                    kv_cache_dtype='kv4', max_model_len=block_size * 2, max_num_seqs=1)
    runner.block_size = block_size
    runner.model = nn.ModuleList([PagedAttention(2, 128, 128**-0.5, 2)])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (2**20, 2**20))
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda: {
        'allocated_bytes.all.peak': 0, 'allocated_bytes.all.current': 0})
    runner.allocate_kv_cache()
    attention = runner.model[0]
    assert attention.v_group == group
    assert attention.block_n == block_size
    assert attention.v_cache[1].shape[-1] == 128 // group
    q = torch.randn(1, 2, 128, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    # Page 1 distinguishes logical-page stride from a coincidental slot-0 hit.
    slots = torch.tensor([block_size], device='cuda', dtype=torch.int32)
    set_context(False, slot_mapping=slots,
                context_lens=torch.ones(1, device='cuda', dtype=torch.int32),
                block_tables=torch.ones(1, 1, device='cuda', dtype=torch.int32))
    try:
        actual = attention(q, k, v)
        _, restored_v = materialize_kv(attention.k_cache, attention.v_cache, slots, v_group=group)
        torch.testing.assert_close(actual, restored_v, rtol=1e-3, atol=1e-3)
    finally:
        reset_context()
    torch.save({'lambda': torch.ones(1, 2, 128), 'kscale': torch.zeros(1, 2, 128)}, calibration)
    with pytest.raises(ValueError, match='kscale'):
        runner.load_smoothing(str(calibration))
