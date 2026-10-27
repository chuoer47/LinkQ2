"""Small deterministic checkpoints shared by runtime and speculative tests."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_collection_modifyitems(items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason='CUDA required; run CPU tests with -m "not gpu"')
        for item in items:
            if item.get_closest_marker('gpu'):
                item.add_marker(skip)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, request):
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, max_position_embeddings=512,
        tie_word_embeddings=getattr(request, 'param', False))
    hf = Qwen3ForCausalLM(cfg).half().eval()
    # A strong, repeatable greedy continuation; attention is checked numerically
    # in the separate mixed KV4 causal reference test.
    with torch.no_grad():
        hf.lm_head.weight.copy_(hf.model.embed_tokens.weight)
    source = tmp_path / 'fp16'
    hf.save_pretrained(source)
    from qslab.compression.pipeline import quantize_model
    from qslab.compression.config import CompressionConfig
    from qslab.compression.checkpoint import save_checkpoint
    artifact = quantize_model(hf, CompressionConfig(group_size=128, device='cpu'))
    quantized = tmp_path / 'w4'
    save_checkpoint(artifact, source, quantized)
    calibration = tmp_path / 'kv4.pt'
    torch.save({'lambda': torch.ones(2, 1, 64),
                'kscale': torch.full((2, 1, 64), 0.3)}, calibration)
    monkeypatch.setattr('qslab.runtime.engine.llm_engine.AutoTokenizer.from_pretrained',
                        lambda *a, **k: SimpleNamespace(eos_token_id=127, all_special_ids=[127], get_vocab=lambda: {str(i): i for i in range(128)}))
    return source, quantized, calibration
