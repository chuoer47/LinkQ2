"""Offline/online boundary and real packed layer lifetime tests."""
import gc
import json
import shutil
import weakref

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file

from qslab.compression.checkpoint import save_checkpoint
from qslab.compression.schema import CompressionArtifact
from qslab.compression.algorithms import AWQAffine, QUANTIZERS, RTN, WeightQuantizer
from qslab.runtime.quantization import QuantizedWeights, load_checkpoint_weights
from qslab.quantization import QuantizationMethod as Method
from qslab.runtime.strategies import RUNTIME_STRATEGIES, RuntimeQuantization


def save_source(model, path):
    path.mkdir()
    save_file(model.state_dict(), str(path / 'model.safetensors'))
    (path / 'config.json').write_text('{}')


@pytest.mark.parametrize('affine', [False, True])
def test_artifact_roundtrip(tmp_path, affine):
    torch.manual_seed(123)
    w = torch.randn(64, 128)
    quantizer = AWQAffine() if affine else RTN()
    q, transform = quantizer.quantize(w, group_size=64, activation_absmean=torch.ones(128))
    transforms = {'input_scale': {'proj.weight': transform}} if transform is not None else {}
    source, output = tmp_path / 'source', tmp_path / 'checkpoint'
    model = nn.ModuleDict({'proj': nn.Linear(128, 64, bias=False)})
    save_source(model, source)
    save_checkpoint(CompressionArtifact(Method.AWQ_AFFINE if affine else Method.RTN, {'proj.weight': q}, transforms), source, output)
    online = QuantizedWeights(output)
    assert online.names == ['proj.weight']
    assert torch.equal(online.layer('proj.weight')['values'], q.values)
    assert torch.equal(online.layer('proj.weight')['scale'], q.scale)
    if affine:
        assert torch.equal(online.layer('proj.weight')['zero_point'], q.zero_point)
        assert torch.equal(online.layer('proj.weight')['input_scale'], transform)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 128)
        self.q_proj = nn.Linear(128, 64, bias=False)
        self.lm_head = nn.Linear(64, 32, bias=False)

    def forward(self, input_ids, use_cache=False):
        return self.lm_head(self.q_proj(self.embed(input_ids)))


def test_quantizer_interface_is_abstract():
    with pytest.raises(TypeError):
        WeightQuantizer()
    with pytest.raises(TypeError):
        RuntimeQuantization()


def test_layer_selection_and_transform_are_explicit(tmp_path):
    """A scale tensor cannot opt a layer into quantization or input smoothing."""
    source, output = tmp_path / 'source', tmp_path / 'checkpoint'
    model = nn.ModuleDict({'proj': nn.Linear(128, 64, bias=False)})
    save_source(model, source)
    q, _ = RTN().quantize(model['proj'].weight, group_size=128)
    save_checkpoint(CompressionArtifact(Method.RTN, {'proj.weight': q}), source, output)
    layers = output / 'quantized_layers.json'
    layers.write_text('[]')
    assert QuantizedWeights(output).names == []
    layers.write_text('["proj"]')
    config_path = output / 'config.json'
    config = json.loads(config_path.read_text())
    config['quantization_config']['quant_method'] = Method.AWQ.value
    config_path.write_text(json.dumps(config))
    weights = QuantizedWeights(output)
    assert isinstance(weights.strategy, RUNTIME_STRATEGIES[Method.AWQ])
    with pytest.raises(Exception, match='input_scale'):
        weights.layer('proj.weight')
    config['quantization_config']['quant_method'] = 'unknown_algorithm'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unknown_algorithm'):
        QuantizedWeights(output)


@pytest.mark.parametrize('algorithm', QUANTIZERS)
def test_offline_pipeline(tmp_path, algorithm):
    from qslab.compression import CompressionConfig, quantize_model
    torch.manual_seed(123)
    model = TinyModel()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    needs_calibration = QUANTIZERS[algorithm].requires_calibration
    ids = [[1, 2, 3, 4], [5, 6, 7, 8]] if needs_calibration else None
    artifact = quantize_model(model, CompressionConfig(algorithm=algorithm, device='cpu'), ids)
    source, output = tmp_path / 'source', tmp_path / 'checkpoint'
    save_source(model, source)
    save_checkpoint(artifact, source, output)
    loaded = QuantizedWeights(output)
    assert loaded.names == ['q_proj.weight']
    assert loaded.config['quant_method'] == algorithm.value
    assert artifact.algorithm == algorithm
    assert json.loads((output / 'quantized_layers.json').read_text()) == ['q_proj']
    assert (loaded.layer('q_proj.weight')['input_scale'] is not None) == needs_calibration
    assert loaded.config['symmetric'] == (algorithm != Method.AWQ_AFFINE)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('algorithm', [Method.AWQ, Method.AWQ_AFFINE])
def test_pipeline_requires_calibration(algorithm):
    from qslab.compression import CompressionConfig, quantize_model
    with pytest.raises(ValueError, match='calibration'):
        quantize_model(TinyModel(), CompressionConfig(algorithm=algorithm, device='cpu'))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('affine', [False, True])
def test_replacement_device_move_and_release(tmp_path, affine):
    from qslab.models.quant_linear import QuantLinear
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(1024, 256, bias=False)).half()
    source = weakref.ref(model[0].weight)
    weight = model[0].weight.detach().float()
    quantizer = AWQAffine() if affine else RTN()
    q, transform = quantizer.quantize(weight, group_size=128, activation_absmean=torch.ones(1024))
    source_path, output_path = tmp_path / 'source', tmp_path / 'checkpoint'
    save_source(model, source_path)
    save_checkpoint(CompressionArtifact(Method.AWQ_AFFINE if affine else Method.RTN, {'0.weight': q},
        {'input_scale': {'0.weight': transform}} if transform is not None else {}), source_path, output_path)
    torch.cuda.synchronize()
    # Establish the persistent cuBLAS workspace before measuring model allocations.
    torch.mm(torch.ones(1, 32, device='cuda'), torch.ones(32, 32, device='cuda'))
    baseline = torch.cuda.memory_allocated()
    assert load_checkpoint_weights(model, output_path) == 1
    gc.collect()
    assert source() is None
    assert isinstance(model[0], QuantLinear)
    packed = model[0]
    assert not list(packed.parameters())
    assert packed.weight_memory_bytes() < weight.numel()
    zero = q.zero_point.float().unsqueeze(-1)
    dense = ((q.values.float().reshape(256, 8, 128) - zero) * q.scale.float().unsqueeze(-1)).reshape(256, 1024).cuda()
    for m in (1, 32, 128):
        x = torch.randn(m, 1024, device='cuda', dtype=torch.float16)
        scaled = x if transform is None else x / transform.cuda()
        expected = scaled.float() @ dense.t()
        actual = model(x).float()
        assert (actual - expected).norm() / expected.norm() < 0.01
    before = model(x)
    model.cpu().cuda()
    assert all(t.device.type == 'cuda' for t in model.buffers())
    assert torch.allclose(model(x), before, rtol=0.01, atol=0.02)
    del model, packed, x, scaled, dense, expected, actual, before
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == baseline


@pytest.fixture
def hf_source(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    config = Qwen3Config(vocab_size=32, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config).bfloat16().eval()
    source = tmp_path / 'original'
    model.save_pretrained(source, max_shard_size='64KB')
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'hello': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(source)
    return model, source


@pytest.mark.parametrize('algorithm', list(Method))
def test_complete_checkpoint_without_source(tmp_path, hf_source, algorithm):
    from adapters.model_builder import build_quantized_causal_lm
    from qslab.compression import CompressionConfig, quantize_model
    from transformers import AutoTokenizer
    model, source = hf_source
    artifact = quantize_model(model, CompressionConfig(algorithm=algorithm, device='cpu'), [[1, 1, 1, 1]])
    output = tmp_path / 'complete'
    before = json.loads((source / 'model.safetensors.index.json').read_text())
    save_checkpoint(artifact, source, output)
    after = json.loads((output / 'model.safetensors.index.json').read_text())
    assert all(after['weight_map'][name] == shard for name, shard in before['weight_map'].items())
    total = 0
    for shard in set(after['weight_map'].values()):
        with safe_open(str(output / shard), framework='pt') as file:
            assert all(after['weight_map'][name] == shard for name in file.keys())
            for name in file.keys():
                t = file.get_tensor(name)
                total += t.numel() * t.element_size()
                if name in before['weight_map']:
                    if name in artifact.weights:
                        assert t.dtype == torch.int8
                        assert torch.equal(t, artifact.weights[name].values)
                    else:
                        assert torch.equal(t, model.state_dict()[name])
                        assert t.dtype == torch.bfloat16
    assert total == after['metadata']['total_size']
    assert not (output / 'manifest.json').exists()
    shutil.rmtree(source)
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert tokenizer.encode('hello') == [1]
    loaded = build_quantized_causal_lm(output, device='cpu', dtype=torch.float32, fake=True)
    for name, quantized in artifact.weights.items():
        scale = quantized.scale.float()
        dense = ((quantized.values.float().reshape(*scale.shape, -1) - quantized.zero_point.float().unsqueeze(-1)) * scale.unsqueeze(-1)).reshape_as(quantized.values)
        input_scale = artifact.transforms.get('input_scale', {}).get(name)
        if input_scale is not None:
            dense /= input_scale.float()
        assert torch.equal(loaded.get_parameter(name), dense)
    with torch.inference_mode():
        logits = loaded(input_ids=torch.tensor([[1, 1]]), use_cache=False).logits
    assert logits.shape == (1, 2, 32) and torch.isfinite(logits).all()
