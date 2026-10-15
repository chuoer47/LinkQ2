"""Offline/online boundary and real packed layer lifetime tests."""
import gc
import weakref

import pytest
import torch
from torch import nn

from qslab.artifacts.io import load_artifact, save_artifact
from qslab.artifacts.schema import CompressionArtifact
from qslab.compression.algorithms import AWQAffine, QUANTIZERS, RTN, WeightQuantizer
from qslab.runtime.quantization import QuantizedWeights, replace_quantized_linears


@pytest.mark.parametrize('affine', [False, True])
def test_artifact_roundtrip(tmp_path, affine):
    torch.manual_seed(123)
    w = torch.randn(64, 128)
    quantizer = AWQAffine() if affine else RTN()
    q, transform = quantizer.quantize(w, group_size=64, activation_absmean=torch.ones(128))
    transforms = {'input_scale': {'proj.weight': transform}} if transform is not None else {}
    save_artifact(CompressionArtifact({'proj.weight': q}, transforms), tmp_path)
    online = QuantizedWeights(tmp_path)
    offline = load_artifact(tmp_path)
    assert online.names == ['proj.weight']
    assert torch.equal(online.layer('proj.weight')['values'], q.values)
    assert torch.equal(offline.weights['proj.weight'].scale, q.scale)
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


@pytest.mark.parametrize('algorithm', QUANTIZERS)
def test_offline_pipeline(tmp_path, algorithm):
    from qslab.compression import CompressionConfig, quantize_model
    torch.manual_seed(123)
    model = TinyModel()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    needs_calibration = QUANTIZERS[algorithm].requires_calibration
    ids = [[1, 2, 3, 4], [5, 6, 7, 8]] if needs_calibration else None
    artifact = quantize_model(model, CompressionConfig(algorithm=algorithm, device='cpu'), ids)
    save_artifact(artifact, tmp_path)
    loaded = QuantizedWeights(tmp_path)
    assert loaded.names == ['q_proj.weight']
    assert (loaded.layer('q_proj.weight')['input_scale'] is not None) == needs_calibration
    assert loaded.config['symmetric'] == (algorithm != 'awq_affine')
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('algorithm', ['awq', 'awq_affine'])
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
    save_artifact(CompressionArtifact({'0.weight': q},
        {'input_scale': {'0.weight': transform}} if transform is not None else {}), tmp_path)
    torch.cuda.synchronize()
    # Establish the persistent cuBLAS workspace before measuring model allocations.
    torch.mm(torch.ones(1, 32, device='cuda'), torch.ones(32, 32, device='cuda'))
    baseline = torch.cuda.memory_allocated()
    assert replace_quantized_linears(model, tmp_path) == 1
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
