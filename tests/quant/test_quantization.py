"""Offline/online boundary and real packed layer lifetime tests."""
import gc
import weakref

import pytest
import torch
from torch import nn

from qslab.artifacts.io import load_artifact, save_artifact
from qslab.artifacts.schema import CompressionArtifact
from qslab.compression.algorithms.w4 import awq_affine, rtn
from qslab.runtime.quantization import QuantizedWeights, replace_quantized_linears


@pytest.mark.parametrize('affine', [False, True])
def test_artifact_roundtrip(tmp_path, affine):
    torch.manual_seed(123)
    w = torch.randn(64, 128)
    q, transform = awq_affine(w, torch.ones(128), group_size=64) if affine else rtn(w, 64)
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


def test_offline_rtn_pipeline(tmp_path):
    from qslab.compression import CompressionConfig, quantize_model
    model = nn.ModuleDict({'q_proj': nn.Linear(128, 64, bias=False)})
    artifact = quantize_model(model, CompressionConfig(device='cpu'))
    save_artifact(artifact, tmp_path)
    assert QuantizedWeights(tmp_path).names == ['q_proj.weight']


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('affine', [False, True])
def test_replacement_device_move_and_release(tmp_path, affine):
    from qslab.models.quant_linear import QuantLinear
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(1024, 256, bias=False)).half()
    source = weakref.ref(model[0].weight)
    weight = model[0].weight.detach().float()
    q, transform = awq_affine(weight, torch.ones(1024), group_size=128) if affine else rtn(weight, 128)
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
