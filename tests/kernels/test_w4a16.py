"""Packed W4 arithmetic against FP32 dequantization, including device lifetime."""
import gc
import weakref

import pytest
import torch
from torch import nn

from qslab.compression.algorithms import AWQAffine, RTN
from qslab.compression.checkpoint import save_checkpoint
from qslab.compression.schema import CompressionArtifact
from qslab.quantization import QuantizationMethod as Method
from qslab.runtime.quantization import load_checkpoint_weights
from safetensors.torch import save_file


def save_source(model, path):
    path.mkdir()
    save_file(model.state_dict(), str(path / 'model.safetensors'))
    (path / 'config.json').write_text('{}')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend,affine', [('gemlite', False), ('gemlite', True)])
def test_replacement_device_move_and_release(tmp_path, affine, backend):
    from qslab.runtime.backends.base import QuantLinear
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
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    assert load_checkpoint_weights(model, output_path, backend=backend) == 1
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
