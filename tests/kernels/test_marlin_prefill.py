"""Large and tail Marlin batches must remain accurate and repeatable."""
import pytest
import torch
import torch.nn.functional as F

from qslab.compression.algorithms.w4 import rtn
from qslab.conversion.qslab_w4 import pack_quantized_w4
from qslab.models.w4linear import W4Linear


@pytest.mark.gpu
def test_marlin_long_batches_are_repeatable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(2026)
    weight = torch.randn(4096, 4096, device="cuda", dtype=torch.float16) * 0.02
    logical, _ = rtn(weight, 128)
    qfp, scale, _ = pack_quantized_w4(logical)
    layer = W4Linear(qfp.cuda(), scale.cuda(), 128, 4096, 4096,
                     backend="w4.marlin").to("cuda")
    dense = (logical.values.float().reshape(4096, -1, 128)
             * logical.scale.float().unsqueeze(-1)).reshape(4096, 4096)
    for m in (32, 48, 65, 2048):
        x = torch.randn(m, 4096, device="cuda", dtype=torch.float16) * 0.1
        expected = F.linear(x.float(), dense)
        first = layer(x)
        second = layer(x)
        assert (first.float() - expected).norm() / expected.norm() < 5e-3
        assert (second.float() - expected).norm() / expected.norm() < 5e-3
        assert (first.float() - second.float()).norm() / expected.norm() < 5e-3
