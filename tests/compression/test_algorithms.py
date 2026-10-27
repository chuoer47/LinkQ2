"""Small numerical checks for the offline quantizers and input transforms."""
import pytest
import torch

from qslab.compression.algorithms import QUANTIZERS, RTN, RTNClip


def restored(q, transform=None):
    shape = (*q.scale.shape, q.group_size)
    weight = ((q.values.float().reshape(shape) - q.zero_point.float().unsqueeze(-1))
              * q.scale.float().unsqueeze(-1)).reshape_as(q.values)
    return weight if transform is None else weight / transform.float()


def test_rtn_uses_independent_group_scales():
    weight = torch.tensor([[-7., -3., 0., 7., -14., -6., 0., 14.]])
    q, transform = RTN().quantize(weight, group_size=4)
    assert transform is None
    assert q.values.tolist() == [[-7, -3, 0, 7, -7, -3, 0, 7]]
    assert q.scale.tolist() == [[1., 2.]]
    torch.testing.assert_close(restored(q), weight, rtol=0, atol=0)


def test_clipping_does_not_increase_group_reconstruction_error():
    weight = torch.linspace(-1, 1, 128).repeat(2, 1)
    weight[0, 0] = 6
    weight[1, 64] = -6
    baseline, _ = RTN().quantize(weight, group_size=64)
    clipped, _ = RTNClip().quantize(weight, group_size=64)
    before = (restored(baseline) - weight).square().reshape(2, 2, 64).sum(-1)
    after = (restored(clipped) - weight).square().reshape(2, 2, 64).sum(-1)
    # The saved FP16 scales can add a small rounding difference to the search.
    assert torch.all(after <= before * 1.005 + 1e-5)


@pytest.mark.parametrize('method', QUANTIZERS)
def test_quantized_linear_preserves_the_scaled_input_contract(method):
    torch.manual_seed(23)
    weight = torch.randn(16, 128)
    original = weight.clone()
    activation = torch.logspace(-1, 1, 128)
    x = torch.randn(64, 128) * activation
    quantizer = QUANTIZERS[method]()
    q, transform = quantizer.quantize(weight, group_size=64,
        activation_absmean=activation if quantizer.requires_calibration else None)
    if transform is not None:
        assert transform.shape == (128,) and torch.isfinite(transform).all()
        assert (transform > 0).all()
    packed_weight = restored(q)
    actual = (x if transform is None else x / transform.float()) @ packed_weight.t()
    reference = x @ original.t()
    assert (actual - reference).norm() / reference.norm() < 0.15
    torch.testing.assert_close(actual, x @ restored(q, transform).t(), rtol=2e-5, atol=2e-5)
    assert torch.equal(weight, original)
