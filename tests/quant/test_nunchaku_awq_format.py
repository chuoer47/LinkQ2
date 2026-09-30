from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from qslab.artifacts.schema import CompressionArtifact, QuantizedTensor
from qslab.conversion.nunchaku_awq import export_nunchaku_awq
from qslab.conversion.nunchaku_awq.format import load_nunchaku_awq, pack_nunchaku_awq
from qslab.runtime.backends.nunchaku_awq import NunchakuAWQLinear
from qslab.runtime.model.loader import swap_w4
from qslab.runtime.model.primitives import Linear


def _reference_pack(values: torch.Tensor) -> torch.Tensor:
    n, k = values.shape
    packed = values.to(torch.int32).reshape(-1, 4, 8)
    packed = packed[:, 0] | (packed[:, 1] << 4) | (packed[:, 2] << 8) | (packed[:, 3] << 12)
    return (packed.reshape(n // 4, 4, k // 64, 16)
            .permute(0, 2, 1, 3).reshape(n // 4, k)
            .to(torch.int16).contiguous())


def test_pack_matches_runtime_layout_and_affine_parameters():
    n, k, group_size = 8, 128, 64
    values = torch.arange(n * k, dtype=torch.int32).remainder(16).reshape(n, k)
    scale = torch.arange(n * (k // group_size), dtype=torch.float32).reshape(
        n, k // group_size).add_(1) / 10
    zero = torch.arange(n * (k // group_size), dtype=torch.float32).reshape_as(scale).remainder(16)

    qweight, scales, zeros = pack_nunchaku_awq(values, scale, zero)

    assert torch.equal(qweight, _reference_pack(values))
    assert qweight.shape == (n // 4, k)
    assert qweight.dtype == torch.int16
    assert torch.equal(scales, scale.to(torch.float16).T.contiguous())
    assert torch.equal(zeros, (-zero * scale).to(torch.float16).T.contiguous())


def test_nunchaku_export_and_load_share_runtime_layout(tmp_path):
    n, k = 8, 128
    values = torch.arange(n * k, dtype=torch.uint8).remainder(16).reshape(n, k)
    scale = torch.full((n, k // 64), 0.125, dtype=torch.float16)
    zero = torch.full((n, k // 64), 7, dtype=torch.float16)
    weight = QuantizedTensor(values, scale, zero, group_size=64,
                             symmetric=False, qmin=0, qmax=15)
    artifact = CompressionArtifact(
        model={"architecture": "test"}, algorithm="nunchaku_awq",
        weights={"layer.weight": weight},
        transforms={"input_scale": {"layer.weight": torch.ones(k)}},
    )

    export_nunchaku_awq(artifact, tmp_path)
    metadata, state = load_nunchaku_awq(tmp_path)

    expected_qweight, expected_scales, expected_zeros = pack_nunchaku_awq(
        values, scale, zero)
    assert metadata["format_id"] == "nunchaku_awq_gemv_v1"
    assert metadata["group_size"] == 64
    assert torch.equal(state["layer.weight.qweight"], expected_qweight)
    assert torch.equal(state["layer.weight.scales"], expected_scales)
    assert torch.equal(state["layer.weight.zeros"], expected_zeros)
    assert torch.equal(state["layer.weight.act_scale"], torch.ones(k))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(k, n)

    model = Model()
    assert swap_w4(model, str(tmp_path)) == 1
    assert isinstance(model.layer, NunchakuAWQLinear)
    assert torch.equal(model.layer.qweight, expected_qweight)
    assert torch.equal(model.layer.scales, expected_scales)
    assert torch.equal(model.layer.zeros, expected_zeros)
    assert torch.equal(model.layer.act_scale, torch.ones(k))


def test_runtime_calls_extension_in_supported_row_chunks(monkeypatch):
    calls = []

    def fake_gemv(x, qweight, scales, zeros, m, n, k, group_size):
        calls.append((m, n, k, group_size, x.clone()))
        return x[:, :1].expand(m, n).contiguous()

    fake_extension = ModuleType("nunchaku._C")
    fake_extension.ops = SimpleNamespace(gemv_awq=fake_gemv)
    monkeypatch.setitem(sys.modules, "nunchaku._C", fake_extension)

    layer = NunchakuAWQLinear(
        torch.zeros((2, 64), dtype=torch.int16),
        torch.ones((2, 8), dtype=torch.float16),
        torch.zeros((2, 8), dtype=torch.float16), 64, 64, 8,
        act_scale=torch.full((64,), 2.0, dtype=torch.float16))
    x = torch.ones((10, 64), dtype=torch.float16)
    output = layer(x)

    assert output.shape == (10, 8)
    assert output.dtype == x.dtype
    assert [call[0] for call in calls] == [8, 2]
    assert all(call[1:4] == (8, 64, 64) for call in calls)
    assert torch.equal(calls[0][4], torch.full((8, 64), 0.5, dtype=torch.float16))
