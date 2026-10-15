"""CPU checks for paired artifacts, HF projection coverage, and PPL accounting."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.fake_quant_quality.evaluate import evaluate_ppl
from benchmarks.real_quant.config import matrix
from benchmarks.real_quant.metrics import cache_inventory, paired_probe, quality, summarize_nll
from benchmarks.real_quant.samples import select_samples
from benchmarks.real_quant.model import install_fake, install_packed, restore_weight
from qslab.artifacts.schema import CompressionArtifact, QuantizedTensor
from qslab.compression.config import CompressionConfig
from qslab.compression.pipeline import quantize_model
from qslab.conversion.export_qslab_w4 import export_qslab_w4
from qslab.conversion.qslab_w4 import unpack_w4


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(256, 256, bias=False, dtype=torch.float16)
        self.lm_head = nn.Linear(256, 8, bias=False, dtype=torch.float16)


class PackedCPU(nn.Module):
    def __init__(self, state, key, group, inputs, outputs):
        super().__init__()
        self.qfp = state[key + ".qfp"]
        self.scale = state[key + ".scale"]
        self.act = state.get(key + ".act_scale")
        self.group, self.inputs = group, inputs

    def forward(self, x):
        w = unpack_w4(self.qfp, self.scale, torch.zeros_like(self.scale),
                      self.inputs, self.group).float()
        if self.act is not None:
            x = x / self.act.float()
        return F.linear(x, w)


def cpu_factory(_backend, state, key, group, inputs, outputs, _device):
    return PackedCPU(state, key, group, inputs, outputs)


def test_logical_export_load_and_projection_replacement(tmp_path):
    torch.manual_seed(7)
    fake, real = ToyModel(), ToyModel()
    original_head = real.lm_head
    artifact = quantize_model(fake, SimpleNamespace(),
        CompressionConfig(algorithm="rtn", device="cpu", artifact_device="cpu"))
    assert len(artifact.weights) == 2
    assert all(weight.values.device.type == "cpu" for weight in artifact.weights.values())
    export_qslab_w4(artifact, tmp_path)
    assert install_fake(fake, artifact) == 2
    count, layers = install_packed(real, tmp_path, "qslab_w4a16", "cpu", factory=cpu_factory)
    assert count == 2 and len(layers) == 2
    assert real.lm_head is original_head
    assert not hasattr(real.q_proj, "weight")
    x = torch.randn(2, 3, 256)
    for name in ("q_proj", "k_proj"):
        expected = F.linear(x, fake.get_submodule(name).weight.float())
        assert torch.allclose(real.get_submodule(name)(x), expected, atol=1e-5)


def test_missing_target_is_rejected_before_replacement(tmp_path):
    model = ToyModel()
    artifact = quantize_model(model, SimpleNamespace(),
        CompressionConfig(algorithm="rtn", device="cpu", artifact_device="cpu"))
    del artifact.weights["k_proj.weight"]
    export_qslab_w4(artifact, tmp_path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        install_packed(model, tmp_path, "qslab_w4a16", "cpu", factory=cpu_factory)
    assert isinstance(model.q_proj, nn.Linear)


def test_awq_scale_is_preserved_through_export_and_load(tmp_path):
    model = ToyModel()
    values = torch.arange(256*256).reshape(256, 256).remainder(15).sub(7).to(torch.int8)
    scale = torch.full((256, 2), 0.125, dtype=torch.float16)
    act = torch.full((256,), 2.0, dtype=torch.float16)
    weights = {name + ".weight": QuantizedTensor(values.clone(), scale.clone(), torch.zeros_like(scale), 128)
               for name in ("q_proj", "k_proj")}
    artifact = CompressionArtifact({}, "awq", weights,
        {"input_scale": {name: act for name in weights}})
    export_qslab_w4(artifact, tmp_path)
    install_packed(model, tmp_path, "marlin", "cpu", factory=cpu_factory)
    x = torch.randn(3, 256)
    expected = F.linear(x, restore_weight(weights["q_proj.weight"], act).float())
    assert torch.allclose(model.q_proj(x), expected, atol=1e-5)


def test_ppl_overlap_accounting_and_progress_callback():
    vector = torch.tensor([0.5, -0.7, 1.2, 0.0, -0.2, 0.8])

    class FixedModel(nn.Module):
        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=vector.expand(*input_ids.shape, 6))

    ids = [i % 6 for i in range(17)]
    progress = []
    row = evaluate_ppl(FixedModel(), ids, "cpu", 6, 2,
                       progress=lambda *args: progress.append(args))
    expected = F.cross_entropy(vector.expand(16, 6), torch.tensor(ids[1:]),
                              reduction="sum").item()
    assert row["tokens"] == 16
    assert abs(row["nll"] - expected) < 5e-6
    assert progress[-1][:3] == (17, 17, 16)


def test_matrix_excludes_unsupported_affine_symmetric_pairs():
    rows = matrix()
    assert len(rows) == 7
    assert ("nunchaku_awq", "nunchaku_awq") in rows
    assert all((algorithm == "nunchaku_awq") == (backend == "nunchaku_awq")
               for algorithm, backend in rows)


def test_long_identical_probe_has_bounded_cosine():
    values = torch.linspace(-10, 10, 200000).reshape(1, 2, -1)
    row = paired_probe(values, values)
    assert row["relative_l2"] == 0
    assert abs(row["cosine_similarity"] - 1) < 1e-12


def test_small_samples_are_deterministic_nonoverlapping_and_reproducible():
    ids = list(range(12000))
    selected = select_samples(ids)
    assert selected == select_samples(ids)
    assert selected["input_tokens"] == 4096
    assert selected["scored_tokens"] == 4088
    assert len(set(t for s in selected["samples"] for t in s["token_ids"])) == 4096
    assert all(s["token_ids"] == ids[s["start"]:s["start"]+512] for s in selected["samples"])
    assert select_samples(ids, seed=12)["token_sha256"] != selected["token_sha256"]


def test_small_ppl_does_not_score_segment_boundaries():
    vector = torch.tensor([0.5, -0.7, 1.2, 0.0])

    class FixedModel(nn.Module):
        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=vector.expand(*input_ids.shape, 4))

    samples = [{"index": 0, "start": 0, "token_ids": [3, 0, 1]},
               {"index": 1, "start": 6, "token_ids": [2, 1, 3, 0]}]
    result = quality(FixedModel(), samples, "cpu")
    targets = torch.tensor([0, 1, 1, 3, 0])
    expected = F.cross_entropy(vector.expand(5, 4), targets, reduction="sum").item()
    assert result["tokens"] == 5
    assert abs(result["nll"] - expected) < 1e-6
    assert result["ppl"] == summarize_nll(result["samples"])["ppl"]


def test_cache_memory_counts_unique_storage_and_dtype():
    tensor = torch.zeros(1, 2, 160, 16, dtype=torch.float16)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=tensor, values=tensor.view_as(tensor))],
                            get_seq_length=lambda: 160)
    result = cache_inventory(cache)
    assert result["storage_bytes"] == tensor.untyped_storage().nbytes()
    assert result["dtypes"] == ["torch.float16"]
    assert result["sequence_length"] == 160
