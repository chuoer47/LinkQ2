"""Export a portable compression artifact to the qslab W4 runtime format."""
from __future__ import annotations

import json
from pathlib import Path

from qslab.artifacts.schema import CompressionArtifact
from qslab.conversion.qslab_w4 import pack_quantized_w4, save_qslab_w4


def export_qslab_w4(artifact: CompressionArtifact, output: str | Path) -> None:
    artifact.validate()
    packed = {}
    group_sizes = set()
    for name, weight in artifact.weights.items():
        group_sizes.add(weight.group_size)
        packed[name] = pack_quantized_w4(weight)
    if len(group_sizes) != 1:
        raise ValueError("qslab W4 format requires one group_size for all layers")
    save_qslab_w4(Path(output), packed, artifact.model, artifact.algorithm,
                  group_sizes.pop(), artifact.calibration, artifact.skipped_layers)
    awq_scales = artifact.transforms.get("input_scale", {})
    if awq_scales:
        (Path(output) / "awq_scales.json").write_text(json.dumps(
            {name: scale.tolist() for name, scale in awq_scales.items()}))
