"""Export a portable artifact to the Nunchaku AWQ GEMV format."""
from __future__ import annotations

from pathlib import Path

from qslab.artifacts.schema import CompressionArtifact
from qslab.conversion.nunchaku_awq.format import pack_nunchaku_awq, save_nunchaku_awq


def export_nunchaku_awq(artifact: CompressionArtifact, output: str | Path) -> None:
    artifact.validate()
    packed = {}
    for name, weight in artifact.weights.items():
        if weight.symmetric:
            raise ValueError("Nunchaku AWQ export requires affine unsigned W4 weights")
        if weight.group_size != 64:
            raise ValueError("Nunchaku AWQ GEMV requires group_size=64")
        packed[name] = pack_nunchaku_awq(weight.values, weight.scale,
                                         weight.zero_point)
    scales = artifact.transforms.get("input_scale", {})
    save_nunchaku_awq(Path(output), packed, artifact.model, artifact.algorithm,
                      artifact.calibration, artifact.skipped_layers, scales)
