"""Read and write backend independent compression artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from qslab.artifacts.schema import CompressionArtifact, QuantizedTensor


def save_artifact(artifact: CompressionArtifact, path: str | Path) -> None:
    artifact.validate()
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, weight in artifact.weights.items():
        tensors[f"{name}.values"] = weight.values.to(device="cpu", dtype=torch.int8)
        tensors[f"{name}.scale"] = weight.scale.cpu()
        tensors[f"{name}.zero_point"] = weight.zero_point.cpu()
    for group, entries in artifact.transforms.items():
        for name, tensor in entries.items():
            tensors[f"transforms.{group}.{name}"] = tensor.cpu()
    save_file(tensors, str(root / "tensors.safetensors"))
    manifest = {
        "schema_version": artifact.schema_version,
        "model": artifact.model,
        "algorithm": artifact.algorithm,
        "calibration": artifact.calibration,
        "skipped_layers": artifact.skipped_layers,
        "weights": {name: {"group_size": w.group_size, "bits": w.bits,
                            "symmetric": w.symmetric, "qmin": w.qmin,
                            "qmax": w.qmax}
                    for name, w in artifact.weights.items()},
        "transforms": {group: sorted(entries) for group, entries in artifact.transforms.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_artifact(path: str | Path) -> CompressionArtifact:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    tensors = load_file(str(root / "tensors.safetensors"))
    weights = {}
    for name, meta in manifest["weights"].items():
        weights[name] = QuantizedTensor(
            tensors[f"{name}.values"], tensors[f"{name}.scale"],
            tensors[f"{name}.zero_point"], meta["group_size"],
            meta["bits"], meta["symmetric"], meta.get("qmin"), meta.get("qmax"))
    transforms = {}
    for group, names in manifest.get("transforms", {}).items():
        transforms[group] = {name: tensors[f"transforms.{group}.{name}"] for name in names}
    result = CompressionArtifact(manifest["model"], manifest["algorithm"], weights,
                                 transforms, manifest.get("calibration", {}),
                                 manifest.get("skipped_layers", []),
                                 manifest["schema_version"])
    result.validate()
    return result
