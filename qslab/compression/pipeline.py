"""Orchestrate calibration and algorithm selection for offline compression."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from qslab.artifacts.schema import CompressionArtifact
from qslab.compression.algorithms import QUANTIZERS
from qslab.compression.calibration import collect_activations
from qslab.compression.config import CompressionConfig
from qslab.compression.model.qwen3 import export_model_config, load_qwen3


def quantize_model(model, model_config, config: CompressionConfig,
                   calib_ids: list[list[int]] | None = None,
                   calibration_meta: dict | None = None) -> CompressionArtifact:
    activations = None
    if config.algorithm in ("awq", "nunchaku_awq"):
        if calib_ids is None:
            raise ValueError("AWQ requires calibration token ids")
        activations = collect_activations(model, calib_ids, config.device)

    weights = {}
    scales = {}
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.endswith(config.target_suffixes):
            skipped.append(name)
            continue
        if config.algorithm in ("awq", "nunchaku_awq"):
            stat = activations.get(name) if activations is not None else None
            if stat is None:
                skipped.append(name)
                continue
        else:
            stat = None
        try:
            quantized, input_scale = QUANTIZERS[config.algorithm](
                module.weight.detach(), group_size=config.group_size,
                activation_absmean=stat)
        except KeyError as exc:
            raise ValueError(f"unknown compression algorithm: {config.algorithm}") from exc
        if input_scale is not None:
            scales[f"{name}.weight"] = input_scale
        weights[f"{name}.weight"] = quantized

    return CompressionArtifact(
        model=export_model_config(model_config), algorithm=config.algorithm,
        weights=weights, transforms={"input_scale": scales} if scales else {},
        calibration=calibration_meta or {}, skipped_layers=skipped)


def compress_checkpoint(model_path: str, config: CompressionConfig,
                        calibration_path: str | None = None) -> CompressionArtifact:
    model, model_config = load_qwen3(model_path, config.device)
    calib_meta = {"dataset": None, "hash": None, "n_samples": 0, "seq_len": 0}
    calib_ids = None
    source = calibration_path or config.calibration_path
    if config.algorithm in ("awq", "nunchaku_awq"):
        if not source or not Path(source).exists():
            raise FileNotFoundError(f"{config.algorithm} calibration file not found: {source}")
        path = Path(source)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        calib_ids = blob["token_ids"]
        calib_meta = {"dataset": blob.get("dataset"),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "n_samples": len(calib_ids), "seq_len": len(calib_ids[0])}
    return quantize_model(model, model_config, config, calib_ids, calib_meta)
