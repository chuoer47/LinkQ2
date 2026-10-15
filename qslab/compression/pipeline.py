"""Offline calibration and weight quantization, independent of execution kernels."""
from pathlib import Path

import torch

from adapters.model_builder import build_hf_causal_lm
from qslab.artifacts.schema import CompressionArtifact
from .algorithms import QUANTIZERS
from .calibration import collect_activations
from .config import CompressionConfig


def quantize_model(model, config: CompressionConfig, calib_ids=None):
    needs_calibration = config.algorithm in ("awq", "awq_affine")
    if needs_calibration and calib_ids is None:
        raise ValueError("activation-aware quantization requires calibration token IDs")
    activations = collect_activations(model, calib_ids, config.device) if needs_calibration else {}
    weights, scales = {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(config.target_suffixes):
            continue
        weight, input_scale = QUANTIZERS[config.algorithm](module.weight.detach(),
            group_size=config.group_size, activation_absmean=activations.get(name))
        if config.artifact_device:
            weight.values = weight.values.to(config.artifact_device)
            weight.scale = weight.scale.to(config.artifact_device)
            weight.zero_point = weight.zero_point.to(config.artifact_device)
            if input_scale is not None:
                input_scale = input_scale.to(config.artifact_device)
        key = name + ".weight"
        weights[key] = weight
        if input_scale is not None:
            scales[key] = input_scale
    return CompressionArtifact(weights, {"input_scale": scales} if scales else {})


def compress_checkpoint(model_path, config: CompressionConfig):
    model = build_hf_causal_lm(model_path, device=config.device)
    calib_ids = None
    if config.algorithm in ("awq", "awq_affine"):
        if not config.calibration_path:
            raise ValueError("provide a calibration token file")
        calib_ids = torch.load(Path(config.calibration_path), map_location="cpu", weights_only=True)["token_ids"]
    return quantize_model(model, config, calib_ids)
