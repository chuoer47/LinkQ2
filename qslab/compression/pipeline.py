"""Offline calibration and weight quantization, independent of execution kernels."""
from pathlib import Path

import torch

from adapters.model_builder import build_hf_causal_lm
from qslab.artifacts.schema import CompressionArtifact
from .algorithms import QUANTIZERS
from .calibration import collect_activations
from .config import CompressionConfig


def quantize_model(model, config: CompressionConfig, calib_ids=None):
    quantizer = QUANTIZERS[config.algorithm]()
    if quantizer.requires_calibration and calib_ids is None:
        raise ValueError("activation-aware quantization requires calibration token IDs")
    activations = collect_activations(model, calib_ids, config.device) if quantizer.requires_calibration else {}
    weights, scales = {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(config.target_suffixes):
            continue
        weight, input_scale = quantizer.quantize(module.weight.detach(),
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
        print(f"quantized {key}", flush=True)
    return CompressionArtifact(weights, {"input_scale": scales} if scales else {})


def compress_checkpoint(model_path, config: CompressionConfig):
    model = build_hf_causal_lm(model_path, device=config.device, dtype='auto')
    print(f"loaded {model_path}: dtype={next(model.parameters()).dtype}", flush=True)
    calib_ids = None
    if QUANTIZERS[config.algorithm].requires_calibration and config.calibration_path:
        calib_ids = torch.load(Path(config.calibration_path), map_location="cpu", weights_only=True)["token_ids"]
    return quantize_model(model, config, calib_ids)
