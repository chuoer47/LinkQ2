"""In-place fake quantization for evaluating algorithms without runtime kernels."""
from __future__ import annotations

import torch

from qslab.compression.algorithms import QUANTIZERS
from qslab.compression.calibration import collect_activations

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                   "up_proj", "down_proj")


@torch.no_grad()
def apply_fake_quant(model: torch.nn.Module, algorithm: str, group_size: int,
                     calibration_ids: list[list[int]] | None,
                     device: str) -> int:
    needs_calibration = algorithm in ("awq", "nunchaku_awq")
    activations = None
    if needs_calibration:
        if calibration_ids is None:
            raise ValueError(f"{algorithm} requires calibration token IDs")
        activations = collect_activations(model, calibration_ids, device)

    quantizer = QUANTIZERS[algorithm]
    changed = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(TARGET_SUFFIXES):
            continue
        stat = activations.get(name) if activations is not None else None
        if needs_calibration and stat is None:
            continue
        quantized, act_scale = quantizer(
            module.weight.detach(), group_size=group_size,
            activation_absmean=stat)
        grouped_values = quantized.values.to(torch.float32).reshape(
            module.out_features, -1, group_size)
        scale = quantized.scale.to(module.weight.device, torch.float32).unsqueeze(-1)
        if quantized.symmetric:
            restored = grouped_values * scale
        else:
            zero = quantized.zero_point.to(module.weight.device, torch.float32).unsqueeze(-1)
            restored = (grouped_values - zero) * scale
        restored = restored.reshape_as(module.weight)
        if act_scale is not None:
            restored = restored / act_scale.to(module.weight.device, torch.float32).reshape(1, -1)
        module.weight.copy_(restored.to(module.weight.dtype))
        changed += 1
    if not changed:
        raise RuntimeError(f"algorithm {algorithm} did not quantize any linear layers")
    return changed
