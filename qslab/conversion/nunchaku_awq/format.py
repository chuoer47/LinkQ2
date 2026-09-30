"""Nunchaku AWQ GEMV weight packing and artifact I/O."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FORMAT_ID = "nunchaku_awq_gemv_v1"


def pack_nunchaku_awq(values: torch.Tensor, scale: torch.Tensor,
                      zero_point: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack affine unsigned W4 into the layout consumed by Nunchaku AWQ GEMV."""
    out_features, in_features = values.shape
    if out_features % 4 or in_features % 64:
        raise ValueError("Nunchaku AWQ GEMV requires N divisible by 4 and K by 64")
    expected_params = (out_features, in_features // 64)
    if tuple(scale.shape) != expected_params or tuple(zero_point.shape) != expected_params:
        raise ValueError(f"scale and zero_point must have shape {expected_params}")
    if values.numel() and (values.min() < 0 or values.max() > 15):
        raise ValueError("Nunchaku AWQ expects unsigned W4 codes in [0, 15]")

    # Interleave four output rows into each 16-bit value, matching TinyChat's
    # AWQ GEMV weight layout. Keep the same representation on CPU and CUDA.
    codes = values.to(torch.int32).reshape(-1, 4, 8)
    packed = (codes[:, 0] | (codes[:, 1] << 4)
              | (codes[:, 2] << 8) | (codes[:, 3] << 12))
    qweight = (packed.reshape(out_features // 4, 4, in_features // 64, 16)
               .permute(0, 2, 1, 3).reshape(out_features // 4, in_features)
               .to(torch.int16).contiguous())

    scales = scale.to(torch.float16).transpose(0, 1).contiguous()
    # gemv_awq computes code * scale + scaled_zero for unsigned W4 codes.
    zeros = (-zero_point.to(torch.float32) * scale.to(torch.float32))
    zeros = zeros.to(torch.float16).transpose(0, 1).contiguous()
    return qweight, scales, zeros


def save_nunchaku_awq(path: Path, tensors: dict[str, tuple[torch.Tensor, ...]],
                      model: dict, algorithm: str, calibration: dict,
                      skipped: list[str], transforms: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, (qweight, scales, zeros) in tensors.items():
        state[f"{name}.qweight"] = qweight.cpu()
        state[f"{name}.scales"] = scales.cpu()
        state[f"{name}.zeros"] = zeros.cpu()
    for name, tensor in transforms.items():
        state[f"{name}.act_scale"] = tensor.cpu()
    save_file(state, str(path / "tensors.safetensors"))
    meta = {"format_id": FORMAT_ID, "algorithm": algorithm,
            "group_size": 64, "model": model,
            "quantized_layers": sorted(tensors), "skipped_layers": skipped,
            "calibration": calibration}
    (path / "config.json").write_text(json.dumps(meta, indent=2))


def load_nunchaku_awq(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    meta = json.loads((path / "config.json").read_text())
    if meta.get("format_id") != FORMAT_ID:
        raise ValueError(f"unsupported Nunchaku format: {meta.get('format_id')}")
    return meta, load_file(str(path / "tensors.safetensors"))
