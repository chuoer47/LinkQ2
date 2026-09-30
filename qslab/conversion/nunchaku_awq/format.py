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
    """Pack affine unsigned W4 into Nunchaku's signed-nibble AWQ GEMV layout."""
    out_features, in_features = values.shape
    if out_features % 4 or in_features % 8:
        raise ValueError("Nunchaku AWQ GEMV requires output divisible by 4 and input by 8")
    codes = values.to(torch.int16) - 8
    packed = torch.zeros((out_features, in_features // 8), dtype=torch.int64,
                         device=values.device)
    for nibble in range(8):
        packed |= (codes[:, nibble::8].to(torch.int64) & 0xF) << (4 * nibble)
    qweight = packed.to(torch.int32).reshape(out_features // 4, in_features // 2)
    scales = scale.to(torch.float16)
    # Kernel computes signed_code * scale + scaled_zero.
    zeros = ((8 - zero_point.to(torch.float32)) * scales.to(torch.float32))
    zeros = zeros.to(torch.float16).transpose(0, 1).contiguous()
    return qweight.cpu(), scales.transpose(0, 1).contiguous().cpu(), zeros.cpu()


def save_nunchaku_awq(path: Path, tensors: dict[str, tuple[torch.Tensor, ...]],
                      model: dict, algorithm: str, calibration: dict,
                      skipped: list[str], transforms: dict[str, torch.Tensor]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, (qweight, scales, zeros) in tensors.items():
        state[f"{name}.qweight"] = qweight
        state[f"{name}.scales"] = scales
        state[f"{name}.zeros"] = zeros
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
