"""qslab_w4_v1 packed format reader/writer (docs/design-m1).

Layout per quantized Linear (out_features=O, in_features=I, group=128):
  qfp   [O, I/8]  uint32  — 8 int4 values packed per uint32, LSB-first:
                            value i (in-feature index) lives in nibble (i%8)
  scale [O, I/128] fp16   — per-group scale
  zero  [O, I/128] fp16   — per-group zero point (v1: all zeros, symmetric)

Container: safetensors with keys "<weight_name>.qfp/.scale/.zero" +
config.json + calib.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FORMAT_VERSION = 1


def pack_w4(w: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize fp16 weight [O, I] -> (qfp uint32 [O, I/8], scale fp16 [O, I/g], zero fp16)."""
    O, I = w.shape
    assert I % group_size == 0, f"in_features {I} not divisible by group {group_size}"
    assert I % 8 == 0, f"in_features {I} not divisible by 8"
    w16 = w.to(torch.float16)
    wg = w16.view(O, I // group_size, group_size)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 7.0).clamp_min(1e-12)                      # [O, I/g, 1]
    q = torch.clamp(torch.round(wg / scale), -8, 7).to(torch.int8)  # symmetric
    zero = torch.zeros_like(scale.squeeze(-1))
    # pack: [O, I/g, g] int8 -> [O, I] int4 nibbles -> uint32 words
    qn = q.view(O, I).to(torch.uint8) & 0xF
    qfp = torch.zeros(O, I // 8, dtype=torch.int64, device=w16.device)
    for nib in range(8):
        qfp |= qn[:, nib::8].to(torch.int64) << (4 * nib)
    qfp = qfp.to(torch.int32).view(torch.uint32).cpu()  # store on cpu
    return qfp, scale.squeeze(-1).to(torch.float16).cpu(), zero.to(torch.float16).cpu()


def unpack_w4(qfp: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
              in_features: int, group_size: int = 128) -> torch.Tensor:
    """(qfp, scale, zero) -> dequantized fp16 weight [O, I]."""
    O = qfp.shape[0]
    qfp_i64 = qfp.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    qn = torch.zeros(O, in_features, dtype=torch.uint8)
    for nib in range(8):
        qn[:, nib::8] = ((qfp_i64 >> (4 * nib)) & 0xF).to(torch.uint8)
    q = qn.view(O, in_features // group_size, group_size).to(torch.int16)
    q = torch.where(q >= 8, q - 16, q)                          # sign-extend nibble
    w = q.to(torch.float16) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return w.view(O, in_features).to(torch.float16)


def save_qslab_w4(out_dir: Path, tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                  model_config: dict, algo: str, group_size: int,
                  calib_meta: dict, skipped: list[str]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st: dict[str, torch.Tensor] = {}
    for name, (qfp, scale, zero) in tensors.items():
        st[f"{name}.qfp"] = qfp
        st[f"{name}.scale"] = scale
        st[f"{name}.zero"] = zero
    save_file(st, str(out_dir / "tensors.safetensors"))
    config = {
        "format_version": FORMAT_VERSION,
        "algo": algo,
        "group_size": group_size,
        "symmetric": True,
        "quantized_layers": sorted(tensors.keys()),
        "skipped_layers": skipped,
        "model_config": model_config,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "calib.json").write_text(json.dumps(calib_meta, indent=2))


def load_qslab_w4(model_dir: Path) -> tuple[dict, dict[str, torch.Tensor], dict]:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {config.get('format_version')}")
    st = load_file(str(model_dir / "tensors.safetensors"))
    calib = json.loads((model_dir / "calib.json").read_text())
    return config, st, calib
