"""Offline quantization driver: HF checkpoint -> qslab_w4_v1 directory.

Usage (inside qslab env, repo root):
  python -m quantizer.quantize --model models/Qwen3-1.7B --algo rtn \
      --out models/Qwen3-1.7B-qslab-w4-rtn [--awq] [--calib results/frozen/calib_c4_128x2048.pt]

Quantized: q/k/v/o_proj, gate/up/down_proj of every decoder layer.
Skipped: embed_tokens, lm_head, all norms, rotary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.reference.loader import load_reference_model, load_model_config
from qslab.quant.packfmt import save_qslab_w4
from qslab.quant.w4 import quantize_weight

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def module_name_to_weight(name: str) -> str:
    return f"{name}.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", choices=["rtn", "rtn_clip", "awq"], default="rtn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="results/frozen/calib_c4_128x2048.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    device = args.device
    model = load_reference_model(args.model, device=device)
    model_cfg = load_model_config(args.model)

    calib_meta = {"dataset": None, "hash": None, "n_samples": 0, "seq_len": 0}
    act_stats = None
    if args.algo == "awq":
        calib_path = Path(args.calib)
        assert calib_path.exists(), (
            f"frozen calib missing: {calib_path}\n"
            "generate it first: python adapters/download.py --build-calib")
        blob = torch.load(calib_path, weights_only=False)
        calib_ids = blob["token_ids"]
        calib_meta = {"dataset": blob.get("dataset", "c4"),
                      "hash": hashlib.sha256(calib_path.read_bytes()).hexdigest()[:16],
                      "n_samples": len(calib_ids), "seq_len": len(calib_ids[0])}
        from qslab.quant.calibrate import collect_activations
        act_stats = collect_activations(model, calib_ids, device)
        print(f"calibration stats for {len(act_stats)} modules")

    tensors: dict[str, tuple] = {}
    awq_scales: dict[str, list] = {}
    skipped = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not name.endswith(TARGET_SUFFIXES):
            skipped.append(name)
            continue
        w = mod.weight.detach()
        x_absmean = None
        if act_stats is not None:
            x_absmean = act_stats.get(name)
        packed, s = quantize_weight(w, algo=args.algo, x_absmean=x_absmean,
                                    group_size=args.group_size)
        tensors[module_name_to_weight(name)] = packed
        if s is not None:
            awq_scales[module_name_to_weight(name)] = s.tolist()
        print(f"quantized {name}: {tuple(w.shape)}")

    model_config_dict = {
        "hidden_size": model_cfg.hidden_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "num_key_value_heads": model_cfg.num_key_value_heads,
        "head_dim": model_cfg.head_dim,
        "intermediate_size": model_cfg.intermediate_size,
        "vocab_size": model_cfg.vocab_size,
        "rms_norm_eps": model_cfg.rms_norm_eps,
        "rope_theta": model_cfg.rope_theta,
        "max_position_embeddings": model_cfg.max_position_embeddings,
        "tie_word_embeddings": model_cfg.tie_word_embeddings,
    }
    save_qslab_w4(Path(args.out), tensors, model_config_dict,
                  algo=args.algo, group_size=args.group_size,
                  calib_meta=calib_meta, skipped=skipped)
    if awq_scales:
        (Path(args.out) / "awq_scales.json").write_text(json.dumps(awq_scales))
    print(f"saved qslab_w4_v1 -> {args.out} ({len(tensors)} layers, algo={args.algo})")


if __name__ == "__main__":
    main()
