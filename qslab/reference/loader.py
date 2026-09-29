"""HF reference loading: safetensors weights -> fp16 torch modules."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from qslab.reference.config import ModelConfig


def load_model_config(model_path: Path | str) -> ModelConfig:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    return ModelConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )


def load_reference_model(model_path: Path | str, device: str = "cuda:0") -> torch.nn.Module:
    """Load the HF Qwen3 model in fp16 (delegates to the adapters layer)."""
    # L2 must not import transformers; construction lives in adapters.model_builder.
    from adapters.model_builder import build_hf_causal_lm
    return build_hf_causal_lm(model_path, device=device)


def read_safetensors_state_dict(model_path: Path | str) -> dict[str, torch.Tensor]:
    """Plain safetensors reader (own-format loading will reuse this)."""
    from safetensors import safe_open

    model_path = Path(model_path)
    sd: dict[str, torch.Tensor] = {}
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {model_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    return sd


def load_w4_model(model_path: Path | str, device: str = "cuda:0") -> torch.nn.Module:
    """Load a qslab_w4_v1 directory into the reference model with the weights dequantized
    back to fp16 — the software path the kernel must match."""
    # For algo=awq the packed weights carry W' = W diag(s), so the loader divides each
    #   quantized Linear's input by s.
    # No runtime caller left; kept as the fp16 equivalence oracle.
    import json as _json
    from qslab.quant.packfmt import load_qslab_w4, unpack_w4

    model_path = Path(model_path)
    config, st, _calib = load_qslab_w4(model_path)

    model = load_reference_model(args_model_dir(model_path), device=device)
    group = config["group_size"]
    replaced = 0
    awq_scales = {}
    if config["algo"] == "awq":
        awq_file = model_path / "awq_scales.json"
        if awq_file.exists():
            awq_scales = _json.loads(awq_file.read_text())
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            key = f"{name}.weight"
            if f"{key}.qfp" not in st:
                continue
            qfp, scale, zero = st[f"{key}.qfp"], st[f"{key}.scale"], st[f"{key}.zero"]
            O, I = mod.weight.shape
            w_hat = unpack_w4(qfp, scale, zero, I, group)
            assert w_hat.shape == (O, I), f"shape mismatch {key}: {w_hat.shape} vs {(O, I)}"
            mod.weight.copy_(w_hat.to(mod.weight.dtype).to(mod.weight.device))
            replaced += 1
            if key in awq_scales:
                s = torch.tensor(awq_scales[key], device=mod.weight.device,
                                 dtype=mod.weight.dtype)
                _wrap_input_scale(mod, s)
    if replaced != len(config["quantized_layers"]):
        raise RuntimeError(f"replaced {replaced} != expected {len(config['quantized_layers'])}")
    print(f"W4 load: {replaced} linears dequantized from {model_path.name} (algo={config['algo']})")
    return model


def _wrap_input_scale(linear: torch.nn.Linear, s: torch.Tensor):
    """Make linear compute (x / s) @ W'^T so that y == x @ W^T with W = W'/s."""
    orig_forward = linear.forward

    def forward(x, *a, **kw):
        return orig_forward(x / s, *a, **kw)

    linear.forward = forward


def args_model_dir(model_path: Path) -> Path:
    """Resolve the original HF dir a packed model was built from."""
    origin = model_path.parent / (model_path.name + ".origin.txt")
    if origin.exists():
        return Path(origin.read_text().strip())
    raise FileNotFoundError(
        f"missing {origin}: write the source HF model dir there so the "
        "backbone can be rebuilt (packed format stores weights only)")
