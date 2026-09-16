"""Model loading: safetensors weights -> torch modules.

M0 path: build the reference transformers Qwen3 model, then keep its weights
as the oracle. The loader also exposes a plain state_dict reader that later
milestones (W4 packing / own format) will build upon.

NOTE: transformers import is tolerated here during M0 because the model
*definition* itself comes from transformers (decision 01 §1.1). The weights
reader below is the part that stays with qslab long-term.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from qslab.config import ModelConfig


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
    """Load the HF Qwen3 model in fp16 as reference/oracle."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    return model.to(device).eval()


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
    """Load a qslab_w4_v1 directory: rebuild the transformers model, replace
    quantized Linears' weights with dequantized fp16 (M1a software path).

    For algo=awq, packed weights carry W' = W diag(s); activation scaling is
    folded back here: the previous op's output scale must be divided by s.
    We implement this by wrapping each quantized Linear with an input-scaling
    forward (x / s) — exact for the first quantized layer after any op, and
    composed correctly because scaling only applies at quantized-layer entry.
    Returns an eval-mode fp16 model functionally equivalent to the quantized
    checkpoint (this is what the W4A16 kernel path must match in M1b).
    """
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
    """The HF source dir the packed model was built from is stored in
    calib/config indirectly; for 1.7B both live next to each other. The
    packed model itself carries model_config, so rebuild from the ORIGINAL
    HF dir recorded at pack time (convention: <packed_dir>.origin.txt)."""
    origin = model_path.parent / (model_path.name + ".origin.txt")
    if origin.exists():
        return Path(origin.read_text().strip())
    raise FileNotFoundError(
        f"missing {origin}: write the source HF model dir there so the "
        "backbone can be rebuilt (packed format stores weights only)")
