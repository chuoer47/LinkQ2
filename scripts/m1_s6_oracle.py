"""M1-S6: W4 path oracle alignment.

The engine's W4 load path (load_w4_model: unpack -> weight copy -> engine
generate) must produce token-identical output to a direct dequantize +
transformers-style reference forward on the same dequantized weights.
Both use greedy decode; both run on the same GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.model.loader import load_w4_model
from adapters.tokenizer import QwenTokenizerAdapter
from quantizer.packfmt import load_qslab_w4, unpack_w4

PACKED = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-qslab-w4-awq2"

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
prompts = [" The capital of France is", " large language model quantization is"]
n_new = 32

# 1) engine W4 path
cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=n_new)
eng = QslabEngine(cfg)
# re-point engine to packed weights: swap in dequantized weights
from qslab.model.loader import _wrap_input_scale  # noqa
import json as _json
config, st, _ = load_qslab_w4(PACKED)
group = config["group_size"]
awq_scales = {}
if config["algo"] == "awq":
    awq_scales = _json.loads((Path(PACKED) / "awq_scales.json").read_text())
with torch.no_grad():
    for name, mod in eng.model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if f"{key}.qfp" not in st:
            continue
        qfp, scale, zero = st[f"{key}.qfp"], st[f"{key}.scale"], st[f"{key}.zero"]
        w_hat = unpack_w4(qfp, scale, zero, mod.weight.shape[1], group)
        mod.weight.copy_(w_hat.to(mod.weight.dtype))
        if key in awq_scales:
            s = torch.tensor(awq_scales[key], device=mod.weight.device, dtype=mod.weight.dtype)
            _wrap_input_scale(mod, s)
engine_out = {}
for p in prompts:
    ids = tok.encode(p)
    engine_out[p] = eng.generate(ids, max_new_tokens=n_new)
del eng
torch.cuda.empty_cache()

# 2) direct reference on identical dequantized weights (load_w4_model path)
m = load_w4_model(PACKED)
ref_out = {}
for p in prompts:
    ids = tok.encode(p)
    x = torch.tensor([ids], device="cuda:0")
    with torch.inference_mode():
        out = m.generate(x, max_new_tokens=n_new, do_sample=False)
    ref_out[p] = out[0][len(ids):].tolist()

all_match = True
for p in prompts:
    match = engine_out[p] == ref_out[p]
    all_match &= match
    print(f"match={match} | {tok.decode(engine_out[p])[:50]!r}")
print("W4_ORACLE_ALL_MATCH:", all_match)
