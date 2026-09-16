"""M4-S2 (split B): dequant-reference generation on 8B, compare with A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.model.loader import load_w4_model
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-8B")
prompts = [" The capital of France is", " large language model quantization is"]
out_a = torch.load("results/m4_s2_kernel_tokens.pt", weights_only=False)

ref = load_w4_model("models/Qwen3-8B-qslab-w4-awq")
all_match = True
for p in prompts:
    ids = tok.encode(p)
    x = torch.tensor([ids], device="cuda:0")
    o2 = ref.generate(x, max_new_tokens=32, do_sample=False)
    gen_ref = o2[0][len(ids):].tolist()
    m = out_a[p] == gen_ref
    all_match &= m
    print(f"match={m} | ref: {tok.decode(gen_ref)[:50]!r}", flush=True)
print("8B_KERNEL_PATH_ALL_MATCH:", all_match)
