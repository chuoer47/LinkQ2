"""M1a correctness gate: W4 (dequant path) vs FP16 reference on PPL.

Usage: python scripts/m1a_ppl_check.py [packed_model_dir]
       default packed dir = models/Qwen3-1.7B-qslab-w4-rtn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

packed_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-1.7B-qslab-w4-rtn"

from qslab.models.loader import load_w4_model, load_reference_model
from adapters.tokenizer import QwenTokenizerAdapter
from benchmarks.bench_ppl import ppl_of_model

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [t for t in ds["text"] if len(t.strip()) > 200][:16]

m4 = load_w4_model(packed_dir)
ppl4 = ppl_of_model(m4, tok, texts, "cuda:0")
print(f"W4 PPL [{packed_dir}]: {ppl4:.4f}")
del m4
torch.cuda.empty_cache()

mref = load_reference_model("models/Qwen3-1.7B", device="cuda:0")
pplr = ppl_of_model(mref, tok, texts, "cuda:0")
print(f"FP16   PPL: {pplr:.4f}")
print(f"degradation: {ppl4 - pplr:+.4f}")
