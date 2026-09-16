"""Debug: target full-forward vs step-by-step decode divergence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine

cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=12)
eng = QslabEngine(cfg)
ids = [12095]  # actually test on a tiny prefix first

# prompt = 5 tokens of " The capital of France is"
tok_ids = [794, 7815, 315, 9856, 374]  # placeholder; real test below uses engine encode-free

from adapters.tokenizer import QwenTokenizerAdapter
tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
prompt = tok.encode(" The capital of France is")
print("prompt ids:", prompt)

# A) one full forward over prompt + 2 extra tokens
eng.reset_cache()
x = torch.tensor([prompt + [12095, 13]], device="cuda:0")
out_a = eng.model(input_ids=x, use_cache=False).logits
a_last2 = out_a[0, len(prompt) - 1:len(prompt) + 1].argmax(-1).tolist()
print("full-fwd argmax at pos4,5:", a_last2)

# B) prefill prompt then decode 2 steps
eng.reset_cache()
lg = eng.prefill(prompt)
b0 = int(lg.argmax(-1))
lg = eng.decode_step(12095, start_pos=len(prompt))
b1 = int(lg.argmax(-1))
print("stepwise argmax:", [b0, b1])
