"""Build the frozen calibration set (C4, 128 samples x 2048 tokens) once."""
# Store token ids once: every quantization and KV calibration must reuse them.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

OUT = Path("results/frozen/calib_c4_128x2048.pt")
N_SAMPLES = 128
SEQ_LEN = 2048

tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

samples = []
buf = []
for row in ds:
    buf.extend(tok(row["text"]).input_ids)
    while len(buf) >= SEQ_LEN:
        samples.append(buf[:SEQ_LEN])
        buf = buf[SEQ_LEN:]
        if len(samples) >= N_SAMPLES:
            break
    if len(samples) >= N_SAMPLES:
        break

assert len(samples) == N_SAMPLES, f"only got {len(samples)} samples"
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"dataset": "c4/en/train", "n": N_SAMPLES, "seq_len": SEQ_LEN,
            "token_ids": samples}, OUT)
print(f"frozen calib saved: {OUT} ({N_SAMPLES} x {SEQ_LEN})")
