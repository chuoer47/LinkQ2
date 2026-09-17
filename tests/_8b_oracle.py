"""Compute the greedy HF reference for the 8B acceptance test.

Runs as a subprocess so the fp16 model's ~16 GB is fully returned to the
driver before the engine is built — an in-process oracle leaves the card
fragmented and the paged runtime cannot allocate.
"""
import json
import sys

sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")

import torch

from qslab.models.loader import load_reference_model
from adapters.tokenizer import QwenTokenizerAdapter

MODEL = "models/Qwen3-8B"
PROMPT = " The capital of France is"

if __name__ == "__main__":
    max_tokens = int(sys.argv[1])
    out_path = sys.argv[2]
    tok = QwenTokenizerAdapter(MODEL)
    ids = tok.encode(PROMPT)
    ref = load_reference_model(MODEL)
    with torch.inference_mode():
        out = ref.generate(torch.tensor([ids], device="cuda"),
                           max_new_tokens=max_tokens, do_sample=False)
    json.dump({"prompt_ids": ids, "tokens": out[0][len(ids):].tolist()},
              open(out_path, "w"))
    print("wrote", out_path)
