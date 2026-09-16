"""Debug: compare final hidden states between append and full paths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")
ext = [12095, 13]
cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0")

eng = QslabEngine(cfg)
eng.reset_cache()
eng.prefill(ids)
outA = eng.model(input_ids=torch.tensor([ext], device="cuda:0"),
                 cache_position=torch.arange(5, 7, device="cuda:0"),
                 use_cache=False,
                 output_hidden_states=True)

eng.reset_cache()
outB = eng.model(input_ids=torch.tensor([ids + ext], device="cuda:0"),
                 use_cache=False,
                 output_hidden_states=True)

hA = outA.hidden_states[-1]         # [1, 2, H]
hB = outB.hidden_states[-1][:, 5:7]  # last layer, positions 5-6
print("last hidden diff:", (hA - hB).abs().max().item())
hA0 = outA.hidden_states[0]
hB0 = outB.hidden_states[0][:, 5:7]
print("emb diff (should be 0):", (hA0 - hB0).abs().max().item())
print("A argmax:", outA.logits[0].argmax(-1).tolist())
print("B pos5,6 argmax:", outB.logits[0, 5:7].argmax(-1).tolist())
