"""Debug: does explicit position_ids fix the continuation forward?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")

cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0")
eng = QslabEngine(cfg)
eng.reset_cache()
lg = eng.prefill(ids)
print("prefill argmax:", int(lg.argmax(-1)))  # expect 12095

# continuation forward WITH explicit position_ids
pos0 = len(ids)
out = eng.model(
    input_ids=torch.tensor([[12095, 13]], device="cuda:0"),
    position_ids=torch.arange(pos0, pos0 + 2, device="cuda:0")[None],
    cache_position=torch.arange(pos0, pos0 + 2, device="cuda:0"),
    use_cache=False)
print("with position_ids:", out.logits[0].argmax(-1).tolist())   # expect [13, 576]

# without position_ids (the broken path)
eng.reset_cache()
eng.prefill(ids)
out2 = eng.model(
    input_ids=torch.tensor([[12095, 13]], device="cuda:0"),
    cache_position=torch.arange(pos0, pos0 + 2, device="cuda:0"),
    use_cache=False)
print("without position_ids:", out2.logits[0].argmax(-1).tolist())
