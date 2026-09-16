"""Debug: find first divergence between spec and target-only greedy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.engine.spec.verify import SpeculativeEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
prompt = " The capital of France is"
ids = tok.encode(prompt)
N = 12

target_cfg = EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0",
                          max_new_tokens=N)
draft_cfg = EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0",
                         max_new_tokens=N)

ref = QslabEngine(target_cfg)
out_ref = ref.generate(ids, max_new_tokens=N)
print("ref :", out_ref)

spec = SpeculativeEngine(target_cfg, draft_cfg, gamma=2)

# manual single round trace
spec.target.reset_cache(); spec.draft.reset_cache()
t_logits = spec.target.prefill(ids)
d_logits = spec.draft.prefill(ids)
t0 = int(t_logits.argmax(-1)); d0 = int(d_logits.argmax(-1))
print("first token: target argmax", t0, "| draft argmax", d0, "| ref[0]", out_ref[0])

# round 1: draft proposes from d_logits
d1 = int(d_logits.argmax(-1))
d_logits2 = spec.draft.decode_step(d1, start_pos=len(ids))
d2 = int(d_logits2.argmax(-1))
proposal = [d1, d2]
t_out = spec.target.model(
    input_ids=torch.tensor([proposal], device="cuda:0"),
    cache_position=torch.arange(len(ids), len(ids) + 2, device="cuda:0"),
    use_cache=False)
ta = t_out.logits[0].argmax(dim=-1)
print("proposal:", proposal, "| target argmaxes:", ta.tolist(), "| ref next2:", out_ref[1:3])
