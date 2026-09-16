"""Debug: trace target cache update positions inside spec round 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine.spec.verify import SpeculativeEngine
from qslab.quant.cache.kv_cache import BaseKVCache

orig_update = BaseKVCache.update
TRACE = []


def traced_update(self, k_new, v_new, start=None):
    st = self.len if start is None else start
    TRACE.append((st, k_new.shape[2]))
    return orig_update(self, k_new, v_new, start)


BaseKVCache.update = traced_update

spec = SpeculativeEngine(
    EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=12),
    EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0", max_new_tokens=12),
    gamma=2)

from adapters.tokenizer import QwenTokenizerAdapter
tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")

spec.target.reset_cache()
spec.draft.reset_cache()
TRACE.clear()
t_logits = spec.target.prefill(ids)
print("after target prefill: cache lens", [c.len for c in spec.target.kv_caches][:3], "| trace head:", TRACE[:2])
TRACE.clear()
d_logits = spec.draft.prefill(ids)
print("after draft prefill: draft lens", [c.len for c in spec.draft.kv_caches][:3])
TRACE.clear()

d1 = int(d_logits.argmax(-1))
dl2 = spec.draft.decode_step(d1, start_pos=len(ids))
d2 = int(dl2.argmax(-1))
print("proposal:", [d1, d2], "draft trace:", TRACE[:4])
TRACE.clear()

t_out = spec.target.model(
    input_ids=torch.tensor([[d1, d2]], device="cuda:0"),
    cache_position=torch.arange(len(ids), len(ids) + 2, device="cuda:0"),
    use_cache=False)
print("target trace (should be (5,2)):", TRACE[:4])
print("t_argmax:", t_out.logits[0].argmax(-1).tolist())

# reference continuation via pure engine
from qslab.engine import QslabEngine
ref = QslabEngine(EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0"))
o = ref.generate(ids, max_new_tokens=4)
print("ref next4:", o)
