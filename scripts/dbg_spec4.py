"""Debug: trace round 2 of speculative generation (gamma=4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.spec.verify import SpeculativeEngine
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("models/Qwen3-1.7B")
ids = tok.encode(" The capital of France is")

spec = SpeculativeEngine(
    EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=64),
    EngineConfig(model_path="models/Qwen3-0.6B", device="cuda:0", max_new_tokens=64),
    gamma=4)

ref = QslabEngine(EngineConfig(model_path="models/Qwen3-1.7B", device="cuda:0", max_new_tokens=64))
out_ref = ref.generate(ids, max_new_tokens=64)

# manual replay of 2 rounds
spec.target.reset_cache(); spec.draft.reset_cache()
t_logits = spec.target.prefill(ids)
d_logits = spec.draft.prefill(ids)
prev_pred = int(t_logits.argmax(-1))
gen = []
print("round1 prev_pred:", prev_pred)

for rnd in range(2):
    # proposals
    proposal = []
    d_last = d_logits
    for i in range(4):
        d_tok = int(d_last.argmax(-1))
        proposal.append(d_tok)
        d_last = spec.draft.decode_step(d_tok, start_pos=len(ids) + len(gen) + i)
    print(f"round{rnd+1} proposal:", proposal, "| draft cache lens:", [c.len for c in spec.draft.kv_caches][:2])
    pos0 = len(ids) + len(gen)
    t_out = spec.target.model(
        input_ids=torch.tensor([proposal], device="cuda:0"),
        position_ids=torch.arange(pos0, pos0 + 4, device="cuda:0")[None],
        cache_position=torch.arange(pos0, pos0 + 4, device="cuda:0"),
        use_cache=False)
    a = t_out.logits[0].argmax(-1).tolist()
    print(f"round{rnd+1} target argmax:", a, "| target cache lens:", [c.len for c in spec.target.kv_caches][:2])
    print(f"        ref next4 from pos {pos0}:", out_ref[pos0 - len(ids):pos0 - len(ids) + 4])
    # verify
    accept = 0
    pred = prev_pred
    for i, g in enumerate(proposal):
        if pred != g: break
        accept += 1
        pred = a[i]
    bonus = a[-1] if accept == 4 else a[accept]
    print(f"round{rnd+1} accept={accept} bonus={bonus} | emitted:", proposal[:accept] + [bonus])
    gen_base = len(ids) + len(gen)
    for c in spec.target.kv_caches: c.len = gen_base + accept
    for c in spec.draft.kv_caches: c.len = gen_base + accept
    t_commit = spec.target.decode_step(bonus, start_pos=gen_base + accept)
    d_logits = spec.draft.decode_step(bonus, start_pos=gen_base + accept)
    prev_pred = int(t_commit.argmax(-1))
    print(f"round{rnd+1} commit: prev_pred={prev_pred} | ref says next should be",
          out_ref[pos0 - len(ids) + accept + 1 - 1 + 1 - 1] if pos0 - len(ids) + accept < len(out_ref) else "?")
    gen.extend(proposal[:accept] + [bonus])
print("gen:", gen)
print("ref:", out_ref[:len(gen)])
