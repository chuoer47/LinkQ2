"""Speculative decoding: draft wrapper + greedy verify + rollback.

M3 main path is GREEDY speculation (matches the engine's greedy decode).
Verification alignment (the subtle part):

  target cache always contains exactly `generated` tokens.
  round:  draft proposes g_1..g_gamma (from its distribution after `generated`)
          target forward feeds [g_1..g_gamma] -> argmax a_1..a_gamma where
          a_i = target's prediction AFTER seeing g_1..g_i.
  Therefore a_i verifies g_{i+1} (the NEXT proposal), and a_gamma verifies
  the bonus. The prediction that verifies g_1 came from the previous round
  (or the initial prefill) and is carried in `self._prev_target_pred`.

  accepted = longest prefix of proposals with g_{i+1} == a_i... concretely:
  walk i=1..gamma: token g_i is accepted iff prev_target_pred == g_i;
  after each accept, prev_target_pred = a_i.

  If all gamma accepted: bonus = a_gamma, emitted = gamma+1 tokens.
  Else: bonus = a_{accept_len} (the first mismatched prediction), emitted =
  accept_len + 1 tokens. Both caches roll back to accepted length, then the
  bonus is committed via one real forward (KV must stay in lockstep with
  `generated`).

Losslessness (greedy): every emitted token equals target's argmax at its
position — identical to target-only greedy decoding.
"""
from __future__ import annotations

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine


class SpeculativeEngine:
    """Wraps a target engine and a draft engine; owns the draft/verify loop.

    Modes (M5):
      "chained"  — classic draft-model speculation (M3 behavior, default)
      "lookahead"— self-speculation: proposals come from n-gram matches in
                   the prompt+generated text (Prompt-Lookup style); no draft
                   model needed, great on repetitive/extractive workloads
      "dynamic"  — chained with adaptive gamma: grow gamma while rounds are
                   fully accepted, shrink after rejects (tracks the AR)
    """

    def __init__(self, target_cfg: EngineConfig, draft_cfg: EngineConfig,
                 gamma: int = 4, kv_mode: str = "fp16", kv_plan_path: str | None = None,
                 target_engine: "QslabEngine | None" = None,
                 mode: str = "chained",
                 dynamic_window: int = 3):
        assert gamma >= 1
        self.gamma = gamma
        self.mode = mode
        self.dynamic_window = dynamic_window
        self._recent_accepts: list[int] = []
        self._ngram_index: dict[tuple[int, ...], list[int]] = {}
        self.target = target_engine or QslabEngine(target_cfg, kv_mode=kv_mode,
                                                   kv_plan_path=kv_plan_path)
        self.draft = None
        if mode in ("chained", "dynamic"):
            self.draft = QslabEngine(draft_cfg, kv_mode="fp16")

    # ---- mode helpers ----

    def _current_gamma(self) -> int:
        """dynamic mode: adapt gamma to the recent acceptance trend."""
        if self.mode != "dynamic" or not self._recent_accepts:
            return self.gamma
        recent = self._recent_accepts[-self.dynamic_window:]
        avg = sum(recent) / len(recent)
        if avg >= self.gamma - 0.1:        # nearly full acceptance -> grow
            return min(self.gamma + 3, 12)
        if avg <= 1.0:                      # frequent rejects -> shrink
            return max(1, self.gamma // 2)
        return self.gamma

    def _ngram_lookup(self, key: tuple[int, ...]) -> list[int]:
        return self._ngram_index.get(key, [])

    def _build_ngram(self, ids: list[int], n: int = 3):
        """Index the prompt with n-gram keys (longest-match candidates)."""
        for i in range(len(ids) - n):
            key = tuple(ids[i:i + n])
            self._ngram_index.setdefault(key, ids[i + n:i + n + self.gamma])

    def _index_ngrams(self, ids: list[int], n: int = 3):
        """Incrementally index newly generated tail (last few tokens)."""
        start = max(0, len(ids) - self.gamma - n)
        for i in range(start, len(ids) - n):
            key = tuple(ids[i:i + n])
            if key not in self._ngram_index:
                self._ngram_index[key] = ids[i + n:i + n + self.gamma]

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int,
                 eos_id: int | None = None) -> list[int]:
        self.target.reset_cache()
        if self.draft is not None:
            self.draft.reset_cache()
        stats = {"rounds": 0, "draft_steps": 0, "target_fwds": 0, "accepted": 0}
        generated: list[int] = []
        self._ngram_index.clear()
        self._recent_accepts.clear()

        # target prefills; its argmax verifies the first proposal
        t_logits = self.target.prefill(input_ids)
        stats["target_fwds"] += 1
        prev_target_pred = int(t_logits.argmax(dim=-1))
        d_logits = None
        if self.draft is not None:
            d_logits = self.draft.prefill(input_ids)

        # n-gram index for lookahead mode (prompt terms)
        if self.mode == "lookahead":
            self._build_ngram(input_ids)

        while len(generated) < max_new_tokens:
            if eos_id is not None and prev_target_pred == eos_id:
                break
            gamma = self._current_gamma()

            # 1) proposals (mode-dependent)
            proposal: list[int] = []
            if self.mode == "lookahead":
                context = tuple(input_ids + generated)
                # try longest n-gram match first (up to gamma tokens)
                for n in range(gamma, 0, -1):
                    key = context[-n:] if n <= len(context) else None
                    if key and key in self._ngram_index:
                        cands = self._ngram_index[key]
                        if cands:
                            # follow the lookup chain up to gamma tokens
                            proposal = list(cands)
                            seq = list(context) + proposal
                            for _ in range(gamma - len(proposal)):
                                nxt = self._ngram_lookup(tuple(seq[-n:]))
                                if not nxt:
                                    break
                                proposal.append(nxt)
                                seq.append(nxt)
                            break
            else:
                d_last = d_logits
                for _ in range(min(gamma, max_new_tokens - len(generated))):
                    d_tok = int(d_last.argmax(dim=-1))
                    proposal.append(d_tok)
                    if len(generated) + len(proposal) >= max_new_tokens:
                        break
                    d_last = self.draft.decode_step(
                        d_tok, start_pos=len(input_ids) + len(generated) + len(proposal) - 1)
                    stats["draft_steps"] += 1
            proposal = proposal[:max(1, max_new_tokens - len(generated))]

            # 2) target scores proposals in one forward (cache already holds
            # exactly `generated`; writes land at [len(generated), ...))
            pos0 = len(input_ids) + len(generated)
            t_out = self.target.model(
                input_ids=torch.tensor([proposal], device=self.target.device),
                position_ids=torch.arange(pos0, pos0 + len(proposal),
                                          device=self.target.device)[None],
                cache_position=torch.arange(pos0, pos0 + len(proposal),
                                            device=self.target.device),
                use_cache=False)
            stats["target_fwds"] += 1
            a = t_out.logits[0].argmax(dim=-1).tolist()   # a[i] verifies g_{i+1}

            # 3) verify: g_i accepted iff prev_target_pred == g_i
            accept_len = 0
            pred = prev_target_pred
            for i, g in enumerate(proposal):
                if pred != g:
                    break
                accept_len += 1
                pred = a[i]

            stats["rounds"] += 1
            stats["accepted"] += accept_len

            if accept_len == len(proposal):
                bonus = a[-1]                              # verified by a_gamma
            else:
                # the first rejected prediction IS target's correct next token:
                # accept==0 -> prev_target_pred; accept=k -> a[k-1] (sees g_1..g_k)
                bonus = prev_target_pred if accept_len == 0 else a[accept_len - 1]
            emitted = proposal[:accept_len] + [bonus]

            # 4) roll both caches back to the accepted prefix, then commit the
            # bonus through a real forward (KV stays in lockstep with generated);
            # the commit forward's argmax is the next round's verifier for g_1
            gen_base = len(input_ids) + len(generated)
            for c in self.target.kv_caches:
                c.len = gen_base + accept_len
            if self.draft is not None:
                for c in self.draft.kv_caches:
                    c.len = gen_base + accept_len
            t_commit = self.target.decode_step(bonus, start_pos=gen_base + accept_len)
            if self.draft is not None:
                d_logits = self.draft.decode_step(bonus, start_pos=gen_base + accept_len)
                stats["draft_steps"] += 1

            generated.extend(emitted)
            prev_target_pred = int(t_commit.argmax(dim=-1))

            # update mode state
            if self.mode == "lookahead":
                self._index_ngrams(input_ids + generated)
            elif self.mode == "dynamic":
                self._recent_accepts.append(accept_len)

        self.last_stats = {
            **stats,
            "ar": (stats["accepted"] + stats["rounds"]) / max(stats["rounds"], 1),
            "tokens": len(generated),
        }
        return generated[:max_new_tokens]
