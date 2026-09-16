"""L3: SpeculationMode — pluggable proposal strategy for speculative decoding.

The verification / rollback / cache-lockstep machinery in verify.py is mode
agnostic; only *where proposals come from* and *how gamma is chosen* differ.
Those two concerns are extracted here (R2), replacing the scattered
``if self.mode == "lookahead"`` branches.

Modes:
  chained   — draft model autoregressively proposes (M3 baseline)
  lookahead — n-gram self-proposal from prompt+generated text (M6, 1.44x)
  dynamic   — chained with adaptive gamma tracking the recent acceptance
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qslab.registry import Registry

SPEC_MODES = Registry("speculation mode")


@dataclass
class ProposalContext:
    """Everything a mode needs to produce proposals."""
    input_ids: list[int]
    generated: list[int]
    gamma: int
    max_new_tokens: int
    draft: object | None          # QslabEngine or None
    draft_logits: object | None   # last draft distribution
    stats: dict = field(default_factory=dict)


class SpeculationMode:
    """Base: draft-model autoregressive proposal (the chained behavior)."""

    name = "chained"
    needs_draft = True

    def on_start(self, input_ids: list[int]) -> None:
        pass

    def current_gamma(self, default_gamma: int) -> int:
        return default_gamma

    def propose(self, ctx: ProposalContext) -> list[int]:
        proposal: list[int] = []
        d_last = ctx.draft_logits
        for _ in range(min(ctx.gamma, ctx.max_new_tokens - len(ctx.generated))):
            d_tok = int(d_last.argmax(dim=-1))
            proposal.append(d_tok)
            if len(ctx.generated) + len(proposal) >= ctx.max_new_tokens:
                break
            d_last = ctx.draft.decode_step(
                d_tok, start_pos=len(ctx.input_ids) + len(ctx.generated)
                + len(proposal) - 1)
            ctx.stats["draft_steps"] = ctx.stats.get("draft_steps", 0) + 1
        return proposal

    def on_round_end(self, input_ids: list[int], generated: list[int],
                     accept_len: int) -> None:
        pass


@SPEC_MODES.register("chained")
class ChainedMode(SpeculationMode):
    name = "chained"


@SPEC_MODES.register("lookahead")
class LookaheadMode(SpeculationMode):
    """n-gram self-proposal: zero draft-model cost (M6)."""

    name = "lookahead"
    needs_draft = False
    NGRAM_N = 3

    def __init__(self):
        self._index: dict[tuple[int, ...], list[int]] = {}

    def on_start(self, input_ids: list[int]) -> None:
        self._index.clear()
        self._index_span(input_ids)

    def propose(self, ctx: ProposalContext) -> list[int]:
        gamma = ctx.gamma
        context = tuple(ctx.input_ids + ctx.generated)
        n = self.NGRAM_N
        # longest-key-first: try to match the tail with as much context as we indexed
        for take in range(min(gamma, len(context)), 0, -1):
            key = context[-take:]
            cands = self._index.get(key)
            if not cands:
                continue
            proposal = list(cands)
            seq = list(context) + proposal
            # extend by following the chain with the fixed n-gram window
            while len(proposal) < gamma and len(seq) >= n:
                nxt = self._index.get(tuple(seq[-n:]))
                if not nxt:
                    break
                proposal.append(int(nxt[0]))
                seq.append(int(nxt[0]))
            return proposal
        return []

    def on_round_end(self, input_ids: list[int], generated: list[int],
                     accept_len: int) -> None:
        self._index_span(input_ids + generated)

    def _index_span(self, ids: list[int], span: int = 8) -> None:
        n = self.NGRAM_N
        start = max(0, len(ids) - span - n)
        for i in range(start, len(ids) - n):
            key = tuple(int(t) for t in ids[i:i + n])
            if key not in self._index:
                self._index[key] = [int(t) for t in ids[i + n:i + n + 4]]


@SPEC_MODES.register("dynamic")
class DynamicMode(SpeculationMode):
    """Chained proposal with gamma adapted to the recent acceptance trend."""

    name = "dynamic"
    needs_draft = True
    WINDOW = 3

    def __init__(self):
        self._recent: list[int] = []

    def on_start(self, input_ids: list[int]) -> None:
        self._recent.clear()

    def current_gamma(self, default_gamma: int) -> int:
        if not self._recent:
            return default_gamma
        recent = self._recent[-self.WINDOW:]
        avg = sum(recent) / len(recent)
        if avg >= default_gamma - 0.1:      # nearly full acceptance -> grow
            return min(default_gamma + 3, 12)
        if avg <= 1.0:                      # frequent rejects -> shrink
            return max(1, default_gamma // 2)
        return default_gamma

    def on_round_end(self, input_ids: list[int], generated: list[int],
                     accept_len: int) -> None:
        self._recent.append(accept_len)


def build_mode(name: str) -> SpeculationMode:
    """Factory: instantiate a speculation mode by registry name."""
    return SPEC_MODES.get(name)()
