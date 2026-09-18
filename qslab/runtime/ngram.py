"""n-gram speculative proposer (vLLM V1 style, CPU-only).

The proposal is a lookup, not a model: take the last n committed tokens as a
key, find its most recent earlier occurrence in the sequence, and propose the
tokens that followed it. Repetition is common in real traffic (copy/edit
tasks, code, summarisation), so this costs zero GPU memory and zero draft
forward while exploiting exactly that structure. Prompts with no match return
an empty proposal — the engine then pads the verify batch, which is
semantically a plain decode step (design-m9 §2, padding neutrality).

Matching is "last occurrence wins" (the most recent context is the best
predictor), same as vLLM's prompt-lookup. The numpy sliding window is O(L)
zero-copy per step, which is noise next to a model forward.
"""
from __future__ import annotations

import numpy as np


class NGramProposer:

    def __init__(self, ngram_size: int, num_drafts: int):
        self.n = ngram_size
        self.gamma = num_drafts

    def propose(self, token_ids: list[int]) -> list[int]:
        """Return up to gamma candidate continuation tokens for the sequence."""
        L = len(token_ids)
        n, g = self.n, self.gamma
        # need: the key itself, plus at least one known token after an earlier
        # occurrence, plus at least one committed token before the key
        if L <= n:
            return []
        a = np.asarray(token_ids, dtype=np.int64)
        key = a[L - n:L]
        # windows over the whole sequence; drop the trailing n windows so the
        # key can never match itself, and drop the tail so a match has at
        # least one follower inside the committed part
        windows = np.lib.stride_tricks.sliding_window_view(a, n)[:L - n]
        hits = np.nonzero((windows == key).all(axis=1))[0]
        if hits.size == 0:
            return []
        p = int(hits[-1])                      # most recent occurrence
        return a[p + n:p + n + g].tolist()

    def propose_batch(self, seqs) -> list[list[int]]:
        return [self.propose(list(s.token_ids)) for s in seqs]


class LookaheadProposer(NGramProposer):
    """Self-speculation with a persistent index and chain extension (M6 mode).

    Migrated from the frozen ``qslab/engine/spec/modes.py`` LookaheadMode, but
    only the part of it that actually does something:

    * **Chain extension** — after a key hit, the proposal keeps walking the
      index with the fixed n-window instead of stopping at the end of the
      matched span. A single hit's contiguous followers are capped by how far
      they run before the sequence ends, which bites exactly at the large γ
      where speculation pays (M9 measured γ=8 optimal on repetition); chaining
      bridges non-contiguous repeats and is the whole reason to prefer this
      proposer over the plain lookup.
    * **First occurrence wins**, matching M6 (the plain lookup above takes the
      most recent hit — a different bet, kept distinct rather than merged).

    NOT ported: M6's longest-key-first ladder. It tried keys of length γ down
    to 1, but its index only ever holds n-length keys, so every rung except
    ``take == n`` was unreachable — a no-op the old tests never noticed.
    """

    def __init__(self, ngram_size: int, num_drafts: int, span: int = 8):
        super().__init__(ngram_size, num_drafts)
        self.span = span
        # per sequence: a shared index would let one request's proposal come
        # from another request's text, making output depend on batch composition
        self._index: dict[int, dict[tuple[int, ...], list[int]]] = {}
        self._indexed: dict[int, int] = {}       # seq_id -> tokens already indexed

    def drop_seqs(self, seq_ids):
        """Called by the engine when a sequence finishes."""
        for sid in seq_ids:
            self._index.pop(sid, None)
            self._indexed.pop(sid, None)

    def propose(self, token_ids: list[int], seq_id: int | None = None) -> list[int]:
        L, n, g = len(token_ids), self.n, self.gamma
        if L <= n:
            return []
        index = self._index_upto(token_ids, seq_id)
        key = tuple(int(t) for t in token_ids[L - n:L])
        cands = index.get(key)
        if not cands:
            return []
        proposal = [int(t) for t in cands][:g]
        # chain: keep following the window through the index to fill the window
        seq = list(token_ids) + proposal
        while len(proposal) < g:
            nxt = index.get(tuple(seq[-n:]))
            if not nxt:
                break
            proposal.append(int(nxt[0]))
            seq.append(int(nxt[0]))
        return proposal

    def _index_upto(self, token_ids: list[int], seq_id: int | None):
        """Index every position once; the tail walk only ever needs new tokens."""
        n = self.n
        index = self._index.setdefault(seq_id, {})
        start = self._indexed.get(seq_id, 0)
        for i in range(start, len(token_ids) - n):
            key = tuple(int(t) for t in token_ids[i:i + n])
            if key not in index:
                index[key] = [int(t) for t in
                              token_ids[i + n:i + n + max(4, self.span)]]
        self._indexed[seq_id] = len(token_ids) - n
        return index

    def propose_batch(self, seqs) -> list[list[int]]:
        return [self.propose(list(s.token_ids), s.seq_id) for s in seqs]
