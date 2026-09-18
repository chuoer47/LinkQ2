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
