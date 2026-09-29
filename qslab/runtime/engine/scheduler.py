from collections import deque

from qslab.runtime.config import Config
from qslab.runtime.state.sequence import Sequence, SequenceStatus
from qslab.runtime.state.block_manager import BlockManager
from qslab.runtime.engine.ngram import NGramProposer, LookaheadProposer


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Speculative decoding: proposals are filled in by the engine's propose phase;
        #   gamma=0 disables the verify path
        self.gamma = config.spec_num_drafts if config.spec_method else 0
        self.max_model_len = config.max_model_len
        if config.spec_method == "ngram":
            self.proposer = NGramProposer(config.spec_ngram_size, config.spec_num_drafts)
        elif config.spec_method == "lookahead":
            # self-proposer: persistent n-gram index + chain extension
            self.proposer = LookaheadProposer(config.spec_ngram_size,
                                              config.spec_num_drafts,
                                              span=config.spec_lookahead_span)
        else:
            # "draft": the engine owns the proposer (it needs a second runtime)
            self.proposer = None
        # acceptance counters for the last generate() (facade stats), and the
        # adaptive draft window. Gated to the draft proposer: a lookup proposal
        # costs a CPU dict hit, so cutting its window could only throw tokens
        # away, whereas the draft has a (currently unrealized, see _adapt_gamma)
        # saving to aim at.
        self._spec = {"steps": 0, "proposals": 0, "accepted": 0, "committed": 0}
        self.adaptive_gamma = (config.spec_adaptive_gamma
                               and config.spec_method == "draft")
        self._recent: deque[int] = deque(maxlen=4)
        self.proposal_gamma = self.gamma

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode (optionally speculative: a verify step processes
        # M = gamma+1 rows per sequence and commits accepted drafts)
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            geff = min(self.gamma, self.max_model_len - len(seq)) if self.gamma else 0
            if geff > 0:
                while not self.block_manager.can_verify(seq, geff):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    # proposals are filled in by the engine's propose phase
                    # (n-gram: CPU lookup; draft model: GPU forwards —
                    # schedulers should not own model execution)
                    seq.spec_drafts = []
                    seq.spec_verify = True
                    seq.num_scheduled_tokens = geff + 1
                    seq.is_prefill = False
                    self.block_manager.reserve_verify(seq, geff)
                    scheduled_seqs.append(seq)
                    continue
                continue    # self-preempted: back to waiting, do not fall through
            seq.spec_drafts = []
            seq.spec_verify = False
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess_verify(self, seqs: list[Sequence], token_ids: list[int]) -> int:
        """Accept the matched prefix, commit accepted + bonus, trim the table."""
        # Acceptance is the longest prefix where the row equals the draft, and the row after
        #   the last accepted draft contributes the bonus token.
        # The rows come from the ratio rule in ModelRunner.rejection_verify, which on
        #   refusal resamples from max(0, q - p) and so can never hand back a rejected row's
        #   own draft token — that is what makes this plain prefix walk correct for sampled
        #   and greedy proposals alike.
        M = self.gamma + 1
        committed = 0
        accepted = 0
        proposals = 0
        for i, seq in enumerate(seqs):
            row = token_ids[i * M:(i + 1) * M]
            drafts = list(seq.spec_drafts or [])
            k = len(drafts)
            a = 0
            while a < k and row[a] == drafts[a]:
                a += 1
            appended = drafts[:a] + [row[a]]
            accepted += a
            proposals += k
            # a draft can hit eos or the token budget mid-list: commit up to
            # that point and finish, exactly like the one-token path would
            budget = seq.max_tokens - seq.num_completion_tokens
            appended = appended[:budget]
            if not seq.ignore_eos and self.eos in appended:
                appended = appended[:appended.index(self.eos) + 1]
            for t in appended:
                seq.append_token(t)
            committed += len(appended)
            # hash blocks that filled up during this commit, so speculative
            # sequences contribute prefix-cache entries like plain ones
            # (hash_blocks slices token_ids per FULL block; multi-token
            # commits can complete several)
            seq.num_scheduled_tokens = len(appended)
            self.block_manager.hash_blocks(seq)
            # the pool now holds KV for positions 0..num_tokens-2 (the last
            # committed token is the freshly sampled bonus; its KV is stored
            # by the next step, like every decode step's input token)
            seq.num_cached_tokens = seq.num_tokens - 1
            seq.num_scheduled_tokens = 0
            seq.spec_drafts = []
            seq.spec_verify = False
            # trim: restore the canonical table length, releasing rejected draft blocks
            while len(seq.block_table) > seq.num_blocks:
                self.block_manager.release_tail(seq)
            finished = (not seq.ignore_eos and appended and appended[-1] == self.eos) \
                or seq.num_completion_tokens >= seq.max_tokens
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        self._spec["steps"] += 1
        self._spec["proposals"] += proposals
        self._spec["accepted"] += accepted
        self._spec["committed"] += committed
        if self.adaptive_gamma:
            # accept_len per sequence, which is what the window trades on: the
            # bonus token is free and independent of it
            self._recent.append(accepted / max(1, len(seqs)))
            self._adapt_gamma()
        return committed

    # ---------------- speculation reporting + adaptive window ----------------

    @property
    def spec_stats(self) -> dict:
        """Counters over the verify steps run so far, plus what they mean."""
        # acceptance_rate = accepted drafts / proposed; mean_len = tokens committed per
        #   verify step, i.e. the speedup the window buys — 1.0 means speculation is pure
        #   overhead.
        s = dict(self._spec)
        s["acceptance_rate"] = s["accepted"] / s["proposals"] if s["proposals"] else 0.0
        s["mean_len"] = s["committed"] / s["steps"] if s["steps"] else 0.0
        return s

    def reset_spec_stats(self):
        self._spec = {"steps": 0, "proposals": 0, "accepted": 0, "committed": 0}
        self._recent.clear()
        self.proposal_gamma = self.gamma

    def _adapt_gamma(self):
        """Shrink-only: the window can be cut, never bought back."""
        # The rule: mean accept length over the last 3 rounds <= 1.0 moves the window to
        #   gamma//2, once.
        # WINDOW is not a reaction-speed knob but a false-positive filter on an action that
        #   cannot be undone — gamma//2 is idempotent, so a wider window only changes when
        #   the single cut lands.
        # The verify graph family stays keyed on M = spec_num_drafts + 1 and the proposer
        #   still runs its full loop, so a runtime cut discards proposals that were already
        #   paid for.
        # reset_spec_stats is what hands the full window back to the next request; growing
        #   past spec_num_drafts is out of reach because the graph family is keyed on it.
        recent = list(self._recent)[-3:]
        if not recent:
            return
        avg = sum(recent) / len(recent)
        if avg <= 1.0:
            self.proposal_gamma = max(1, self.gamma // 2)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
