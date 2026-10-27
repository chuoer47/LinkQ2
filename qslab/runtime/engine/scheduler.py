from collections import deque

from qslab.runtime.config import Config
from qslab.runtime.state.sequence import Sequence, SequenceStatus
from qslab.runtime.state.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.enable_chunked_prefill = getattr(config, 'enable_chunked_prefill', False)
        self.prefill_reserve_blocks = getattr(config, 'prefill_reserve_blocks', 0)
        # Speculative decoding: proposals are filled in by the engine's propose phase;
        #   gamma=0 disables the verify path
        self.gamma = config.spec_num_drafts if config.spec_method else 0
        self.max_model_len = config.max_model_len
        # acceptance counters for the last generate() (facade stats), and the
        # adaptive draft window. The engine passes this limit to the paid
        # draft execution before generating candidates.
        self._spec = {"steps": 0, "proposals": 0, "accepted": 0, "committed": 0}
        self.adaptive_gamma = (config.spec_adaptive_gamma
                               and config.spec_method == "draft")
        self._recent: deque[int] = deque(maxlen=4)
        self.proposal_gamma = self.gamma

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def finish(self, seq: Sequence, reason: str):
        seq.finish_reason = reason
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)

    def abort(self, seq_id: int, reason: str = 'cancelled'):
        for seq in (*self.waiting, *self.running):
            if seq.seq_id == seq_id:
                self.finish(seq, reason)
                return seq

    def schedule(self) -> tuple[list[Sequence], bool]:
        # nano-vLLM PR #276: decode first, then use the remaining token
        # budget for prefill. Pure decode retains the CUDA Graph path.
        if self.enable_chunked_prefill:
            seqs = self._schedule_decode()
            num_decode = len(seqs)
            self._schedule_prefill(seqs, sum(s.num_scheduled_tokens for s in seqs))
            has_prefill = len(seqs) > num_decode
        else:
            seqs = []
            self._schedule_prefill(seqs, 0)
            has_prefill = bool(seqs)
            if not seqs:
                seqs = self._schedule_decode()
        if not seqs:
            raise RuntimeError('scheduler cannot make progress: no sequence can be scheduled')
        return seqs, has_prefill

    def _schedule_prefill(self, scheduled_seqs, num_batched_tokens):
        num_prefill = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            if not seq.block_table:
                num_cached_blocks, num_new_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                if len(self.block_manager.free_block_ids) - num_new_blocks < self.prefill_reserve_blocks:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and num_prefill > 0:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.spec_drafts = []
            seq.spec_verify = False
            seq.is_prefill = True
            seq.should_sample = seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            num_prefill += 1

    def _schedule_decode(self):
        scheduled = []
        budget = self.max_num_batched_tokens
        while self.running and len(scheduled) < self.max_num_seqs and budget:
            seq = self.running.popleft()
            seq.should_sample = True
            remaining = min(seq.max_tokens - seq.num_completion_tokens,
                            self.max_model_len - len(seq))
            count = min(len(seq.spec_drafts), self.gamma, budget - 1, remaining - 1)
            seq.spec_drafts = seq.spec_drafts[:max(0, count)]
            seq.spec_verify = bool(seq.spec_drafts)
            count = len(seq.spec_drafts)
            while not self.block_manager.can_verify(seq, count):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = count + 1
                seq.is_prefill = False
                self.block_manager.reserve_verify(seq, count)
                scheduled.append(seq)
                budget -= seq.num_scheduled_tokens
        self.running.extendleft(reversed(scheduled))
        return scheduled

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.spec_drafts = []
        seq.spec_verify = False
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def commit_verified(self, seq: Sequence, token_ids: list[int], accepted: int) -> int:
        """Commit an algorithm decision; own KV validity and termination only."""
        proposals = len(seq.spec_drafts)
        budget = min(seq.max_tokens - seq.num_completion_tokens,
                     self.max_model_len - len(seq))
        token_ids = token_ids[:budget]
        if not seq.ignore_eos and self.eos in token_ids:
            token_ids = token_ids[:token_ids.index(self.eos) + 1]
        for token in token_ids:
            seq.append_token(token)
        # All but the final correction/bonus have valid KV. Rejected slots are
        # hidden by the new length and overwritten when queried again.
        seq.num_scheduled_tokens = len(token_ids)
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens = len(seq) - 1
        seq.num_scheduled_tokens = 0
        seq.spec_drafts = []
        seq.spec_verify = False
        while len(seq.block_table) > seq.num_blocks:
            self.block_manager.release_tail(seq)
        self._spec['steps'] += 1  # per-request verify rounds, not batch count
        self._spec['proposals'] += proposals
        self._spec['accepted'] += min(accepted, len(token_ids))
        self._spec['committed'] += len(token_ids)
        if self.adaptive_gamma:
            self._recent.append(min(accepted, len(token_ids)))
            self._adapt_gamma()
        self._finish_if_done(seq)
        return len(token_ids)

    def _finish_if_done(self, seq):
        if not seq.ignore_eos and seq.last_token == self.eos:
            self.finish(seq, 'stop')
        elif seq.num_completion_tokens >= seq.max_tokens or len(seq) >= self.max_model_len:
            self.finish(seq, 'length')

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
        # The verify graph family stays keyed on M = spec_num_drafts + 1;
        # the engine limits actual draft generation before paying its cost.
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
            self._finish_if_done(seq)
