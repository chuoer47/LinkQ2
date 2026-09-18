"""Draft-model speculative proposer (chained, Qwen3-0.6B style).

The n-gram proposer (ngram.py) is a CPU lookup; this one is a small model
drafting autoregressively. Both expose propose_batch() and plug into the same
verify machinery (design-m9 §2) — the target engine never knows which kind of
proposer is attached.

The draft is a full second runtime: its own ModelRunner (paged int4 KV pool,
CUDA-graphed decode) and its own Scheduler. That is vLLM V1's shape (a
separate draft worker with its own engine) rather than a bolt-on cache, and
it means drafting reuses the same store/schedule/acceptance discipline the
target uses — no new KV code.

Lockstep protocol (per propose_batch call, before the target's verify):
  1. SYNC: each draft sequence is truncated to the target's committed tokens.
     The draft's KV below the committed prefix is valid by construction: it
     was computed causally over exactly those tokens (accepted drafts were
     fed at their positions; rejected proposals wrote slots that the next
     drafting overwrites, same append-only recovery as the target).
  2. CATCHUP + DRAFT: gamma plain decode steps through the draft scheduler
     (batched across sequences). The first step processes the target's last
     committed token — its KV slot is the one the bonus token invalidated —
     and each step's greedy sample is the next proposal.
  3. Proposals = the tokens the draft appended beyond the committed prefix.

The draft's block table is trimmed on sync so can_append/may_append stay
canonical (the target's verify trims the same way).
"""
from __future__ import annotations

import torch

from qslab.runtime.config import Config
from qslab.runtime.context import reset_context, set_context
from qslab.runtime.model_runner import ModelRunner
from qslab.runtime.sampler import Sampler  # noqa: F401  (import order sanity)
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.scheduler import Scheduler
from qslab.runtime.sequence import Sequence


class DraftProposer:

    def __init__(self, model: str, gamma: int, max_model_len: int,
                 max_num_seqs: int, gpu_memory_utilization: float = 0.95,
                 smooth_kv: str | None = None, w4: str | None = None,
                 w4_backend: str = "w4.auto"):
        self.gamma = gamma
        # the draft never verifies or proposes speculatively itself: its own
        # scheduler runs the plain decode path only. W4 weights are the
        # measured lever for the proposal phase (840MB fp16 reads at 35%
        # bandwidth -> 210MB packed; notes/M9 §5)
        self.config = Config(model, max_model_len=max_model_len,
                             max_num_seqs=max_num_seqs,
                             gpu_memory_utilization=gpu_memory_utilization,
                             smooth_kv=smooth_kv, w4=w4,
                             w4_backend=w4_backend, compile=True)
        # normally set by LLMEngine; keep the class invariant ourselves so
        # the proposer also works standalone (sequence granularity must
        # match the runner's block size)
        Sequence.block_size = self.config.kvcache_block_size
        self.runner = ModelRunner(self.config, 0, [])
        self.sched = Scheduler(self.config)
        self.drafts: dict[int, Sequence] = {}          # target seq_id -> draft seq
        self._warm()

    def _warm(self):
        """Compile everything the proposal path will ever run.

        The engine init's graph capture warms the decode path under
        inference_mode, but propose_batch runs outside it — dynamo treats
        that as a different guard state and recompiles (~5s) on the FIRST
        real proposal. A dummy propose here pays that once, before serving.
        """
        dummy = Sequence([0] * 8, SamplingParams(temperature=1e-6,
                                                 max_tokens=1 << 30,
                                                 ignore_eos=True))
        self.propose_batch([dummy])
        self.drop_seqs([dummy.seq_id])

    # ---------------- lifecycle ----------------

    def _make_draft(self, target: Sequence) -> Sequence:
        # drafts run greedy; it never finishes on its own (huge budget,
        # ignore eos) — its lifecycle follows the target's
        sp = SamplingParams(temperature=1e-6, max_tokens=1 << 30,
                            ignore_eos=True)
        d = Sequence(list(target.token_ids), sp)
        self.sched.add(d)
        self.drafts[target.seq_id] = d
        return d

    def drop_seqs(self, finished_ids: list[int]):
        """Free draft sequences whose target is done."""
        for seq_id in finished_ids:
            d = self.drafts.pop(seq_id, None)
            if d is None:
                continue
            if d.status.value == "RUNNING" or d in self.sched.running:
                self.sched.running.remove(d)
            else:
                try:
                    self.sched.waiting.remove(d)
                except ValueError:
                    pass
            self.sched.block_manager.deallocate(d)

    def exit(self):
        for seq_id in list(self.drafts):
            self.drop_seqs([seq_id])
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.model = None
            runner.kv_cache = None
            del self.runner
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- proposal ----------------

    def _sync(self, d: Sequence, target: Sequence):
        """Truncate the draft to the target's committed tokens.

        No recompute is ever needed: the draft's KV below position n-1 was
        computed causally over exactly the target's tokens (accepted drafts
        were fed at their positions by the draft itself), the bonus token's
        slot is rewritten by the first catchup query, and rejected-proposal
        slots are masked by causality until the next drafting overwrites
        them. The block table is trimmed to canonical length so the scheduler
        appends blocks on demand (may_append is idempotent).
        """
        committed = target.token_ids
        n = len(committed)
        d.token_ids = list(committed)
        d.num_tokens = n
        d.last_token = committed[-1]
        # KV validity: positions 0..n-2 are computed (causal, correct tokens);
        # position n-1 (the bonus) is written by the first catchup query.
        # num_cached must track that, or the drift accumulates and pushes
        # hash_blocks past the trimmed table (found by the draft bench).
        d.num_cached_tokens = n - 1
        while len(d.block_table) > d.num_blocks:
            self.sched.block_manager.release_tail(d)

    def propose_batch(self, seqs: list[Sequence]) -> list[list[int]]:
        targets = []
        for t in seqs:
            d = self.drafts.get(t.seq_id)
            if d is None:
                d = self._make_draft(t)
            self._sync(d, t)
            targets.append((t, d))

        # the synced length IS the proposal base: tokens the draft appends
        # from here are proposals (for a fresh draft the prefill's own sample
        # is the first one)
        starts = {d.seq_id: len(d.token_ids) for _, d in targets}

        # one prefill pass if any draft is fresh (flash prefill writes its
        # prompt KV; the prefill's sample is that draft's first proposal)
        if self.sched.waiting:
            draft_seqs, is_prefill = self.sched.schedule()
            assert is_prefill
            token_ids = self.runner.call("run", draft_seqs, True)
            self.sched.postprocess(draft_seqs, token_ids, True)

        self._draft_steps([d for _, d in targets])
        out = []
        for t, d in targets:
            start = starts[d.seq_id]
            out.append(list(d.token_ids[start:start + self.gamma]))
        return out

    def _draft_steps(self, draft_seqs: list[Sequence]) -> list[list[int]]:
        """gamma tight greedy decode steps, batched across draft sequences.

        This bypasses the scheduler round-trips (schedule/postprocess per
        step) that cost more than the 0.6B forward itself: the draft window's
        slots are reserved up front (same position-driven discipline as the
        target's verify), each step is prepare -> graph replay -> argmax ->
        append, and the next sync trims whatever the window over-reserved.
        Pool pressure: a sequence whose window cannot be reserved drafts
        nothing this round (padding neutrality makes that a plain step).
        """
        gamma = self.gamma
        mr = self.runner
        bm = self.sched.block_manager
        bs = len(draft_seqs)
        active = []
        for d in draft_seqs:
            required = (len(d) + gamma - 1) // bm.block_size + 1
            if len(bm.free_block_ids) >= max(0, required - len(d.block_table)):
                while len(d.block_table) < required:
                    d.block_table.append(bm._allocate_block())
                active.append(d)
        for _ in range(gamma):
            if not active:
                break
            input_ids, positions, slots, ctx = [], [], [], []
            for d in active:
                L = len(d)
                input_ids.append(d.last_token)
                positions.append(L - 1)
                ctx.append(L)
                slots.append(d.block_table[(L - 1) // bm.block_size] * bm.block_size
                         + (L - 1) % bm.block_size)
            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            slots = torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            ctx = torch.tensor(ctx, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            max_len = max(len(d.block_table) for d in active)
            bt = [d.block_table + [-1] * (max_len - len(d.block_table)) for d in active]
            bt = torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(False, slot_mapping=slots, context_lens=ctx, block_tables=bt)
            logits = mr.run_model(input_ids, positions, False)
            reset_context()
            tok = logits.float().argmax(dim=-1).tolist()
            for d, t in zip(active, tok):
                d.append_token(t)
                d.num_cached_tokens += 1
        return draft_seqs
