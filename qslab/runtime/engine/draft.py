"""Draft-model speculative proposer: a small model drafting autoregressively, behind the
same propose_batch() interface as the n-gram lookup, so the target engine never knows
which is attached."""
# The draft is a full second runtime — its own ModelRunner and Scheduler — so drafting
#   reuses the target's store, schedule and acceptance discipline instead of adding KV code.
# A proposer that samples also exposes propose_probs: a [bs*(gamma+1), V] float32 tensor
#   whose row i*(gamma+1)+m is the distribution proposal m of sequence i was drawn from.
#   Absent means every proposal was deterministic, i.e. p is one-hot.
# Lockstep per propose_batch call: sync each draft sequence to the target's committed
#   tokens, run gamma plain decode steps batched across sequences, and propose whatever was
#   appended beyond the committed prefix. The first drafting step re-queries the target's
#   last committed token — the slot its bonus token invalidated.
from __future__ import annotations

import torch

from qslab.runtime.config import Config
from qslab.runtime.model.context import reset_context, set_context
from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.execute.sampler import Sampler  # noqa: F401  (import order sanity)
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.state.sequence import Sequence


def _sample_rows(logits: torch.Tensor, temps: list[float]):
    """(tokens, probs) — the sample and the distribution behind it."""
    # A fully greedy batch skips the softmax and takes argmax: a near-tie would make the
    #   draft's choice a coin flip, and the one-hot fast path assumes a deterministic
    #   proposal.
    # Drafting temperature is a correctness matter, not just quality: a sampling proposer
    #   must report the distribution behind every token, and drafts at each target
    #   sequence's own temperature.
    if max(temps) <= 1e-3:
        return logits.argmax(dim=-1), None
    t = torch.tensor(temps, dtype=torch.float32, device=logits.device)
    probs = torch.softmax(logits.float().div_(t.unsqueeze(1)), dim=-1)
    tok = probs.argmax(dim=-1)
    sampled = t > 1e-3
    if bool(sampled.any()):
        tok = tok.clone()
        tok[sampled] = torch.multinomial(probs[sampled], 1).squeeze(1)
    return tok, probs


class DraftProposer:

    def __init__(self, model: str, gamma: int, max_model_len: int,
                 max_num_seqs: int, gpu_memory_utilization: float = 0.95,
                 smooth_kv: str | None = None, w4: str | None = None,
                 w4_backend: str = "w4.auto"):
        self.gamma = gamma
        # the draft never verifies or proposes speculatively itself: its own scheduler runs
        #   the plain decode path only.
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
        # proposer interface: [bs*(gamma+1), V] rows for the last call, or None
        # when every sequence in it was greedy (see the module docstring)
        self.propose_probs: torch.Tensor | None = None
        self._warm()

    def _warm(self):
        """Compile everything the proposal path will ever run."""
        # propose_batch runs outside the inference_mode the graph capture warmed under, so
        #   dynamo treats it as a new guard state and recompiles on the first real proposal;
        #   a dummy propose pays that once, before serving.
        dummy = Sequence([0] * 8, SamplingParams(temperature=1e-6,
                                                 max_tokens=1 << 30,
                                                 ignore_eos=True))
        self.propose_batch([dummy])
        self.drop_seqs([dummy.seq_id])

    # ---------------- lifecycle ----------------

    def _make_draft(self, target: Sequence) -> Sequence:
        # the draft follows its target's temperature: the ratio test needs the
        # distribution a proposal actually came from, so a temperature-1
        # sequence must not be served temperature-0 proposals. It never
        # finishes on its own (huge budget, ignore eos) — its lifecycle
        # follows the target's
        sp = SamplingParams(temperature=target.temperature,
                            max_tokens=1 << 30, ignore_eos=True)
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
        """Truncate the draft to the target's committed tokens."""
        # No recompute is needed: the draft's KV below the committed prefix was computed
        #   causally over exactly those tokens, the bonus token's slot is rewritten by the
        #   first drafting query, and rejected proposals sit in slots causality hides until
        #   the next drafting overwrites them.
        # The block table is trimmed to canonical length so the scheduler appends blocks on
        #   demand (may_append is idempotent).
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
        # proposal index -> distribution it was drawn from, per draft sequence,
        # appended in the order the tokens land in the stream
        rows: dict[int, list[torch.Tensor]] = {}

        # one prefill pass if any draft is fresh (flash prefill writes its
        # prompt KV; the prefill's sample is that draft's first proposal, so
        # its distribution has to be captured with it)
        if self.sched.waiting:
            draft_seqs, is_prefill = self.sched.schedule()
            assert is_prefill
            token_ids, probs = self._prefill_capture(draft_seqs)
            self.sched.postprocess(draft_seqs, token_ids, True)
            for i, d in enumerate(draft_seqs):
                if d.temperature > 1e-3:
                    rows[d.seq_id] = [probs[i:i + 1]]

        self._draft_steps([d for _, d in targets], rows)
        out = []
        proposals = []
        for t, d in targets:
            start = starts[d.seq_id]
            props = list(d.token_ids[start:start + self.gamma])
            out.append(props)
            r = rows.get(d.seq_id)
            kept = r[:len(props)] if r else []
            proposals.append(kept or None)
        self.propose_probs = self._assemble_probs(proposals, len(seqs))
        return out

    def _prefill_capture(self, draft_seqs: list[Sequence]):
        """The draft's prefill, keeping the softmax behind proposal 0."""
        # Same calls as runner.run(seqs, True) — that path throws the distribution away, and
        #   a proposal whose p is unknown cannot be ratio-tested losslessly.
        mr = self.runner
        input_ids, positions = mr.prepare_prefill(draft_seqs)
        logits = mr.run_model(input_ids, positions, True)
        tok, probs = _sample_rows(logits, [d.temperature for d in draft_seqs])
        reset_context()
        return tok.tolist(), probs

    def _assemble_probs(self, per_seq, bs: int) -> torch.Tensor | None:
        """[bs*(gamma+1), V] rows laid out the way the verify batch is flat: row
        i*(gamma+1)+m backs proposal m of sequence i."""
        # float32, not the model's fp16: underflowing a low-probability proposal to 0 would
        #   turn a rejection into a certain accept.
        # Bonus rows, padded rows and greedy sequences stay zero; the acceptance mask never
        #   reads them.
        if not any(per_seq):
            return None
        M = self.gamma + 1
        vocab = next(r[0].shape[-1] for r in per_seq if r)
        mat = torch.zeros(bs * M, vocab, dtype=torch.float32, device="cuda")
        for i, r in enumerate(per_seq):
            if not r:
                continue
            for m, row in enumerate(r):
                mat[i * M + m] = row[0]
        return mat

    def _draft_steps(self, draft_seqs: list[Sequence],
                     rows: dict[int, list[torch.Tensor]]):
        """Gamma tight decode steps, batched across draft sequences."""
        # Bypasses the per-step schedule/postprocess round-trips: the window's slots are
        #   reserved up front, each step is prepare -> replay -> sample -> append, and the
        #   next sync trims whatever the window over-reserved.
        # Pool pressure: a sequence whose window cannot be reserved drafts nothing this
        #   round (padding neutrality makes that a plain step).
        gamma = self.gamma
        mr = self.runner
        bm = self.sched.block_manager
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
            temps = [d.temperature for d in active]
            max_len = max(len(d.block_table) for d in active)
            bt = [d.block_table + [-1] * (max_len - len(d.block_table)) for d in active]
            bt = torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            set_context(False, slot_mapping=slots, context_lens=ctx, block_tables=bt)
            logits = mr.run_model(input_ids, positions, False)
            reset_context()
            tok, probs = _sample_rows(logits, temps)
            tok = tok.tolist()
            for i, (d, t) in enumerate(zip(active, tok)):
                d.append_token(t)
                d.num_cached_tokens += 1
                if d.temperature > 1e-3:
                    rows.setdefault(d.seq_id, []).append(probs[i:i + 1])
