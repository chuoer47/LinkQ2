"""Autoregressive draft execution; owns its cache, never target acceptance."""
from copy import copy

import torch

from qslab.runtime.config import Config
from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.model.context import reset_context
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.block_manager import BlockManager
from qslab.runtime.state.sequence import Sequence
from qslab.runtime.speculation import DraftProposal


def sample_proposals(logits, temperatures):
    """Return the sampled token and exactly the distribution used to draw it."""
    temps = torch.tensor(temperatures, device=logits.device, dtype=torch.float32)
    greedy = temps <= 1e-6
    tokens = logits.argmax(-1)
    if bool(greedy.all()):
        return tokens.tolist(), None
    probs = torch.softmax(logits.float() / temps.clamp_min(1e-6).unsqueeze(1), -1)
    tokens[~greedy] = torch.multinomial(probs[~greedy], 1).squeeze(1)
    return tokens.tolist(), probs


class DraftProposer:
    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.runner = ModelRunner(config)
        self.config = config
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # Speculative branches are mutable; do not publish them as reusable prefixes.
        self.block_manager.ENABLE_PREFIX_CACHE = False
        self.drafts: dict[int, Sequence] = {}
        self.prefill_calls = 0
        self.prefill_queries = 0
        self.decode_queries = 0
        self.catchup_decode_queries = 0

    def drop_seqs(self, ids):
        for seq_id in ids:
            seq = self.drafts.pop(seq_id, None)
            if seq is not None:
                self.block_manager.deallocate(seq)

    def exit(self):
        self.drop_seqs(list(self.drafts))
        if hasattr(self, 'runner'):
            self.runner.exit()
            del self.runner

    def _sync(self, target):
        seq = self.drafts.get(target.seq_id)
        if seq is None:
            seq = Sequence(target.token_ids, SamplingParams(temperature=target.temperature,
                max_tokens=target.max_tokens, ignore_eos=True))
            self.drafts[target.seq_id] = seq
        # Only the equal, actually computed prefix has reusable KV. In particular,
        # full acceptance leaves the last draft token AND the bonus to catch up.
        common = 0
        for a, b in zip(seq.token_ids[:seq.num_cached_tokens], target.token_ids):
            if a != b:
                break
            common += 1
        seq.num_cached_tokens = min(common, len(target) - 1)
        seq.token_ids = list(target.token_ids)
        seq.num_tokens = len(target)
        seq.last_token = target.last_token
        seq.temperature = target.temperature
        while len(seq.block_table) > seq.num_blocks:
            self.block_manager.release_tail(seq)
        return seq

    @torch.inference_mode()
    def propose_batch(self, targets, limits):
        """One bounded catch-up batch, followed by at most k M=1 decode steps.

        An unfinished catch-up returns no candidates; the target may decode
        normally while this model builds its history over subsequent rounds.
        """
        proposals = {t.seq_id: DraftProposal([]) for t in targets}
        ready, catchup, short_catchup = [], [], []
        catchup_budget = self.config.max_num_batched_tokens
        for target in targets:
            count = limits[target.seq_id]
            if not count:
                continue
            seq = self._sync(target)
            # Cover all proposal inputs, but not the final sampled token's KV.
            needed = (len(seq) + count - 2) // self.config.kvcache_block_size + 1
            missing = max(0, needed - len(seq.block_table))
            if missing > len(self.block_manager.free_block_ids):
                self.drop_seqs([target.seq_id])
                continue
            while len(seq.block_table) < needed:
                seq.block_table.append(self.block_manager._allocate_block())
            pending = len(seq) - seq.num_cached_tokens
            if pending == 1:
                ready.append((target.seq_id, seq))
            elif pending == 2 and not self.runner.enforce_eager and catchup_budget:
                short_catchup.append((target.seq_id, seq))
                catchup_budget -= 1
            elif catchup_budget:
                seq.is_prefill = True
                seq.num_scheduled_tokens = min(pending, catchup_budget)
                seq.should_sample = seq.num_scheduled_tokens == pending
                catchup_budget -= seq.num_scheduled_tokens
                catchup.append((target.seq_id, seq))

        # Full acceptance typically leaves two known tokens to compute. Catch
        # up the first with an M=1 replay, then sample after the second below.
        # This keeps short catch-up on Graph without a second graph family.
        if short_catchup:
            queries = []
            for _, seq in short_catchup:
                query = copy(seq)
                query.num_tokens = seq.num_cached_tokens + 1
                query.last_token = seq.token_ids[seq.num_cached_tokens]
                queries.append(query)
            inputs, positions = self.runner.prepare_decode(queries)
            try:
                self.runner.run_model(inputs, positions, False)
            finally:
                reset_context()
            for _, seq in short_catchup:
                seq.num_cached_tokens += 1
            self.decode_queries += len(queries)
            self.catchup_decode_queries += len(queries)
            ready.extend(short_catchup)

        if catchup:
            seqs = [seq for _, seq in catchup]
            inputs, positions = self.runner.prepare_prefill(seqs)
            try:
                hidden = self.runner.run_model(inputs, positions, True)
                indices, offset = [], 0
                for seq in seqs:
                    offset += seq.num_scheduled_tokens
                    if seq.should_sample:
                        indices.append(offset - 1)
                selected = [item for item in catchup if item[1].should_sample]
                if selected:
                    logits = self.runner.model.compute_logits(hidden[indices])
                    self._append(selected, logits, proposals)
            finally:
                reset_context()
            for _, seq in catchup:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                self.prefill_queries += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
            self.prefill_calls += 1
            ready.extend(selected)

        for _ in range(max(limits.values(), default=0)):
            active = [(sid, seq) for sid, seq in ready
                      if len(proposals[sid].tokens) < limits[sid]]
            if not active:
                break
            seqs = [seq for _, seq in active]
            inputs, positions = self.runner.prepare_decode(seqs)
            try:
                hidden = self.runner.run_model(inputs, positions, False)
                self._append(active, self.runner.model.compute_logits(hidden), proposals)
            finally:
                reset_context()
            for seq in seqs:
                seq.num_cached_tokens += 1
            self.decode_queries += len(seqs)
        return proposals

    def _append(self, active, logits, proposals):
        tokens, probabilities = sample_proposals(logits, [seq.temperature for _, seq in active])
        for i, ((sid, seq), token) in enumerate(zip(active, tokens)):
            seq.append_token(token)
            proposals[sid].tokens.append(token)
            if probabilities is not None and seq.temperature > 1e-6:
                row = probabilities[i:i + 1]
                previous = proposals[sid].probabilities
                proposals[sid].probabilities = row if previous is None else torch.cat((previous, row))
