import atexit
from copy import copy
from dataclasses import replace
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch

from qslab.runtime.config import Config  # noqa: F401
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.state.sequence import Sequence
from qslab.runtime.engine.scheduler import Scheduler
from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.model.context import reset_context
from qslab.runtime.speculation.factory import prepare_proposer, create_proposer
from qslab.speculation.acceptance import accepted_prefix, rejection_sample


class LLMEngine:

    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        draft_config, dflash_config, feature_layers = prepare_proposer(config, self.tokenizer)
        runner_config = copy(config)
        if draft_config or dflash_config:
            runner_config.gpu_memory_utilization -= config.draft_gpu_memory_reserve
        self.model_runner = ModelRunner(runner_config, feature_layers=feature_layers)
        config.num_kvcache_blocks = runner_config.num_kvcache_blocks
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.spec_method = config.spec_method
        self.proposer = None
        try:
            self.proposer = create_proposer(config, self.model_runner.model, draft_config, dflash_config)
        except Exception:
            self.exit()
            raise
        if dflash_config:
            # Reusable target KV prefixes do not include DFlash hidden features.
            self.scheduler.block_manager.ENABLE_PREFIX_CACHE = False
        atexit.register(self.exit)

    def exit(self):
        # idempotent: atexit fires this after a test may already have released
        # the runner to free the card for the next module
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        if hasattr(self, 'scheduler'):
            self.abort_all()
        proposer = getattr(self, "proposer", None)
        if proposer is not None and hasattr(proposer, "exit"):
            proposer.exit()
        runner.exit()
        del self.model_runner
        atexit.unregister(self.exit)

    def abort_request(self, seq_id: int, reason: str = 'cancelled'):
        seq = self.scheduler.abort(seq_id, reason)
        proposer = getattr(self, 'proposer', None)
        if seq is not None and hasattr(proposer, 'drop_seqs'):
            proposer.drop_seqs([seq_id])

    def abort_all(self, reason: str = 'cancelled'):
        for seq in (*self.scheduler.waiting, *self.scheduler.running):
            self.abort_request(seq.seq_id, reason)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        """Queue one request and hand back its Sequence."""
        if self.spec_method == 'dflash' and sampling_params.temperature > 1e-6:
            raise ValueError('DFlash currently supports greedy decoding only')
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt or len(prompt) >= self.scheduler.max_model_len:
            raise ValueError('prompt must be nonempty and shorter than max_model_len')
        budget = min(sampling_params.max_tokens, self.scheduler.max_model_len - len(prompt))
        max_blocks = len(self.scheduler.block_manager.blocks)
        required = (len(prompt) + budget - 2) // self.scheduler.block_size + 1
        if required > max_blocks:
            raise ValueError('request cannot fit in the KV pool; reduce context or enlarge the pool')
        sampling_params = replace(sampling_params, max_tokens=budget)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        if self.is_finished():
            return [], 0
        try:
            return self._step()
        except Exception:
            self.abort_all('error')
            reset_context()
            raise

    def _step(self):
        proposal_probs = self._propose()
        seqs, has_prefill = self.scheduler.schedule()
        num_queries = sum(seq.num_scheduled_tokens for seq in seqs)
        if any(seq.spec_verify for seq in seqs):
            committed = self._verify(seqs, has_prefill, proposal_probs)
        else:
            token_ids = self.model_runner.run(seqs, has_prefill)
            before = sum(len(seq) for seq in seqs)
            self.scheduler.postprocess(seqs, token_ids, has_prefill)
            committed = sum(len(seq) for seq in seqs) - before
        if self.spec_method == 'dflash':
            self._observe_target(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        proposer = getattr(self, 'proposer', None)
        if outputs and hasattr(proposer, 'drop_seqs'):
            proposer.drop_seqs([seq_id for seq_id, _ in outputs])
        if self.spec_method in ('draft', 'dflash'):
            running = {seq.seq_id for seq in self.scheduler.running}
            if self.spec_method == 'dflash':
                running.update(seq.seq_id for seq in self.scheduler.waiting if seq.num_cached_tokens)
            proposer.drop_seqs([sid for sid in proposer.drafts if sid not in running])
        return outputs, num_queries if has_prefill else -committed

    def _propose(self):
        """The runtime supplies histories; the scheduler later limits their query budget."""
        proposer = getattr(self, 'proposer', None)
        seqs = list(self.scheduler.running)
        if proposer is None or not seqs:
            return None
        if self.spec_method in ('draft', 'dflash'):
            # Paid proposals are limited before running the small model.
            if self.scheduler.waiting and not self.scheduler.enable_chunked_prefill:
                return None
            budget, selected, limits = self.scheduler.max_num_batched_tokens, [], {}
            for seq in seqs[:self.scheduler.max_num_seqs]:
                if budget <= 0:
                    break
                count = max(0, min(self.scheduler.proposal_gamma, budget - 1,
                    seq.max_tokens - seq.num_completion_tokens - 1,
                    self.scheduler.max_model_len - len(seq) - 1))
                selected.append(seq)
                limits[seq.seq_id] = count
                budget -= count + 1
            proposals = proposer.propose_batch(selected, limits)
            for seq in seqs:
                seq.spec_drafts = proposals[seq.seq_id].tokens if seq.seq_id in proposals else []
            return {sid: proposal.probabilities for sid, proposal in proposals.items()
                    if proposal.probabilities is not None}
        else:
            drafts = proposer.propose_batch([seq.token_ids for seq in seqs],
                                            [seq.seq_id for seq in seqs])
            probabilities = None
        for seq, tokens in zip(seqs, drafts):
            seq.spec_drafts = tokens[:self.scheduler.proposal_gamma]
        return probabilities

    def _observe_target(self, seqs):
        by_id = {seq.seq_id: seq for seq in seqs}
        runner = self.model_runner
        for seq_id, start, offset, count in runner.feature_ranges:
            seq = by_id[seq_id]
            if seq.is_finished:
                continue
            kept = min(count, seq.num_cached_tokens - start)
            if kept > 0:
                self.proposer.observe(seq_id, start, runner.context_features[offset:offset + kept])

    def _verify(self, seqs, has_prefill, proposal_probs):
        logits, counts = self.model_runner.verify_logits(seqs, has_prefill)
        temperatures, drafts = [], []
        for seq, count in zip(seqs, counts):
            temperatures.extend([seq.temperature] * count)
            drafts.extend(seq.spec_drafts + [-1] if seq.spec_verify else [-1] * count)
        temps = torch.tensor(temperatures, device=logits.device, dtype=torch.float32)
        sampled = self.model_runner.sampler(logits, temps)
        if any(seq.spec_verify and seq.temperature > 1e-6 for seq in seqs):
            draft_probs = None
            if proposal_probs is not None:
                rows = []
                for seq, count in zip(seqs, counts):
                    row = torch.zeros(count, logits.shape[-1], device=logits.device, dtype=torch.float32)
                    if seq.spec_verify and seq.seq_id in proposal_probs:
                        n = len(seq.spec_drafts)
                        row[:n] = proposal_probs[seq.seq_id][:n]
                    rows.append(row)
                draft_probs = torch.cat(rows)
            sampled = rejection_sample(logits, temps,
                torch.tensor(drafts, device=logits.device, dtype=torch.int64), draft_probs, sampled)
        verified = sampled.tolist()
        offset, committed = 0, 0
        for seq, count in zip(seqs, counts):
            row = verified[offset:offset + count]
            offset += count
            if seq.spec_verify:
                tokens, accepted = accepted_prefix(seq.spec_drafts, row)
                committed += self.scheduler.commit_verified(seq, tokens, accepted)
            else:
                before = len(seq)
                self.scheduler.postprocess([seq], row or [0], seq.is_prefill)
                committed += len(seq) - before
        return committed

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError('one sampling_params entry is required per prompt')
        queued = []
        try:
            for prompt, sp in zip(prompts, sampling_params):
                queued.append(self.add_request(prompt, sp))
        except Exception:
            for seq in queued:
                self.abort_request(seq.seq_id)
            pbar.close()
            raise
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
