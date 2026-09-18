import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from qslab.runtime.config import Config  # noqa: F401
from qslab.runtime.sampling_params import SamplingParams
from qslab.runtime.sequence import Sequence
from qslab.runtime.scheduler import Scheduler
from qslab.runtime.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # llama-style naming: `model` is positional, everything else by field
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        # qslab: single-GPU runtime; nano-vllm's TP worker processes removed.
        self.events = []
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        if config.spec_method == "draft":
            # a draft model proposer owns a second runtime (runner+scheduler)
            # for the small model — proposals are GPU forwards, so the
            # propose phase lives in step(), not in the scheduler
            import os
            from qslab.runtime.draft import DraftProposer
            assert config.draft_model, "spec_method='draft' needs draft_model"
            calib = "results/smooth_kv4_" + os.path.basename(config.draft_model).lower() + ".pt"
            assert os.path.exists(calib), f"draft calibration missing: {calib}"
            self.scheduler.proposer = DraftProposer(
                config.draft_model, config.spec_num_drafts,
                config.max_model_len, config.max_num_seqs,
                gpu_memory_utilization=config.draft_gpu_memory_utilization,
                smooth_kv=calib, w4=config.draft_w4,
                w4_backend=config.draft_w4_backend)
        atexit.register(self.exit)

    def exit(self):
        # idempotent: atexit fires this after a test may already have released
        # the runner to free the card for the next module
        proposer = getattr(self.scheduler, "proposer", None) if hasattr(self, "scheduler") else None
        if proposer is not None and hasattr(proposer, "exit"):
            proposer.exit()
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        runner.call("exit")
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        elif any(seq.spec_verify for seq in seqs):
            # speculative verify step: one M-row forward, greedy acceptance
            # (design-m9). Sequences without drafts ride along padded — their
            # step is semantically an ordinary decode.
            self._propose(seqs)
            token_ids = self.model_runner.call("run_verify", seqs)
            num_tokens = -self.scheduler.postprocess_verify(seqs, token_ids)
        else:
            num_tokens = -len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if outputs:
            proposer = getattr(self.scheduler, "proposer", None)
            if proposer is not None and hasattr(proposer, "drop_seqs"):
                proposer.drop_seqs([seq_id for seq_id, _ in outputs])
        return outputs, num_tokens

    def _propose(self, seqs):
        """Fill spec_drafts for the greedy sequences of a verify batch.

        Only greedy: the argmax-equality acceptance rule is a faithful accept
        only for greedy (temperature sampling needs the ratio rule,
        design-m9 §7). Non-greedy sequences keep [] and degrade to a plain
        decode step via padding neutrality."""
        proposer = self.scheduler.proposer
        if proposer is None:
            return
        greedy = [s for s in seqs if s.temperature <= 1e-3]
        if not greedy:
            return
        drafts = proposer.propose_batch(greedy)
        gamma = self.scheduler.gamma
        max_len = self.scheduler.max_model_len
        for seq, d in zip(greedy, drafts):
            geff = min(gamma, max_len - len(seq))
            seq.spec_drafts = [t for t in d if t is not None][:max(0, geff)]

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
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
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
