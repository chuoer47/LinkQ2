"""Public qslab interface: LLM(model_path).generate(...)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qslab.runtime.engine.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams as _RuntimeParams

# The runtime has no greedy branch — it always samples, and dividing the logits
# by 1e-6 makes softmax an argmax one-hot. That is the convention every runtime
# test and the draft proposer already use.
GREEDY = 1e-6


@dataclass(slots=True)
class SamplingParams:
    """Decode-time knobs."""
    max_tokens: int = 128
    temperature: float = GREEDY
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature <= 0.0:
            self.temperature = GREEDY

    def to_runtime(self) -> _RuntimeParams:
        return _RuntimeParams(temperature=self.temperature,
                              max_tokens=self.max_tokens,
                              ignore_eos=self.ignore_eos)


class LLM:
    """Unified qslab entry point over the runtime."""

    def __init__(self, model_path: str | Path, *,
                 w4: str | None = None,               # qslab_w4_v1 packed dir
                 w4_backend: str = "w4.auto",         # w4.v1 | w4.marlin | w4.auto
                 smooth_kv: str | None = None,        # SmoothAttention calibration
                 spec: str | None = None,             # ngram | lookahead | draft
                 spec_gamma: int = 4,
                 spec_ngram_size: int = 3,
                 spec_lookahead_span: int = 8,
                 spec_adaptive_gamma: bool = False,
                 draft: str | None = None,            # draft model dir (spec="draft")
                 draft_w4: str | None = None,
                 draft_w4_backend: str = "w4.v1",
                 **runtime_config):
        self.model_path = str(model_path)
        if spec == "draft":
            assert draft, "spec='draft' needs draft=<model dir>"
        elif draft:
            raise ValueError(f"draft model given but spec={spec!r} — use spec='draft'")
        self._engine = LLMEngine(
            self.model_path, w4=w4, w4_backend=w4_backend, smooth_kv=smooth_kv,
            spec_method=spec, spec_num_drafts=spec_gamma,
            spec_ngram_size=spec_ngram_size,
            spec_lookahead_span=spec_lookahead_span,
            spec_adaptive_gamma=spec_adaptive_gamma,
            draft_model=draft, draft_w4=draft_w4,
            draft_w4_backend=draft_w4_backend, **runtime_config)

    # ------------------------------------------------------------------
    @property
    def engine(self):
        """The wrapped LLMEngine."""
        # Public because the service surface drives step() directly and must not reach into
        #   a private attribute.
        return self._engine

    @property
    def tokenizer(self):
        return self._engine.tokenizer

    @property
    def stats(self) -> dict:
        """Speculative counters for the last call: steps/proposals/accepted/committed,
        acceptance_rate, mean_len; empty when speculation is off."""
        sched = self._engine.scheduler
        return dict(sched.spec_stats) if sched.gamma else {}

    def generate(self, prompt: str | list[int],
                 params: SamplingParams | None = None) -> dict:
        """Generate one prompt; returns text, token_ids and stats."""
        return self.generate_batch([prompt], params)[0]

    def generate_batch(self, prompts: list[str | list[int]],
                       params: SamplingParams | None = None) -> list[dict]:
        """Several prompts, run through the scheduler's continuous batching."""
        p = params or SamplingParams()
        self._engine.scheduler.reset_spec_stats()
        out = self._engine.generate(prompts, p.to_runtime(), use_tqdm=False)
        stats = self.stats
        return [{"text": o["text"], "token_ids": o["token_ids"], "stats": stats}
                for o in out]

    def kv_memory_bytes(self) -> int:
        """Resident int4 KV pool (all layers, K and V plus their scales)."""
        pool = getattr(self._engine.model_runner, "kv_cache", None) or []
        return sum(t.numel() * t.element_size()
                   for (kq, ks), (vq, vs) in pool for t in (kq, ks, vq, vs))

    def weight_memory_bytes(self) -> int:
        """Packed W4 modules plus whatever stays in fp16 (embeddings, norms)."""
        model = self._engine.model_runner.model
        packed = sum(m.weight_memory_bytes() for m in model.modules()
                     if hasattr(m, "weight_memory_bytes"))
        # a swapped W4Linear keeps its bytes in buffers or in its backend's
        # repack, never in parameters(), so this sums the unswapped fp16 modules
        return packed + sum(p.numel() * p.element_size() for p in model.parameters())
