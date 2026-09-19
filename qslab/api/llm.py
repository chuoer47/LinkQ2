"""L4: the public qslab interface — ``LLM(model_path).generate(...)``.

This is the only entry point users need. It wraps the mainline runtime
(``qslab/runtime``, M8+): paged int4 KV pool, continuous batching, CUDA graphs,
W4 packed weights and speculative decoding.

The frozen M0-M7 engine (``qslab/engine``) deliberately has no facade here — it
stays reachable as ``qslab.engine.QslabEngine``, which is how the M0-M7
benchmark scripts and the oracle cross-checks already construct it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams as _RuntimeParams

# The runtime has no greedy branch — it always samples, and dividing the logits
# by 1e-6 makes softmax an argmax one-hot. That is the convention every runtime
# test and the draft proposer already use.
GREEDY = 1e-6


@dataclass(slots=True)
class SamplingParams:
    """Decode-time knobs. Greedy by default, matching the M0+ measurement protocol."""
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
    """Unified qslab entry point over the runtime.

    Example:
        llm = LLM("models/Qwen3-8B", w4="models/Qwen3-8B-qslab-w4-awq",
                  smooth_kv="results/smooth_kv4_qwen3-8b.pt")
        print(llm.generate(" The capital of France is",
                           SamplingParams(max_tokens=32))["text"])

    Unrecognised keywords pass straight through to ``qslab.runtime.config.Config``,
    so any runtime knob (``enforce_eager``, ``gpu_memory_utilization``,
    ``kvcache_block_size`` ...) is reachable without duplicating its signature.
    """

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
    def tokenizer(self):
        return self._engine.tokenizer

    @property
    def stats(self) -> dict:
        """Speculative counters for the last call: steps / proposals / accepted
        / committed plus acceptance_rate and mean_len (tokens per verify step,
        i.e. the speedup the window buys). Empty when speculation is off."""
        sched = self._engine.scheduler
        return dict(sched.spec_stats) if sched.gamma else {}

    def generate(self, prompt: str | list[int],
                 params: SamplingParams | None = None) -> dict:
        """One prompt. Returns {'text', 'token_ids', 'stats'}."""
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
