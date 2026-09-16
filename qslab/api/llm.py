"""L4: the public qslab interface — ``LLM(model_path).generate(...)``.

This is the only entry point users need. It assembles the layers:
  - L2 models:  the Qwen3 backbone + optional W4 quantized linears
  - L1 quant:   the W4 backend and the KV cache strategy
  - L3 engine:  the decode loop, or the speculative loop with a draft model

Everything above the engine (tokenization, batching of prompts, result
formatting) lives here so the engine stays a pure loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from qslab.engine.spec import SpeculativeEngine


@dataclass
class SamplingParams:
    """Decode-time knobs (greedy by default — matches the M0-M6 protocol)."""
    max_tokens: int = 128
    temperature: float = 0.0       # 0.0 = greedy
    top_k: int = 0
    top_p: float = 1.0
    stop_token_id: int | None = None


class LLM:
    """Unified qslab entry point.

    Example:
        llm = LLM("models/Qwen3-8B", w4="models/Qwen3-8B-qslab-w4-awq",
                  kv_mode="kv4", kv_plan="results/kv4_plan_8b.json")
        print(llm.generate(" The capital of France is", SamplingParams(max_tokens=32)))
    """

    def __init__(self, model_path: str | Path, *,
                 device: str = "cuda:0",
                 w4: str | None = None,            # packed checkpoint dir
                 w4_backend: str = "w4.auto",      # w4.v1 | w4.marlin | w4.auto
                 kv_mode: str = "fp16",            # fp16 | kv8 | kv4
                 kv_plan: str | None = None,       # kv4 plan json
                 draft: str | None = None,         # draft model path (enables spec)
                 spec_mode: str = "lookahead",     # chained | lookahead | dynamic
                 spec_gamma: int = 4):
        self.model_path = str(model_path)
        self.cfg = EngineConfig(model_path=Path(model_path), device=device)

        # ---- assemble the target engine (L3) with the KV strategy (L1) ----
        self._target = QslabEngine(self.cfg, kv_mode=kv_mode, kv_plan_path=kv_plan)

        # ---- optional W4 quantization (L1 backend mounted on L2 linears) ----
        self.w4 = w4
        if w4:
            from qslab.models.w4linear import swap_w4_linears
            n = swap_w4_linears(self._target.model, w4, backend=w4_backend)
            self._swapped = n

        # ---- optional speculative decoding (L3) ----
        self.draft_path = draft
        if draft:
            from qslab.engine.spec import SpeculativeEngine as _SE
            draft_cfg = EngineConfig(model_path=Path(draft), device=device)
            self._engine = _SE(self.cfg, draft_cfg, gamma=spec_gamma,
                               target_engine=self._target, mode=spec_mode)
        else:
            self._engine = self._target

        self._tokenizer = None

    # ------------------------------------------------------------------
    @property
    def tokenizer(self):
        """Lazy adapter (the only transformers-touching part of qslab)."""
        if self._tokenizer is None:
            from adapters.tokenizer import QwenTokenizerAdapter
            self._tokenizer = QwenTokenizerAdapter(self.model_path)
        return self._tokenizer

    @property
    def stats(self) -> dict:
        """Last-call statistics (acceptance rate etc. when speculating)."""
        return getattr(self._engine, "last_stats", {})

    def generate(self, prompt: str | list[int],
                 params: SamplingParams | None = None) -> dict:
        """Greedy decode (M0-M6 protocol). Returns {'text', 'token_ids', 'stats'}."""
        p = params or SamplingParams()
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if isinstance(self._engine, SpeculativeEngine):
            out = self._engine.generate(ids, max_new_tokens=p.max_tokens,
                                        eos_id=p.stop_token_id)
        else:
            out = self._engine.generate(ids, max_new_tokens=p.max_tokens,
                                        eos_id=p.stop_token_id)
        return {"text": self.tokenizer.decode(out), "token_ids": out,
                "stats": dict(self.stats)}

    def kv_memory_bytes(self) -> int:
        caches = self._target.kv_caches
        return sum(c.memory_bytes_valid() for c in caches)

    def weight_memory_bytes(self) -> int:
        total = 0
        for m in self._target.model.modules():
            fn = getattr(m, "weight_memory_bytes", None)
            if fn is not None:
                total += fn()
        return total
