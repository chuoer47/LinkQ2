"""qslab engine: decode-only, batch=1 inference loop.

The engine owns:
- the model (transformers Qwen3 backbone, attention replaced by PatchedQwen3Attention)
- the per-layer KV caches
- the decode loop: prefill (one forward over the prompt) then token-by-token decode

M2: kv_mode selects cache precision per layer ("fp16" | "kv8" | "kv4"), with
a plan json optionally overriding which layers stay fp16 (kv4_plan.json).
"""
from __future__ import annotations

import json

import torch

from qslab.config import EngineConfig
from qslab.model.loader import load_reference_model, load_model_config
from qslab.model.patched import patch_model


class QslabEngine:
    def __init__(self, cfg: EngineConfig, kv_mode: str = "fp16",
                 kv_plan_path: str | None = None):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = getattr(torch, cfg.dtype)

        self.model_cfg = load_model_config(cfg.model_path)
        self.model = load_reference_model(cfg.model_path, device=self.device)

        # per-layer cache classes (M2): default all fp16, plan overrides
        from qslab.cache.kv_cache import FP16KVCache, KV8Cache, KV4Cache
        classes = {"fp16": FP16KVCache, "kv8": KV8Cache, "kv4": KV4Cache}
        assert kv_mode in classes, f"unknown kv_mode {kv_mode}"
        fp16_layers: set[int] = set()
        kv_group = 64
        if kv_plan_path:
            plan = json.loads(open(kv_plan_path).read())
            fp16_layers = set(plan.get("kv_fp16_layers", []))
            kv_group = plan.get("group_size", 64)

        cache_classes = []
        for i in range(self.model_cfg.num_hidden_layers):
            if kv_mode == "fp16" or i in fp16_layers:
                cache_classes.append(FP16KVCache)
            else:
                cache_classes.append(classes[kv_mode])

        # replace attention with engine-managed cache version; patch_model
        # accepts a factory so each layer gets its own cache class
        self.kv_caches = patch_model(
            self.model,
            num_layers=self.model_cfg.num_hidden_layers,
            num_kv_heads=self.model_cfg.num_key_value_heads,
            head_dim=self.model_cfg.head_dim,
            max_len=self.model_cfg.max_position_embeddings,
            device=self.device,
            cache_factory=lambda i: cache_classes[i](
                batch=1,
                num_kv_heads=self.model_cfg.num_key_value_heads,
                head_dim=self.model_cfg.head_dim,
                max_len=self.model_cfg.max_position_embeddings,
                device=self.device,
                **({} if cache_classes[i] is FP16KVCache else {"group": kv_group}),
            ),
        )
        self.model.eval()

    def reset_cache(self):
        """Zero all engine KV caches and reset lengths (between generations)."""
        for c in self.kv_caches:
            c.k.zero_()
            c.v.zero_()
            c.len = 0

    @torch.inference_mode()
    def prefill(self, input_ids: list[int]) -> torch.Tensor:
        """Run the prompt through the model once; return logits of last position."""
        x = torch.tensor([input_ids], device=self.device)
        out = self.model(input_ids=x, use_cache=False)
        return out.logits[:, -1, :]

    @torch.inference_mode()
    def decode_step(self, token_id: int, start_pos: int) -> torch.Tensor:
        """One decode step: feed the last token, read KV from engine caches.

        Since our PatchedQwen3Attention reads/writes the engine cache directly
        (which already holds positions [0, start_pos)), we feed only the new
        token. `cache_position` tells the backbone the true position so RoPE
        cos/sin are computed for the right index.
        Returns logits for the next token.
        """
        x = torch.tensor([[token_id]], device=self.device)
        cache_pos = torch.tensor([start_pos], device=self.device, dtype=torch.long)
        out = self.model(input_ids=x, cache_position=cache_pos, use_cache=False)
        return out.logits[:, -1, :]

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int | None = None,
                 eos_id: int | None = None) -> list[int]:
        """Greedy decode. Returns generated token ids (excluding prompt)."""
        self.reset_cache()
        max_new = max_new_tokens or self.cfg.max_new_tokens
        logits = self.prefill(input_ids)
        next_tok = int(logits.argmax(dim=-1))
        generated = [next_tok]

        for _ in range(max_new - 1):
            if eos_id is not None and next_tok == eos_id:
                break
            logits = self.decode_step(next_tok, start_pos=len(input_ids) + len(generated) - 1)
            next_tok = int(logits.argmax(dim=-1))
            generated.append(next_tok)
        return generated

    def kv_memory_bytes(self) -> int:
        return sum(c.memory_bytes_valid() for c in self.kv_caches)
