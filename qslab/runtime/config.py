import os
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    rope_scaling: dict | None = None  # {"rope_type": "yarn", "factor": 3.2} → 128K
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "fp16"    # fp16 | kv4; independent of weight quantization
    v_group: int = 64              # V quant group size along head_dim
    smooth_kv: str | None = None   # SmoothAttention calibration file
    quant_backend: str = "gemlite"    # local fused W4A16 backend
    # speculative decoding: the verify forward reuses the paged decode kernel with
    #   spec_num_drafts + 1 queries per sequence
    spec_method: str | None = None        # None | "ngram" | "lookahead" | "draft"
    spec_num_drafts: int = 4              # gamma
    spec_ngram_size: int = 3              # lookup window n
    # shrink/raise the draft window per sequence from its recent acceptance
    # (only helps the proposers that pay per draft: "draft")
    spec_adaptive_gamma: bool = False
    spec_lookahead_span: int = 8          # "lookahead": indexed tokens per commit
    draft_model: str | None = None        # spec_method="draft": small model
    draft_gpu_memory_utilization: float = 0.95   # draft pool sizing
    draft_quant_backend: str = "gemlite"     # fused backend for the draft model
    compile: bool = False                # torch.compile the runtime model
    compile_mode: str = "default"        # "default" | "max-autotune"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if self.kv_cache_dtype not in {"fp16", "kv4"}:
            raise ValueError('kv_cache_dtype must be fp16 or kv4')
        if self.kv_cache_dtype == 'kv4' and not self.smooth_kv:
            raise ValueError('KV4 requires smooth_kv calibration with positive static K scales')
        if self.smooth_kv and not Path(self.smooth_kv).is_file():
            raise FileNotFoundError(self.smooth_kv)
        alignment = 256 if self.kv_cache_dtype == 'fp16' else 128
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % alignment:
            raise ValueError(f'kvcache_block_size must be a positive multiple of {alignment}')
        if min(self.max_model_len, self.max_num_seqs, self.max_num_batched_tokens) <= 0:
            raise ValueError('context and batch limits must be positive')
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError('gpu_memory_utilization must be in (0, 1]')
        self.hf_config = AutoConfig.from_pretrained(self.model)
        head_dim = getattr(self.hf_config, 'head_dim',
                           self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        if (self.v_group < 8 or self.v_group & (self.v_group - 1)
                or head_dim % self.v_group):
            raise ValueError('v_group must be a power of two >= 8 and divide head_dim')
        if self.rope_scaling:
            scaling = dict(self.rope_scaling)
            native = int(self.hf_config.max_position_embeddings)
            factor = float(scaling.get("factor", 1.0))
            assert factor > 1.0, "rope_scaling only extends; leave it None to stay native"
            scaling.setdefault("rope_type", scaling.pop("type", "yarn"))
            # HF's BC alias, so `rope_type` below is the only name that matters
            scaling.pop("type", None)
            original = int(scaling.get("original_max_position_embeddings") or native)
            scaling["original_max_position_embeddings"] = original
            # A YaRN release ships exactly this: the extended
            # max_position_embeddings plus the rope dict that reaches it.
            # Rewriting hf_config (not just our own field) keeps the rope cache
            # size and the clamp below on one ceiling.
            self.hf_config.rope_scaling = scaling
            self.hf_config.max_position_embeddings = int(original * factor)
            self.rope_scaling = scaling
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
