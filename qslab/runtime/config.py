import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    rope_scaling: dict | None = None  # {"rope_type": "yarn", "factor": 3.2} → 128K
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 128
    num_kvcache_blocks: int = -1
    v_group: int = 64              # V quant group size along head_dim
    smooth_kv: str | None = None   # SmoothAttention calibration file
    w4: str | None = None          # qslab_w4_v1 dir; swaps in packed weights
    w4_backend: str = "w4.auto"    # w4.v1 / w4.marlin / w4.auto
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
    draft_w4: str | None = None          # packed W4 dir for the draft model
    draft_w4_backend: str = "w4.v1"     # v1 GEMV is built for M=1 decode
    compile: bool = False                # torch.compile the runtime model
    compile_mode: str = "default"        # "default" | "max-autotune"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 128 == 0  # qslab: KV4 block alignment
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
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
