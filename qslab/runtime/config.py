import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
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
    # speculative decoding (design-m9). n-gram only for now: the proposer is
    # a CPU lookup, the verify forward reuses the paged decode kernel with
    # M = spec_num_drafts + 1 queries per sequence
    spec_method: str | None = None        # None | "ngram" | "draft"
    spec_num_drafts: int = 4              # gamma
    spec_ngram_size: int = 3              # lookup window n
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
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
