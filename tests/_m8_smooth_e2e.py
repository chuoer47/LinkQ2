"""M8-S2 end-to-end with SmoothAttention + static-K KV4."""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
import torch
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

MODEL = "/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B"
CKPT = "/home/<user>/<workdir>/qserve-lab/results/smooth_kv4_qwen3-1.7b.pt"
ORACLE = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315, 15344, 374, 21718, 13]

engine = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                   enforce_eager=True, gpu_memory_utilization=0.5,
                   smooth_kv=CKPT)
print("kv blocks:", engine.model_runner.config.num_kvcache_blocks)
out = engine.generate([" The capital of France is"],
                      SamplingParams(temperature=1e-6, max_tokens=16),
                      use_tqdm=False)
got = out[0]["token_ids"]
print("runtime:", got)
print("oracle :", ORACLE)
print("LOSSLESS MATCH:", got == ORACLE)
