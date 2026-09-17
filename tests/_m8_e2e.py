"""M8-S2: end-to-end generation through the nano-vllm runtime with qslab W4."""
import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")

import torch
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

MODEL = "/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B"

# HF oracle
from qslab.models.loader import load_reference_model
from adapters.tokenizer import QwenTokenizerAdapter
tok = QwenTokenizerAdapter(MODEL)
ids = tok.encode(" The capital of France is")
ref = load_reference_model(MODEL)
with torch.inference_mode():
    o = ref.generate(torch.tensor([ids], device="cuda:0"), max_new_tokens=16,
                     do_sample=False)
ref_tokens = o[0][len(ids):].tolist()
ref_text = tok.decode(ref_tokens)
print("HF oracle :", ref_tokens, repr(ref_text[:40]))
del ref
torch.cuda.empty_cache()

engine = LLMEngine(model=MODEL, max_model_len=4096, max_num_seqs=8,
                   enforce_eager=True, gpu_memory_utilization=0.6)
print("engine up; kv blocks:", engine.model_runner.config.num_kvcache_blocks)

outs = engine.generate([" The capital of France is"],
                       SamplingParams(temperature=1e-6, max_tokens=16),
                       use_tqdm=False)
got_tokens = outs[0]["token_ids"]
print("runtime   :", got_tokens, repr(outs[0]["text"][:40]))
print("LOSSLESS MATCH:", got_tokens == ref_tokens)
