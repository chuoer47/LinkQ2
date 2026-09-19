"""vLLM baseline: what an off-the-shelf engine costs on the same box.

Protocol (deliberately narrow — see benchmarks/10-vllm-compare/README.md)
----------------------------------------------------------------------
Same card, same fp16 Qwen3-1.7B checkpoint, same prompt token ids, greedy,
ignore_eos, 256 output tokens, max_num_seqs=32, gpu_memory_utilization=0.5.
Throughput is reported BOTH ways: wall (prefill included, the only protocol
vLLM's offline API exposes) and vLLM's own generated-token count. The
decode-window column exists only for our runtime (benchmarks/09-batch-throughput).

What this is NOT
----------------
* Not a quantization comparison. vLLM 0.11 has AWQ/GPTQ W4A16 kernels but no
  int4 KV cache at all, and this project's W4 packs are its own
  `qslab_w4_v1` format (models/*.origin.txt: packed from the local fp16
  checkpoint), which vLLM cannot load. So the weight-precision-matched cell
  is fp16 <-> fp16, and KV4 has no counterpart to compare against.
* Not a kernel comparison. Our Marlin path is vendored from upstream and our
  paged decode shares ancestry with vLLM's, so a kernel-level win/loss is not
  what this measures; this measures engine/scheduler/CUDA-graph overhead.
* No per-token equality checks anywhere: 4-bit greedy is chaotic, the only
  admissible metrics are PPL / throughput / acceptance rate.

Prompts are rebuilt here with the same RNG (seed 11, randint(1000, 200000) %
vocab, one per request, cells iterated ascending by bs) as
bench_batch.py:distinct_prompts, so the two engines see identical token ids.

Env: MODEL TOKENS BATCHES UTIL
Sampler: VLLM_USE_FLASHINFER_SAMPLER is forced to 0 (see below).
"""
import os
import random
import sys
import time
from pathlib import Path

# vLLM 0.11 routes top-k/top-p sampling through flashinfer, whose kernels are
# JIT-compiled at first use and need nvcc. The dedicated `vllm` conda env ships
# no nvcc and there is no /usr/local/cuda on this box, so engine init dies in
# profile_run -> _dummy_sampler_run with "Could not find nvcc". The sampler is
# not what this baseline measures (engine / scheduler / CUDA-graph overhead), so
# pin vLLM onto its native torch sampler instead of dragging a CUDA toolkit in.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("MODEL", "./models/Qwen3-1.7B")
N_TOKENS = int(os.environ.get("TOKENS", "256"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,2,4,8,16,32").split(",")]
UTIL = float(os.environ.get("UTIL", "0.5"))

COPY_PROMPT = (" The capital of France is Paris. The capital of Germany is Berlin. "
               "The capital of Italy is Rome. The capital of Spain is Madrid. "
               "The capital of France is")


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    body = tok.encode(COPY_PROMPT)
    kw = dict(dtype="float16", max_model_len=4096, max_num_seqs=max(BATCHES),
              gpu_memory_utilization=UTIL, enforce_eager=False, seed=0)
    print(f"=== vllm baseline: {Path(MODEL).name} tokens={N_TOKENS} "
          f"prompt_len={len(body)} util={UTIL} max_num_seqs={max(BATCHES)} ===")
    llm = LLM(model=MODEL, **kw)
    # vLLM 0.11 exposes the engine config only through the LLMEngine wrapper.
    vcfg = llm.llm_engine.vllm_config
    mcfg, ccfg = vcfg.model_config, vcfg.cache_config
    print(f"[vllm] py={sys.version.split()[0]} version={__import__('vllm').__version__} "
          f"quantization={mcfg.quantization} dtype={mcfg.dtype} "
          f"block_size={ccfg.block_size} kv_cache_dtype={ccfg.cache_dtype} "
          f"prefix_caching={ccfg.enable_prefix_caching} "
          f"enforce_eager={mcfg.enforce_eager} "
          f"max_num_batched_tokens={vcfg.scheduler_config.max_num_batched_tokens}")

    sp = SamplingParams(temperature=0.0, max_tokens=N_TOKENS, ignore_eos=True)
    rng = random.Random(11)

    def prompts(n):
        used = set()
        out = []
        for _ in range(n):
            while True:
                head = rng.randint(1000, 200000) % tok.vocab_size
                if head not in used:
                    used.add(head)
                    break
            out.append({"prompt_token_ids": [head] + body})
        return out

    # warm-up call outside the measured cells (vLLM compiles/captures lazily)
    llm.generate(prompts(1), SamplingParams(temperature=0.0, max_tokens=8,
                                            ignore_eos=True), use_tqdm=False)
    print(f"{'bs':>4} {'tok/s(wall)':>12} {'tok/s/req':>10} {'wall_s':>8}")
    rows = []
    for bs in BATCHES:
        p = prompts(bs)
        t = time.perf_counter()
        outs = llm.generate(p, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
        agg = n_gen / dt
        rows.append((bs, agg))
        print(f"{bs:>4} {agg:12.1f} {agg / bs:10.1f} {dt:8.2f}")
    base = rows[0][1]
    print("speedup vs bs=1: " + " ".join(f"bs{b}={a / base:.2f}x" for b, a in rows[1:]))


if __name__ == "__main__":
    main()
