# qslab

A lightweight LLM inference engine with a hand-built quantization stack, targeting a
single consumer GPU (RTX 4090, sm_89). Built as a from-scratch study of **what makes
4-bit inference fast and accurate, and where the limits are**.

Everything is measured, not asserted: benchmarks live in `benchmarks/` (organized by
topic, each with an evidence chain + repro commands), raw result files in `results/`,
and every milestone has a notes file — including the experiments that did *not* work.

**Read this first: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — the single entry
point: layer map, the M0→M9 evolution (motivation → design → numbers → verdict, with
falsified hypotheses kept), current best numbers, known limitations, doc map.

## Headline results (Qwen3-8B, one RTX 4090 D, greedy)

| | tok/s | note |
|---|---|---|
| W4A16 + KV4 decode (new runtime + Marlin) | **97.2** | 1.75× over the engine's own FP16 |
| + n-gram speculation (γ=4, repetitive) | **310.4** | 3.19×; token-identical to plain greedy |
| + draft model (γ=2, natural text) | 114.4 | 1.17×; n-gram owns repetitive, draft owns natural |
| prefix cache hit (3800-tok prefix) | TTFT **63.0 ms** | 8.86× vs cold |

Quality: W4 PPL +1.24, KV4 PPL +0.369 (WikiText-2); weights 16G→2.5G (6.6×);
32K needle-in-a-haystack 100% for both FP16 and KV4.

## Quick start

```python
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams

eng = LLMEngine(model="models/Qwen3-8B",
                w4="models/Qwen3-8B-qslab-w4-awq",
                spec_method="ngram", spec_num_drafts=4)   # or spec_method="draft"
eng.add_request("The capital of France is", SamplingParams(temperature=1e-6, max_tokens=64))
while not eng.is_finished():
    out, n = eng.step()
```

> Note: `qslab/api/llm.py` (LLM/CLI facade) still points at the legacy engine
> (`qslab/engine`, M0–M7). The runtime above is the current main line — see
> ARCHITECTURE.md §3 for the two-engine situation.

## Repository map

```
qslab/runtime/    the current engine (paged int4 KV, continuous batching, CUDA Graph,
                  n-gram + draft speculation, prefix cache)
qslab/kernels/    CUDA kernels (w4a16 GEMV, vendored Marlin) + Triton paged decode
qslab/quant/      packing format, W4 backends (v1/marlin/auto), quantizer, calibrators
qslab/models/     Qwen3 backbone + W4Linear (used by the legacy engine path)
qslab/engine/     legacy engine (M0-M7, frozen: oracle reference + lookahead/dynamic)
benchmarks/       evidence chains by topic (each dir: README + scripts)
results/          raw result files (evidence; not regenerated)
docs/             ARCHITECTURE.md (entry) + archive/ (historical designs)
notes/            milestone write-ups M0..M9 with raw numbers
tests/            pytest suite (32 non-e2e + 27 e2e + 2 8B, all green)
TODO.md           evidence-chain gaps + remaining work
```
