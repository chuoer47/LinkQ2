# qslab

A lightweight LLM inference engine with a hand-built quantization stack, targeting a
single consumer GPU (RTX 4090, sm_89). Built as a from-scratch study of **what makes
4-bit inference fast and accurate, and where the limits are**.

Everything is measured, not asserted: benchmarks live in `benchmarks/` (organized by
topic, each with an evidence chain + repro commands), raw result files in `results/`,
and every milestone has a notes file — including the experiments that did *not* work.

**Read this first: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — the single entry
point: layer map, the M0→M10 evolution (motivation → design → numbers → verdict, with
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
from qslab.api import LLM, SamplingParams

llm = LLM("models/Qwen3-8B",
          w4="models/Qwen3-8B-qslab-w4-awq",
          smooth_kv="results/smooth_kv4_qwen3-8b.pt",
          spec="ngram", spec_gamma=4)        # or spec="draft", draft="models/Qwen3-0.6B"
res = llm.generate(" The capital of France is", SamplingParams(max_tokens=64))
print(res["text"], res["stats"])             # stats = steps/proposals/accepted/mean_len
```

```bash
python -m qslab.api.cli generate --model models/Qwen3-8B \
  --w4 models/Qwen3-8B-qslab-w4-awq --smooth-kv results/smooth_kv4_qwen3-8b.pt \
  --spec ngram --spec-gamma 4 --device cuda:3 --max-tokens 64
```

Low-level (what the benchmarks and step-level tests use — the facade is a thin wrapper
over this):

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

> Greedy is `temperature=1e-6` everywhere: the runtime always samples, and dividing the
> logits by 1e-6 turns the softmax into an argmax one-hot. The legacy M0–M7 engine stays
> reachable as `qslab.engine.QslabEngine` (oracle cross-checks) — see ARCHITECTURE.md §3.

## Repository map

```
qslab/api/        L4 facade: LLM / SamplingParams / CLI (wraps the runtime below)
qslab/runtime/    the current engine (paged int4 KV, continuous batching, CUDA Graph,
                  n-gram / lookahead / draft speculation + adaptive gamma, prefix cache)
qslab/kernels/    CUDA kernels (w4a16 GEMV, vendored Marlin) + Triton paged decode
qslab/quant/      packing format, W4 backends (v1/marlin/auto), quantizer, calibrators
qslab/models/     Qwen3 backbone + W4Linear (used by the legacy engine path)
qslab/engine/     legacy engine (M0-M7, frozen: oracle reference + lookahead/dynamic)
benchmarks/       evidence chains by topic (each dir: README + scripts)
results/          raw result files (evidence; not regenerated)
docs/             ARCHITECTURE.md (entry) + archive/ (historical designs)
notes/            milestone write-ups M0..M9 with raw numbers
tests/            pytest suite (43 non-e2e + 38 e2e, 81 total, all green)
TODO.md           evidence-chain gaps + remaining work
```
