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
| + n-gram speculation (γ=4, repetitive) | **310.4** | 3.19×; greedy-only, see caveat below |
| + draft model (γ=2, natural text) | 116.5 | ~1.19×; M10 reproduced this cell, γ≥3 sits in a noise band |
| prefix cache hit (3800-tok prefix) | TTFT **62.3 ms** | 8.92× vs cold (M10 re-run; matches the 09-18 verbal 63.0 ms within 1.1%) |
| W4 mainline weight residency (8B) | **3.34 GB** = 0.26× fp16 | Marlin's repack *replaces* the v1 pack; M10 released the 3.335 GB dead copy → KV pool 1.52→3.16 GB (+108%, measured — the earlier "+220%" assumed freed bytes land 1:1 in the pool, they don't) |

> **Speculation numbers are greedy.** At `temperature>0` acceptance runs on the Leviathan
> ratio rule, which is distribution-lossless but changes the game: n-gram on repetitive text
> drops 3.06×→2.68× and lookahead on natural text loses everything (acceptance 0.62→0.00,
> tok/step → 1.00). Only the draft model holds ≈1.0×. Measured in
> `results/m10_spec_temperature.txt` — use γ≤2, or no speculation, when sampling.

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
                  n-gram / lookahead / draft speculation + adaptive gamma
                  (a one-way ratchet: the window shrinks, it never grows back), prefix cache)
qslab/kernels/    CUDA kernels (w4a16 GEMV, vendored Marlin) + Triton paged decode
qslab/quant/      packing format, W4 backends (v1/marlin/auto), quantizer, calibrators
qslab/models/     Qwen3 backbone + W4Linear (used by the legacy engine path)
qslab/engine/     legacy engine (M0-M7, frozen: oracle reference + lookahead/dynamic)
benchmarks/       evidence chains by topic (each dir: README + scripts)
results/          raw result files (evidence; not regenerated)
docs/             ARCHITECTURE.md (entry) + archive/ (historical designs)
notes/            milestone write-ups M0..M9 with raw numbers
tests/            pytest suite (43 non-e2e + 38 e2e, 81 total, all green)
TODO.md           evidence-chain gaps (all six closed by 09-19) + remaining functional work
```
