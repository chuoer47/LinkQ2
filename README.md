# qslab

A lightweight LLM inference engine with a hand-built quantization stack, targeting a
single consumer GPU (RTX 4090, sm_89). Built as a from-scratch study of **what makes
4-bit inference fast and accurate, and where the limits are**.

Everything is measured, not asserted: benchmarks live in `benchmarks/` (organized by
topic, each with an evidence chain + repro commands), raw result files in `results/`,
and each milestone's write-up lives in `notes/` (M0–M9) or `docs/ARCHITECTURE.md`
(M10–M12) — including the experiments that did *not* work.

**Read this first: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — the single entry
point: layer map, the M0→M12 evolution (motivation → design → numbers → verdict, with
falsified hypotheses kept), current best numbers, known limitations, doc map.

## Headline results (Qwen3-8B, one RTX 4090 D, greedy)

| | tok/s | note |
|---|---|---|
| W4A16 + KV4 decode (new runtime + Marlin) | **97.2** | 1.75× over the engine's own FP16 |
| + n-gram speculation (γ=4, repetitive) | **310.4** | 3.19×; greedy-only, see caveat below |
| + draft model (γ=2, natural text) | 116.5 | 1.19× single-run; **n=5 repeat: 1.08×, band [1.02, 1.17]** — and pinning γ=2 costs the repetitive family 13% (1.61× vs 1.85×), which is why the shipped config is γ=4 + shrink-only window |
| prefix cache hit (3800-tok prefix) | TTFT **62.3 ms** | 8.92× vs cold (M10 re-run; matches the 09-18 verbal 63.0 ms within 1.1%) |
| W4 mainline weight residency (8B) | **3.34 GB** = 0.26× fp16 | Marlin's repack *replaces* the v1 pack; M10 released the 3.335 GB dead copy → KV pool 1.52→3.16 GB (+108%, measured — the earlier "+220%" assumed freed bytes land 1:1 in the pool, they don't) |
| context length axis (W4A16+KV4, YaRN) | 32K **13.3** · 64K **7.1** · 128K **3.7** | Each doubling costs ~×0.53 decode (KV-read-bound) and ×2.7–3.0 cold prefill (6.7 → 17.8 → 53.2 s) at 13.4 GB peak — `results/m11_yarn_niah.txt` |
| **concurrency axis**, 1.7B fp16+KV4 (M12) | bs1→32 **24.43×** (3572.8 tok/s decode window; 12.99× to bs=16 on 8B W4A16KV4) | Per-request 146 → 112 tok/s. The finding is a *negative* one: n-gram speculation **reverses** under concurrency on 1.7B (spec/off = 1.90× at bs=1 → 0.90× at bs=16 → 0.77× at bs=32) — `results/m12_batch_1.7b.txt` |
| vs **official vLLM 0.11** (same card, same prompt token ids, fp16↔fp16, wall clock) | **0.67–0.73× of vLLM**, flat across bs | Batch scaling curves coincide (24.43× vs 23.91×) → looks like a per-step constant cost, not a scheduler defect (not profiled, *unproven*). What 4-bit does buy is capacity: KV pool **1.79×** at equal utilization (135,936 vs 75,808 tokens) — `results/m12_vllm_1.7b.txt` |

> **Speculation numbers are greedy.** At `temperature>0` acceptance runs on the Leviathan
> ratio rule, which is distribution-lossless but changes the game: n-gram on repetitive text
> drops 3.06×→2.68× and lookahead on natural text loses everything (acceptance 0.62→0.00,
> tok/step → 1.00). Only the draft model holds ≈1.0×. Measured in
> `results/m10_spec_temperature.txt` — use γ≤2, or no speculation, when sampling.

> **Long context needs the same reading discipline.** At 128K the needle test collapses to
> 3/8 (vs 21/24 across the other tiers, Fisher p≈0.012) — but 4 of those 5 misses have the
> value's **leading 4–5 digits right and the tail dropped** (`92142` for 921426), and every
> answer terminates itself at 22–25 tokens. So the wall is at transcription, not at attention
> lookup; the same family already trips at 32K (`6363685` for 636685). What this round does
> *not* show: any YaRN effect on recall — the two 32K columns are the same code and needles
> and still shift a whole cell when the haystack phase changes, and there is no comparable
> 128K baseline (unscaled rope past 40960 is out-of-range extrapolation, not a control).

Quality: W4 PPL +1.24, KV4 PPL +0.369 (WikiText-2); weights 16G→2.5G (6.6×);
32K needle-in-a-haystack 100% for both FP16 and KV4 (M2/M4), and 8B W4A16+KV4 at
64K 7/8 / 128K 3/8 (M11, strict-match lower bound — see the caveat above).
Long-context PPL is structurally unavailable here: per-position logits at 131072
× vocab 151936 are ~40 GB.

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

```python
# Long context (M11): rope_scaling is a Config field, and the facade forwards
# unknown kwargs straight to Config — so 128K needs no new API surface.
llm = LLM("models/Qwen3-8B",
          w4="models/Qwen3-8B-qslab-w4-awq",
          smooth_kv="results/smooth_kv4_qwen3-8b.pt",
          max_model_len=131072,
          rope_scaling={"rope_type": "yarn", "factor": 3.2})   # 40960 × 3.2 = 131072
```

> `factor` must be > 1 (this field extends the ceiling, never shrinks it), and any
> `rope_type` other than `yarn` raises `NotImplementedError` rather than silently
> falling back to native rope — a length-extension run that quietly does nothing
> still finishes and still prints plausible numbers.

```bash
python -m qslab.api.cli generate --model models/Qwen3-8B \
  --w4 models/Qwen3-8B-qslab-w4-awq --smooth-kv results/smooth_kv4_qwen3-8b.pt \
  --spec ngram --spec-gamma 4 --device cuda:3 --max-tokens 64
```

```bash
# Service surface (M12): OpenAI-compatible, one pump thread owns the engine.
# Optional extra keeps the core package web-dependency-free:  pip install -e .[serve]
python -m qslab.api.server --model models/Qwen3-8B \
  --w4 models/Qwen3-8B-qslab-w4-awq --smooth-kv results/smooth_kv4_qwen3-8b.pt \
  --device 1 --port 8077 --max-num-seqs 16
curl -s localhost:8077/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24,"stream":true}'
# Measured (on the 1.7B model): 8 concurrent requests land in one batch —
# 384 tokens / 0.444 s = 865 tok/s wall, results/m12_service_smoke.txt.
# Known limits: a disconnect does not cancel, and there is no auth.
```

Low-level (what the benchmarks and step-level tests use — the facade is a thin wrapper
over this):

```python
from qslab.runtime.engine.llm_engine import LLMEngine
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
qslab/api/        L4 facade: LLM / SamplingParams / CLI, plus an OpenAI-compatible HTTP
                  server (M12, aiohttp behind the `.[serve]` extra — one pump thread owns
                  the engine so concurrency still forms real batches)
qslab/runtime/    the current engine (paged int4 KV, continuous batching, CUDA Graph,
                  n-gram / lookahead / draft speculation + adaptive gamma
                  (a one-way ratchet that only trims the proposal list — it cannot yet
                  save a draft forward, so its upside is structurally capped), prefix cache,
                  YaRN rope scaling for context past the checkpoint's ceiling)
qslab/kernels/    CUDA kernels (w4a16 GEMV, vendored Marlin) + Triton paged decode
qslab/quant/      packing format, W4 backends (v1/marlin/auto), quantizer, calibrators
qslab/models/     Qwen3 backbone + W4Linear (used by the legacy engine path)
qslab/engine/     legacy engine (M0-M7, frozen: oracle reference + lookahead/dynamic)
benchmarks/       evidence chains by topic (each dir: README + scripts)
results/          raw result files (evidence; not regenerated)
docs/             ARCHITECTURE.md (entry) + archive/ (historical designs)
notes/            milestone write-ups M0..M9 with raw numbers (M10+ is recorded in
                  docs/ARCHITECTURE.md + TODO.md instead, since it is measurement re-runs)
tests/            pytest suite (63 non-e2e + 41 e2e, 104 total, all green)
TODO.md           evidence-chain gaps (all closed by 09-19) + M12's new accounts 12–16
```
