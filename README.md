# qslab

A lightweight LLM inference engine with a hand-built quantization stack, targeting a
single consumer GPU (RTX 4090, sm_89). Built as a from-scratch study of **what makes
4-bit inference fast and accurate, and where the limits are**.

Everything is measured, not asserted: every milestone has a notes file with the raw
numbers, including the experiments that did *not* work out.

---

## Results (Qwen3-8B, one RTX 4090)

| Configuration | tok/s | PPL (WikiText-2) | KV cache @512 | Weight memory |
|---|---|---|---|---|
| FP16 baseline | 42.4 | 16.07 | 94.2 MB | 16.4 GB |
| **W4A16** | 39.8 | 17.31 (+1.24) | 94.2 MB | **2.49 GB (6.6×)** |
| **+ KV4 cache** | — | 17.70 (+1.63) | **27.0 MB (3.5×)** | 2.49 GB |
| **+ lookup speculation** | 56.6 (**1.44×**) | — | 27.0 MB | 2.49 GB |

Also validated: 32K-context needle-in-a-haystack recall 100% for both FP16 and KV4
caches; losslessness of all three speculation modes (token-identical to greedy target
decoding); a three-way kernel benchmark (custom / Marlin / cuBLAS) over the real
model's shapes.

---

## Architecture (five layers, strictly one-way dependencies)

```
L4  qslab/api/         LLM facade + CLI            "user entry point"
L3  qslab/engine/      decode loop, speculative decoding
L2  qslab/models/      Qwen3 backbone, patched attention, W4 Linear
L1  qslab/quant/       algorithms, packing format, KV cache, pluggable backends
L0  qslab/kernels/     CUDA kernels (custom W4A16 GEMV, vendored Marlin)
         adapters/     the ONLY place that imports transformers
```

**The rule:** a layer may only import the layer directly below it. `L2` never touches
`L0` — quantized linears call `L1`'s `QuantBackend.linear(x)` and are unaware a CUDA
kernel exists. This is enforced by convention and checked in review; see
`docs/design-r1.md`.

### Three pluggable strategies (registry + factory)

| Interface | Layer | Implementations |
|---|---|---|
| `QuantBackend` | L1 | `w4.v1` (custom GEMV), `w4.marlin` (tensor-core GEMM), `w4.auto` (hybrid, switches at M>8) |
| `KVCacheStrategy` | L1 | `fp16`, `kv8`, `kv4`, `kv4.plan` (per-layer FP16 override) |
| `SpeculationMode` | L3 | `chained` (draft model), `lookahead` (n-gram self-proposal), `dynamic` (adaptive γ) |

Adding an implementation is one class plus a `@register("<name>")` decorator. The
`KVCacheStrategy` interface has a **reserved paged slot** for the roadmap below.

---

## Quick start

```bash
conda env create -f environment.yml && conda activate qslab
# flash-attn ships as a prebuilt wheel — see requirements.txt for the URL

python -m qslab.api.cli generate \
  --model models/Qwen3-8B \
  --w4 models/Qwen3-8B-qslab-w4-awq \
  --prompt " The capital of France is" --max-tokens 32
```

Or from Python:

```python
from qslab.api import LLM, SamplingParams

llm = LLM("models/Qwen3-8B",
          w4="models/Qwen3-8B-qslab-w4-awq",   # weight backend
          kv_mode="kv4", kv_plan="results/kv4_plan_8b.json",
          draft="models/Qwen3-0.6B", spec_mode="lookahead")
print(llm.generate(" The capital of France is", SamplingParams(max_tokens=32))["text"])
```

The three capabilities compose independently: `w4` (weights), `kv_mode` (cache),
`draft` (speculation).

---

## The quantization stack

**Packing format `qslab_w4_v1`** — 8 signed int4 values per `uint32`, one FP16 scale
per 128-wide group. Symmetric in v1 (`zero` field reserved), no weight shuffle.
`qslab/quant/packfmt.py`.

**W4A16 GEMM** — two backends, benchmark-selected per shape:

- `w4.v1`: a custom CUDA GEMV. Unpacks nibbles in registers and dots with FP16
  activations. Best at M=1 (decode), where it beats cuBLAS FP16.
- `w4.marlin`: the vendored [Marlin](https://github.com/IST-DASLab/marlin) kernel
  (Apache-2.0) for tensor-core GEMM, fed by a converter from our packing format.
  Best from M≥8.
- `w4.auto`: dispatches between them at M>8 (the crossover measured in
  `results/m5_kernel_bench.json`).

**KV4 cache** — KIVI-style asymmetric quantization: keys per-channel (stored
transposed), values per-token, group 64, with the first layer kept in FP16 (its key
outliers measure at ~350× the median). `qslab/quant/cache/kv_cache.py`.

**Weight quantization algorithms** — RTN (group-wise) and AWQ (activation-aware
scaling with per-group MSE-clip search), plus the calibration pipeline.
`qslab/quant/{w4,calibrate,quantize}.py`.

---

## Speculative decoding

All three modes share one verification/rollback core; they differ only in where
proposals come from.

| Mode | Proposal source | Measured (8B W4, repetitive text) |
|---|---|---|
| `lookahead` | n-gram match in the prompt+generated text — zero model cost | **1.44×** |
| `chained` | a Qwen3-0.6B draft model | 0.72× |
| `dynamic` | chained with adaptive γ | 0.76× |

`chained`'s sub-1× result is a finding, not a bug: our Python engine's per-forward
dispatch cost dominates at 8B scale. The notes file works through the accounting.

Every mode is **provably lossless** (each accepted token is verified against the
target's own argmax) and tested as such.

---

## Tests

```bash
pytest tests/ -v                    # all 19
pytest tests/ -m "not e2e"          # skip the model-dependent ones
```

| Layer | Tests | What they pin down |
|---|---|---|
| CPU (2s) | 9 | packing roundtrip ≤ half a quant step, registry semantics, rejection-sampling distribution matches the target law |
| GPU (2s) | 6 | custom kernel ≡ dequantized matmul, per-precision cache error bounds, KV4 memory >3× smaller, odd-length appends |
| e2e (24s) | 4 | engine ≡ HF greedy oracle, all three speculation modes lossless |

---

## Repository map

```
qslab/           the engine (five layers, see above)
adapters/        tokenizer + HF model builder (only import site for transformers)
quantizer/       offline quantization tool (HF checkpoint -> qslab_w4_v1)
benchmarks/      reusable measurements (throughput, PPL, kernel, speculation, NIAH)
tests/           pytest suite
notes/           eight milestone write-ups with raw numbers
docs/            design decisions and the milestone protocol
results/         committed result JSONs (logs and checkpoints are gitignored)
scripts/         environment/kernel build scripts; archive/ holds one-off experiments
third_party/     cutlass + marlin checkouts (gitignored)
```

---

## Roadmap

The engine today is **batch=1, decode-only**. Two projects would extend it:

**1. Quantized paged attention (M7).** The current KV4 cache dequantizes the whole
sequence to FP16 on every read, which costs O(context) bandwidth per step and
makes the transient FP16 copy the memory peak at long context. The fix is an
attention kernel that reads the packed format directly — a Triton paged-attention
skeleton with a ~15-line dequantization step in the tile load, following KIVI's
approach. It also needs a layout decision: per-channel K groups span paged blocks,
so either regroup per-block (small accuracy cost) or keep cross-block scales.
The `KVCacheStrategy` interface already has the slot for it.

**2. nano-vllm integration.** A ~1.3k-line reimplementation of vLLM is checked out
under `.cache/nano-vllm-main`. It supplies the engine mechanics we lack — paged KV
with slot-mapping writes, CUDA-graph capture, continuous batching, FlashAttention.
The plan is to keep our L1 quantization stack and L2 model definitions and swap in
its L3 runtime, which is where the CUDA-graph problem (dynamic KV write positions)
disappears by construction.

See `docs/design-r1.md` for the layering rules these would need to respect.
