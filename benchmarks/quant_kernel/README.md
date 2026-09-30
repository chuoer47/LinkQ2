# Quant Kernel Correctness and Performance

This benchmark compares quantized linear kernels against an FP32 reference built from
the same logical quantized weights. It records numerical error and operator latency.
It does not load a language model or measure PPL.

The matrix covers the project W4A16 kernel and Marlin on symmetric W4/group size 128,
plus Nunchaku AWQ GEMV on affine W4/group size 64. AWQ activation scaling is applied to
the input in both the kernel path and reference. Nunchaku batches larger than eight rows
are split into supported GEMV calls and measured as one operator call.

Run after activating `qslab`:

```bash
python -m benchmarks.quant_kernel.run --device cuda:0
```

For the Nunchaku row, install a wheel built for the environment's Python, PyTorch, and
CUDA versions. The validated server environment used Python 3.11, PyTorch 2.5.1+cu124,
and Nunchaku 0.3.0+torch2.5, with `diffusers` and Pillow available for its package imports.
The upstream release wheel is available from
`https://github.com/nunchux-ai/nunchaku/releases/tag/v0.3.0`.

Select one backend or case with `--backend` and `--case`. `--warmup` and `--repetitions`
control timing. Results are written to `results/quant_kernel/summary.json`. Steady-state
operator latency uses synchronized host wall-clock timing for every backend, keeping the
Nunchaku extension's stream-buffer behavior compatible and the comparison consistent.
Preparation time and first-call time are reported separately. The result records GPU
memory and utilization before and after the run and flags timings collected while the GPU
was already occupied; those timings should be treated as shared-device observations.
