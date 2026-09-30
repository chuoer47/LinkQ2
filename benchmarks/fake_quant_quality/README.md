# Fake Quant Quality

This benchmark measures language-model PPL after weight fake quantization. It does not
export packed deployment weights or call qslab, Marlin, or Nunchaku runtime kernels.

The default run compares the FP16 Qwen3-8B baseline with `rtn`, `rtn_clip`, `awq`, and
`nunchaku_awq`. RTN variants and AWQ use group size 128; Nunchaku AWQ uses group size 64.
AWQ variants collect activation statistics from 128 C4 sequences of 2048 tokens. PPL is
computed on the WikiText-2 raw test split with 2048-token windows and stride 512. Each
target token is scored once, including tokens in overlapping windows.

Run from the repository root in the activated qslab environment:

```bash
python -m benchmarks.fake_quant_quality.run --device cuda:0
```

Select algorithms or paths with `--algorithms`, `--model`, and `--output`. Data token IDs are cached beneath `datasets/fake_quant_quality/`; metric JSON files are written
beneath `results/fake_quant_quality/`. Each algorithm runs in a separate child process so
the model and its CUDA allocations are released before the next algorithm starts.
