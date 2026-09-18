# benchmarks — 性能证据链索引

按主题分类。每类目录内的 `README.md` = 证据链（数字 → results 原始文件 → 结论）+ 复现指令。
`results/` 原始文件**保留不动**，是证据底稿。

## 复现的环境前置（全部脚本通用）

```bash
# 服务器 4090_public，conda env: qslab；所有命令在仓库根目录执行（脚本按 cwd 写 results/）
conda activate qslab    # 或直接用 <conda-env>/bin/python
# 涉及 W4/marlin 扩展时（扩展已编译则可省）：完整工具链环境变量不可少，
# 系统 gcc>13 会报 nvcc 错误——见 docs/archive/02-开发环境.md 与 scripts/build_kernels.sh
export PATH=<conda-env>/bin:$PATH
export CONDA_PREFIX=<conda-env>
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

## 目录

| 目录 | 主题 | 里程碑 | 脚本 |
|---|---|---|---|
| `01-decode-baseline/` | decode 吞吐基线 + NIAH 检索 | M0/M2/M4 | bench_throughput, bench_niah |
| `02-w4-quant/` | W4 kernel 三方对比（v1/Marlin/cuBLAS） | M5 | bench_kernel |
| `03-kv4-quality/` | PPL 主线 + KV4 代价（HF 模拟法） | M0-M4/M8 | bench_ppl, bench_ppl_kv |
| `04-runtime-m8/` | 8B 四方矩阵 + paged 验收 + runtime PPL | M4/M7/M8 | bench_engine_matrix, m7_acceptance, bench_ppl_runtime |
| `05-spec-ngram/` | n-gram 投机（新 runtime） | M9 | bench_spec_ngram |
| `06-spec-draft/` | draft 投机（新 runtime） | M9 | ⚠ 缺入库脚本（见其 README） |
| `07-prefix-cache/` | 前缀缓存 TTFT | M9 | bench_prefix_cache |
| `archive-legacy/` | 旧引擎 spec bench（M3/M6 口径） | M3/M6 | bench_spec, bench_spec_modes |

## 关键结果速查（→ 底稿文件）

| 结论 | 数字 | 底稿 |
|---|---|---|
| 8B FP16 基线 | 42.4 tok/s | results/m4_matrix_fp16.json |
| 8B W4A16KV4 decode（新 runtime+Marlin） | 97.2 tok/s（1.75×） | results/m9_spec_8b.txt（off 行） |
| n-gram 复读 γ=4 | 310.4 tok/s（3.19×） | results/m9_spec_8b.txt |
| draft γ=8 复读（融合后） | 225.1 tok/s（2.30×） | results/m9_gamma_sweep_draft.txt |
| draft γ=2 自然（融合后） | 114.4 tok/s（1.17×） | results/m9_draft_fused.txt |
| 前缀缓存 8B TTFT | 558.3→63.0ms（8.86×） | results/m9_prefix_cache.txt |
| KV4 PPL 代价 | +0.369 | results/m8_kv4_ppl.json |
| W4 kernel 三方 | M≥8 Marlin 碾压 | results/m5_kernel_bench.json |
