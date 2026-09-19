# benchmarks — 性能证据链索引

按主题分类。每类目录内的 `README.md` = 证据链（数字 → results 原始文件 → 结论）+ 复现指令。
`results/` 原始文件**保留不动**，是证据底稿。

**底稿怎么来的**：多数脚本只往 stdout 打印，落盘方式是"命令行 + 原始输出"一起重定向进
`results/m<里程碑>_*.txt`（例：`m9_prefix_cache.txt`、`m11_yarn_niah.txt`）。因此每份底稿
开头都能直接复制出命令；改脚本时**不要**让它去覆盖已有底稿。

## 复现的环境前置（全部脚本通用）

```bash
# 服务器 4090_public，conda env: qslab；所有命令在仓库根目录执行（脚本按 cwd 写 results/）
conda activate qslab
# ⚠ 跑之前先看 nvidia-smi 挑空卡（本项目全程单卡跑）。0 号与 2 号是别人的常驻作业，
#   历史上所有底稿用的都是 1 号或 3 号：CUDA_VISIBLE_DEVICES=1/3
# ⚠ 只把 env/bin 塞进 PATH、不 activate 是不够的（M10 踩过一次，8B W4 直接跑挂）：
#   CUDA_HOME/CC/CXX 未设时 torch 会把 marlin 扩展当没编译过，重新 JIT 到新的缓存目录，
#   然后用系统 gcc(>13) 编译失败（还报 cusparse.h 找不到）。纯 fp16 脚本感觉不到，
#   碰 W4 的脚本必须 activate。完整工具链环境变量见 scripts/build_kernels.sh
#   与 docs/archive/02-开发环境.md：
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
| `02-w4-quant/` | W4 kernel 三方对比（v1/Marlin/cuBLAS）+ 权重常驻字节 | M5/M10 | bench_kernel, bench_w4_residency |
| `03-kv4-quality/` | PPL 主线 + KV4 代价（HF 模拟法） | M0-M4/M8 | bench_ppl, bench_ppl_kv |
| `04-runtime-m8/` | 8B 四方矩阵 + paged 验收 + runtime PPL | M4/M7/M8 | bench_engine_matrix, m7_acceptance, bench_ppl_runtime |
| `05-spec-ngram/` | n-gram 投机（新 runtime） | M9 | bench_spec_ngram |
| `06-spec-draft/` | draft 投机 + 概率比接受的温度代价 + 动态 γ 扫参 | M9/M10 | bench_spec_draft, bench_spec_temperature, bench_spec_adaptive_tune |
| `07-prefix-cache/` | 前缀缓存 TTFT | M9 | bench_prefix_cache |
| `08-long-context/` | YaRN 分档 NIAH（32K/64K/128K 速度 + 召回 + 失败形态） | M11 | bench_niah_yarn |
| `archive-legacy/` | 旧引擎 spec bench（M3/M6 口径） | M3/M6 | bench_spec, bench_spec_modes |

## 关键结果速查（→ 底稿文件）

| 结论 | 数字 | 底稿 |
|---|---|---|
| 8B FP16 基线 | 42.4 tok/s | results/m4_matrix_fp16.json |
| 8B W4A16KV4 decode（新 runtime+Marlin） | 97.2 tok/s（1.75×） | results/m9_spec_8b.txt（off 行） |
| n-gram 复读 γ=4 | 310.4 tok/s（3.19×） | results/m9_spec_8b.txt |
| draft γ=8 复读（融合后） | 225.1 tok/s（2.30×） | results/m9_gamma_sweep_draft.txt |
| draft γ=2 自然（融合后） | 114.4 tok/s（1.17×）；M10 复测 116.5，γ≥3 全在 1.0× 噪声带 | results/m9_draft_fused.txt + m10_draft_sweep.txt |
| 动态 γ 只裁不涨（8B natural，γ 上限 4） | 钉上限 0.98×[0.90,1.05] vs 棘轮 1.06×[1.01,1.14]（n=5 重测；M10 原表 0.91→1.07 的因果读法已撤）；回升分支证伪 | results/m10_draft_sweep.txt + m10_adaptive_tune.txt |
| ↳ 动态 γ 的两个先验扫参 | 落点是唯一决定性轴：级联/阶梯/一步到 1 一律 0.79–0.83×；WINDOW 缩到 1/2 在 copy 上 −27%/−24%（误触发），W≥3 十次抽取从不触发；**裁窗口不省时间**（步价只有 21.5/16.3ms 两档）⇒ 代码不改，空间在早停 | results/m10_adaptive_tune.txt |
| T>0 概率比接受的吞吐税（1.7B，ngram copy） | 3.06× → 2.68×（+12.4%）；lookahead natural 0.62→0.00 acc | results/m10_spec_temperature.txt |
| 前缀缓存 8B TTFT（M10 重跑） | 555.9→62.3ms（8.92×，fp16 权重口径） | results/m9_prefix_cache.txt |
| Marlin 路径双份 int4 常驻（8B 死重量） | 3.335 GB；**已释放**（后端 `uses_v1_pack`）→ 8B 常驻 6.674→3.339 GB，0.52×→**0.26×** fp16 | results/m10_w4_residency.txt |
| ↳ 腾出的字节折成 KV 池（8B @util=0.6） | 1.52→3.16 GB（**实测 +108%**；此前按"字节直接进池"估的 +220% 不成立，池只兑现预算的 ~51%） | 同上 |
| KV4 PPL 代价 | +0.369 | results/m8_kv4_ppl.json |
| W4 kernel 三方 | M≥8 Marlin 碾压 | results/m5_kernel_bench.json |
| 长上下文速度曲线（8B W4A16KV4 + YaRN） | decode 13.3 / 7.1 / 3.7 tok/s @32K/64K/128K；冷 prefill 6.7 / 17.8 / 53.2 s；peak 13.38 GiB | results/m11_yarn_niah.txt |
| 分档 NIAH 召回 | 6/8、8/8、7/8、**3/8**（128K）。**这是严格匹配下界，不是检索成功率**：128K 的漏 4/5 是"数字前缀全对、尾部丢"⇒ 塌点在转写；YaRN 对召回的贡献**未证**（无 128K 基线） | 同上 + `benchmarks/08-long-context/README.md` |
