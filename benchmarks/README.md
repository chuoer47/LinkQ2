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
| `09-batch-throughput/` | **batch>1 吞吐扫描**（并发轴首次定价）+ KV 池计费实测 | M12 | bench_batch |
| `10-vllm-compare/` | 官方 vLLM 0.11 同 prompt 墙钟对照（fp16↔fp16） | M12 | bench_vllm |
| `11-service-surface/` | OpenAI 兼容服务面冒烟（验收脚本，不是吞吐实验） | M12 | smoke, smoke_concurrency |
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
| 批量加速（1.7B fp16 权重+KV4，关投机） | bs1→32 **24.43×**（3572.8 tok/s 解码窗口 / 3516.0 墙钟）；每请求 146→112 tok/s | results/m12_batch_1.7b.txt |
| ↳ 8B W4A16KV4 同轴 | bs1→16 **12.99×**（1271.9 tok/s）；每请求 98→80 tok/s | results/m12_batch_8b.txt |
| ↳ **投机在高并发翻负**（本轮最有价值的负性结论） | 1.7B ngram/off：bs1 1.90× → bs8 1.10× → bs16 **0.90×** → bs32 **0.77×**；8B 到 bs16 仍有 1.50× | results/m12_batch_1.7b_spec.txt + m12_batch_8b_spec.txt |
| ↳ 口径边界：扫描 bs=1 的投机读数 ≠ 出厂读数 | 1.7B 278.1 vs 自检 441.3（−37%）、8B 313.5 vs 310.6（+0.9%）；除首 token 外 prompt 相同 ⇒ 机制**未证**，两表不可互比 | benchmarks/09-batch-throughput/README.md |
| 官方 vLLM 对照（1.7B fp16↔fp16，同 token ids、墙钟口径） | 本引擎 = vLLM 的 **0.67–0.73×**，比例不随并发变；批量曲线重合（24.43× vs 23.91×） | results/m12_vllm_1.7b.txt |
| ↳ 量化侧的容量收益（同 util=0.5） | KV 池 **135,936 vs 75,808 tokens（1.79×）**；按真实字节 28.9 vs 112 KiB/token = **3.87×**，只兑现到 1.79× 是因为下面那条超订 | 同上 + results/m12_batch_1.7b.txt |
| KV 池超订（TODO 11 由读代码转成实测） | 计费 vs 实配 = **1.94×**（两个模型都是）；8B 池 431 块 = 55,168 tok = **13 条** 4096-token 序列 = 并发上限 | 四份 results/m12_batch_*.txt 的 `[kvschema]` 行 |
| 服务面 8 并发（HTTP 面，1.7B fp16+KV4） | 384 tok / 0.444 s = **865 tok/s**，8 条同批完成；与进程内 bs=8 扫描（1015）差 14.8%，**不可整笔记在 HTTP 上** | results/m12_service_smoke.txt |
