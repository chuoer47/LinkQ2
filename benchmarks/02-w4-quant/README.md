# 02 — W4 kernel 三方对比：v1 自研 / Marlin / cuBLAS（M5）

## 证据链

| M | N4096 K12288 | v1 | marlin | cuBLAS(fp16 deq) | 底稿 |
|---|---|---|---|---|---|
| 1 | | **90.1µs** | 179.2µs | 130.0µs | results/m5_kernel_bench.json |
| 8 | | 681µs | **185.3µs** | 166.9µs | 同上 |
| 512 | | 43212µs | **566µs** | 380µs | 同上 |
| 4096 | | 345675µs | **3095µs** | 2918µs | 同上 |

Marlin 锚点（vllm env）：M=1 27.6µs @ 911 GB/s（带宽极限 ~1008）。

**⚠ 重要更正（M9）**：本微基准当时得出 crossover=8 的 dispatch 结论，
被 M9 e2e 推翻——e2e 上 Marlin 在 M=1 也赢（8B decode 91.1 vs 54.1 tok/s），
因为 v1 GEMV 逐行重读权重、图内 252 层的固定开销模型不成立。
现行 `CROSSOVER_M=0`（`qslab/quant/w4_backends.py`）。**本目录数据用于理解
kernel 行为，不用于 dispatch 决策。**

## 复现

```bash
python benchmarks/02-w4-quant/bench_kernel.py
# L2 冲刷内建（M<=8 每迭代 memset 96MB）；需要 marlin 扩展已编译（见总 README 环境段）
```

---

## M10 追加：权重常驻字节（不是速度，是显存）

`bench_w4_residency.py` 遍历真实引擎里的每个 `W4Linear`，把 buffer 逐张加总：
v1 pack（`qfp`+`scale`）、Marlin 重排（`_B`+`_s`+workspace）、AWQ `act_scale`，
并报告"如果 v1 pack 被释放，KV 池能长多少"。

底稿 results/m10_w4_residency.txt：

| 模型 | backend | 常驻 | vs fp16 | 死重量（v1 pack） | KV 池 |
|---|---|---|---|---|---|
| 1.7B | w4.auto / w4.marlin | 1.355 GB | 0.52× | 0.677 GB | 4.41 → 5.08 GB（+15%） |
| 1.7B | w4.v1 | 0.678 GB | 0.26× | 0 | — |
| 8B | w4.auto / w4.marlin | 6.674 GB | 0.52× | **3.335 GB** | 1.52 → 4.85 GB（**+220%**） |

**结论修正（重要）**：`w4.auto` 与 `w4.marlin` 两行逐列相同。多出来的一份 int4
**不是 hybrid dispatch 的代价**，而是 `W4MarlinBackend` 本身在 `W4Linear` 的
`qfp/scale` 之外另存 Marlin 排布（其 `memory_bytes()` 直接相加）。也就是说
"因为 `CROSSOVER_M=0` 所以 auto 退化成 marlin、于是白占一份"这个说法只对了一半——
换成 `w4.marlin` 也省不下来，唯一省的下法是 `w4.v1`（M9 已证伪其速度）。

价值集中在 8B：util=0.6 下 KV 池只有 1.52 GB ≈ 161 block ≈ 2.1 万 token，
腾出 3.335 GB 可换到 ≈ 6.6 万 token。释放 repack 后源 pack 的改动**尚未实现**，
本底稿只证明这笔空间存在。

```bash
# 需要 W4 目录 + results/smooth_kv4_*.pt；必须 conda activate（见总 README 环境段）
CUDA_VISIBLE_DEVICES=1 python benchmarks/02-w4-quant/bench_w4_residency.py
CUDA_VISIBLE_DEVICES=1 MODEL=models/Qwen3-8B W4=models/Qwen3-8B-qslab-w4-awq \
  BACKENDS=w4.auto UTIL=0.6 python benchmarks/02-w4-quant/bench_w4_residency.py
```
