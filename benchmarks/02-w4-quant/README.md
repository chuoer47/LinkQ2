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
