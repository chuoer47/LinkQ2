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

底稿 results/m10_w4_residency.txt（改动前）：

| 模型 | backend | 常驻 | vs fp16 | 死重量（v1 pack） | KV 池 |
|---|---|---|---|---|---|
| 1.7B | w4.auto / w4.marlin | 1.355 GB | 0.52× | 0.677 GB | 4.41 → 5.08 GB（估 +15%） |
| 1.7B | w4.v1 | 0.678 GB | 0.26× | 0 | — |
| 8B | w4.auto / w4.marlin | 6.674 GB | 0.52× | **3.335 GB** | 1.52 → 4.85 GB（估 **+220%**） |

**当时的结论（一半已被推翻）**：`w4.auto` 与 `w4.marlin` 两行逐列相同，多出来的
一份 int4 **不是 hybrid dispatch 的代价**，而是 `W4MarlinBackend` 本身在
`W4Linear` 的 `qfp/scale` 之外另存 Marlin 排布。由此得出"唯一省的下法是
`w4.v1`（M9 已证伪其速度）"——这句错了，见下。

## M10 续：释放已落地，同脚本复测

改动：后端声明 `uses_v1_pack`，Marlin repack 完就把 `qfp/scale` 丢掉，
`W4Linear` 只在 `uses_v1_pack` 为真时注册这两个 buffer，`memory_bytes()` 只算
实际常驻的那一份。`w4.auto` 在 `CROSSOVER_M=0` 下同样报假，所以主线一并生效。

| 模型 | backend | 常驻 | vs fp16 | 死重量 | KV 池（实测） |
|---|---|---|---|---|---|
| 1.7B | w4.auto / w4.marlin | **0.678 GB** | **0.26×** | 0 | 4.41 → 4.73 GB（+7%） |
| 8B | w4.auto | **3.339 GB** | **0.26×** | 0 | 1.52 → **3.16 GB（+108%）** |

1. **Marlin 主线现在和 `w4.v1` 逐列同宽**（0.678 / 3.339 GB，0.26× fp16），
   省显存不再要用 decode 速度换。
2. **"腾出 3.335 GB 就能长 3.335 GB KV 池"的算术不成立**：实测腾出的字节进的是
   分配预算（8B budget 2.94→6.14 GiB，正好 +3.2 GiB），而池子两次都只兑现预算的
   ~51%，所以 1.52→3.16 GB（+108%）而非 4.85 GB。token 容量 ≈2.1 万→≈4.4 万。
   预算为何只兑现一半**未证**（peak/current 与 block 取整都在吃）。
3. **省下的显存没有用 decode 速度付款**：同一次运行里 `w4.auto` 与 `w4.marlin`
   的 graph decode 逐行相等（1.7B：178.8/179.4、178.8/179.0、171.4/170.3 tok/s，
   差 ≤0.6%）。绝对值**不可跨会话比**（与 M9 记录的 155.2 口径不同）。PPL 未复测：
   `_B/_s` 逐字节不变，数值面由全量测试（84/84）守。

```bash
# 需要 W4 目录 + results/smooth_kv4_*.pt；必须 conda activate（见总 README 环境段）
CUDA_VISIBLE_DEVICES=1 python benchmarks/02-w4-quant/bench_w4_residency.py
CUDA_VISIBLE_DEVICES=1 MODEL=models/Qwen3-8B W4=models/Qwen3-8B-qslab-w4-awq \
  BACKENDS=w4.auto UTIL=0.6 python benchmarks/02-w4-quant/bench_w4_residency.py

# 读数 3 的速度对照（本目录只测字节，脚本在 05-spec-ngram）
CUDA_VISIBLE_DEVICES=1 MODEL=models/Qwen3-1.7B W4=models/Qwen3-1.7B-qslab-w4-awq2 \
  UTIL=0.5 TOKENS=256 GAMMA=4 python benchmarks/05-spec-ngram/bench_spec_ngram.py
```
