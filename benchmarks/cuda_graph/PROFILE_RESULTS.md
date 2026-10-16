# Decode CUDA Graph 加速归因

2026-10-16，RTX 4090 D，Qwen3-8B，batch=1，512 token 输入、16 token 输出。测量代码提交 `b17f08c`；runtime 和 kernel 与 `8425fdb` 相同。没有修改算子或推理实现，没有重新量化、校准或跑 PPL。

实验确认：W4 的 M=1 矩阵 kernel 已有速度优势，但 eager 中的主机下发开销将其掩盖；Graph 重放后优势兑现。当前 KV4 在该长度下增加延迟，组合反超的速度收益来自 W4。

## 完整引擎延迟

每组独立进程，组内共享同一份模型、KV 池和捕获图，交替切换 eager/graph；各预热一次、测三次取中位数。图内模型 forward 和 KV 读写；图外调度、元数据上传、lm_head 和采样。关闭 prefix cache，不使用 chunked prefill 或投机推理。

| 权重 / KV | eager TPOT ms | Graph TPOT ms | 自身加速比 |
|---|---:|---:|---:|
| FP16 / KV16 | 33.476 | 19.254 | 1.74× |
| W4 / KV16 | 77.426 | 8.916 | 8.68× |
| FP16 / KV4 | 36.188 | 21.923 | 1.65× |
| W4 / KV4 | 79.956 | 11.701 | 6.83× |

开启 Graph 后，W4/KV4 相对 FP16/KV16 快 **1.65×**；W4/KV16 快 **2.16×**。

## 为什么会反超

固定 forward 使用相同 token=42、position=512、已填充 KV、输入地址、元数据和输出缓冲；不包含 lm_head、采样和调度。延迟与 Profiler、cProfile 分开采集。

1. **矩阵 kernel 本身没有因 Graph 明显变快。** 时间线中，252 次 FP16 矩阵 kernel 累计为 eager **14.846 ms**、Graph **14.850 ms**；相同 KV16 下，252 次 W4 GEMV 累计为 **4.382 ms / 4.338 ms**。W4 GEMV 本来就更快，Graph 前后变化很小。W4 的输出清零计入“其他 kernel”，不能把 GEMV 数字当成全部 Linear 开销。

2. **Graph 消除了大量逐个下发与 GPU 操作间隙。** 固定 W4/KV4 forward 在 eager 下执行 **2853 个 kernel**，有 **2853 次 Runtime/Driver kernel 下发**；Graph 下执行 **2854 个 kernel**，主机只下发 **1 次 Graph launch**。两种模式的 kernel 名称和数量差异仅为 Graph 多一次 `memcpy32_post`，没有发生算子融合。

   | 配置 | 执行 | kernel 累计 ms | 首尾 GPU 活动跨度 ms | GPU 活动间隙 ms |
   |---|---|---:|---:|---:|
   | FP16/KV16 | eager | 18.105 | 51.100 | 32.990 |
   | FP16/KV16 | Graph | 17.836 | 17.946 | 0.110 |
   | W4/KV4 | eager | 10.637 | 99.260 | 88.622 |
   | W4/KV4 | Graph | 10.261 | 10.385 | 0.124 |

   这些是 **Profiler 采集期间**的时间，采集会放大 eager 开销，不能替代上面的无 Profiler TPOT，也不能把间隙全部归为 CPU 计算。间隙按 GPU kernel、memcpy、memset 区间并集计算。

3. **额外开销不只是 kernel 数量多。** W4/KV4 的 kernel 数只比 FP16/KV16 多约 11%。补充 cProfile 对同样 KV16、三次固定 forward 的采集：FP16 eager 为 **59,206 次函数调用**，W4 eager 为 **305,554 次**。额外调用集中在 `GemLiteLinear.forward → ops.linear → GEMV forward → Triton Autotuner.run / JIT.run / 参数特化`，并包含 PyTorch 默认设备的分发处理。预热后采样中没有 benchmark 或编译调用；`Autotuner.run` 是缓存命中后的下发路径，不代表每次重新调优。Graph 重放绕过这条 Python 调用链。

## KV4 的真实贡献

同样开启 Graph，KV16 换为 KV4 后，FP16 权重延迟从 **19.254 → 21.923 ms**，W4 权重从 **8.916 → 11.701 ms**。时间线中 attention 累计约 **0.35 ms → 2.98 ms**，说明当前 KV4 实现在该长度下仍较慢。

因此，本次数据支持 W4 带来的 GPU 工作耗时收益，以及 Graph 带来的下发开销收益。减少权重搬运是合理解释，但本实验没有采集 DRAM 带宽计数器，不把“带宽瓶颈”视为已被直接测量的事实。结论仅适用于本次 M=1 decode，不能推广到 prefill。

## 复现与原始记录

```bash
.conda/bin/python -m benchmarks.cuda_graph.profile --gpu 1 \
  --tuning results/cuda_graph/2026-10-16/autotune.json
```

不提供 `--tuning` 时会在预热中正常调优。默认输出带时间戳目录。

本次服务器原始目录：`/home/amax01/lingchen/LinkQ2/results/cuda_graph_profile/2026-10-16/`；本地镜像：`results/cuda_graph_profile/2026-10-16/`。包含四组完整请求和固定 forward JSON、各三次 eager/graph 的压缩 GPU trace、CUDA Runtime/Driver 调用统计、GemLite 调优参数及中文报告。FP16/KV16、W4/KV16 的 CPU 调用栈补充采集另存 `*.pstats`、`*-eager.txt`、`*-graph.txt` 和 `*.cpu_run.json`；补充采集脚本在仓库外，持久入口已集成同样的 cProfile 步骤。

四组均实际重放 Graph，固定 forward 输出有限，结束后 KV 页归还。已有 W4 数值不确定性和严格 graph/eager 一致性未通过的结论仍成立；本实验完成的是性能归因，不是精度验收。
