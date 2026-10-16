# FP16 KV / KV4 的 CUDA Graph 对照

当前测量与严格一致性状态见 [结果记录](RESULTS.md)。现有 W4+KV4 存在重复运行的数值波动，严格 graph/eager 关卡未通过；入口会保存报告并返回非零，不能当作无损等价验收通过。

```bash
.conda/bin/python -m benchmarks.cuda_graph.run --gpu 1
```

需已有原始 Qwen3-8B、RTN 完整 checkpoint 和 KV4 校准文件，默认路径与 `benchmarks/engine` 相同。不会重新量化、校准或安装依赖。

如果已有第一版引擎的 HF 参考，可避免重复生成：

```bash
.conda/bin/python -m benchmarks.cuda_graph.run --gpu 1 \
  --references results/engine/2026-10-16-v1-final
```

固定测量 FP16 权重+FP16 KV、GemLite W4A16+KV4 两组；每组分别 eager 和 graph，独立进程顺序运行。小规模默认：128/512 token 输入、16 token 输出，预热一次，测三次；固定 1024 token KV 容量、256 token 页大小、batch=1，不使用投机推理。

沿用 nano-vLLM 的 decode graph 边界：图内是模型 forward 和 KV 读写；prefill、最后 lm_head 和采样在图外。记录实际 graph replay 次数，避免将 fallback 当成 graph 测量。小模型回归另覆盖 3 条请求补齐到 4 行的安全性。

指标：TTFT、TPOT、整次延迟、输出吞吐、相对自身 eager 的加速比，以及两种配置在 graph 下的加速比；记录常驻、启动/运行峰值 allocated、权重与 KV 字节数和关闭后的工作区基线。库工作区按普通流/捕获流预初始化，不能把库保留误计为模型未释放。

精度沿用三段提示×16 个位置的 HF 参考，并保存相同参考历史下的 logits。graph/eager 配对要求相对 L2≤0.001、top1 一致率≥99%；这是执行一致性检查，不重新评判量化 PPL。

输出 `results/cuda_graph/<时间>/`：提交与参数、参考、原始 JSON/logits、日志、摘要及中文 `report.md`。上游固定版本与适配范围见 `qslab/runtime/UPSTREAM.md`。

Decode 加速归因可单独运行：

已完成的四组消融、GPU 时间线和 CPU 调用栈结论见 [归因结果](PROFILE_RESULTS.md)。

```bash
.conda/bin/python -m benchmarks.cuda_graph.profile --gpu 1
```

默认 512 token 输入、16 token 输出，三次完整请求测量；增加 W4/KV16、FP16/KV4 两组消融。每组在同一进程交替测 eager/graph，复用相同权重和 KV 池。固定 forward 进一步使用相同 token、已填充 KV、地址、元数据和输出复制，排除调度、lm_head、采样。无需重新量化、生成 HF 参考或跑 PPL。

无 Profiler 延迟、PyTorch Profiler 时间线和 cProfile CPU 调用栈分开采集。时间线记录 kernel 数量、累计耗时、GPU 活动间隙和 CUDA Runtime/Driver API 调用；间隙不能直接视为 CPU 计算耗时，CUDA Event 时间也包含 CPU 下发不及时产生的空隙。不会直接测硬件显存带宽。默认每种执行采三次固定 forward，压缩 trace 可用 Perfetto 查看；CPU 调用栈各采三次，保存 pstats 和按累计耗时排序的文本，不把采集耗时用于延迟结论。

可用 `--tuning results/cuda_graph/2026-10-16/autotune.json` 复用已有调优参数。输出 `results/cuda_graph_profile/<时间>/`，包含参数、各组 JSON、调优参数、压缩 trace、日志及中文报告。只采集性能和执行结构，不改变已有 W4 严格数值一致性未通过的结论。
