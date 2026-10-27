# Benchmark

测量分为离线压缩、Kernel、Runtime 和投机推理。各目录包含运行说明、默认小规模输入和结果说明。

| 分组 | 实验 | 内容 |
|---|---|---|
| compression | [离线压缩](compression/weights/README.md) | 比较 FP16 与 RTN、RTN-Clip、AWQ、AWQ-Affine。记录 PPL、权重包大小和在线权重存储量。 |
| compression | [KV4 分布与误差](compression/kv4/README.md) | 比较真实激活分布、平滑与分组量化的误差。 |
| kernels | [W4A16 单算子](kernels/w4a16/README.md) | 比较相同矩阵形状下的 FP16 Linear 和融合 W4A16。 |
| kernels | [KV4 Attention](kernels/kv4/README.md) | 比较 FP16 KV 与 KV4 的单层计算时间，并拆出写入、Attention 和前缀物化。 |
| runtime | [推理引擎](runtime/engine/README.md) | 比较 FP16/W4 权重与 FP16/KV4 缓存的四种组合。 |
| runtime | [CUDA Graph](runtime/cuda_graph/README.md) | 比较 eager 与 Graph 的请求延迟。 |
| runtime | [分块预填充](runtime/chunked_prefill/README.md) | 观察长请求到达时，混合调度对短请求延迟和整体吞吐的影响。 |
| runtime | [KV4 整模质量](runtime/kv4/README.md) | 在完整模型中观察 KV4 对 PPL 的影响。 |
| runtime | [整模峰值显存](runtime/memory/README.md) | 比较相同 W4 模型下 FP16 KV 与 KV4 的全流程峰值显存。 |
| runtime | [Nsight 性能分析](runtime/profile/README.md) | 查看 Attention 或完整模型执行中的 GPU 内核时间和 CPU 提交间隙。 |
| speculation | [N-gram 投机推理](speculation/ngram/README.md) | 比较不使用草稿模型时，候选窗口 1/2/4 的实际收益。 |
| speculation | [自回归草稿](speculation/draft/README.md) | 比较普通解码与 Qwen3-1.7B 草稿投机。 |
| speculation | [DFlash 投机推理](speculation/dflash/README.md) | 比较普通解码与 DFlash 的端到端生成性能。 |

表格中的 GPU 测量使用 NVIDIA GeForce RTX 4090 D。不同实验的执行配置见各自说明。

结果在仓库根目录的 `results/` 中，与实验目录同名。`results.json` 保存可直接阅读的指标，`report.md` 展示表格；需要查看 GPU 时间线时使用 Nsight 的报告文件。

[正确性测试](../tests/README.md)负责数值和状态断言。benchmark 中的有限值、输出工作量和执行路径检查用于排除无效测量，不代替完整测试。
