# LinkQ2 文档

| 专题 | 阅读内容 |
|---|---|
| [量化与离线包](quantization/README.md) | 四种 W4A16 算法、KV4 方案、校准和权重包使用 |
| [Kernel](kernels/README.md) | W4 的布局与计算路径、KV4 Attention 的分页、复用、并行与精度取舍 |
| [Runtime](runtime/README.md) | 模型加载、请求调度、KV 页、分块 prefill，以及独立的 CUDA Graph 设计说明 |
| [投机推理](speculative/README.md) | 候选生成、接受概率、草稿状态和性能取舍 |

[项目架构](PROJECT_MAINLINE.md)提供代码导航。[Benchmark](../benchmarks/README.md)给出测量入口和数据，[tests](../tests/README.md)说明正确性测试。
