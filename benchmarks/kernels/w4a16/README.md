# W4A16 单算子

比较相同矩阵形状下的 FP16 Linear 和融合 W4A16。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.kernels.w4a16.run --gpu 1
```

覆盖 Qwen3-8B 的三种投影形状与一个非对称形状，M=1/32/128/512。GPU 时间由 Triton 的 CUDA Graph 计时工具测量，排除编译和 Python 提交间隙；同步主机时间由 PyTorch Timer 测量，包含调用和同步。

输入为固定种子的随机张量，不需要模型。`--reference <GemLite 源码目录>` 可增加原包比较。单算子加速比不能直接换算为整模加速比。

结果写入 `results/kernels/w4a16/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
