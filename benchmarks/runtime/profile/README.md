# Nsight 性能分析

查看 Attention 或完整模型执行中的 GPU 内核时间和 CPU 提交间隙。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.runtime.profile.run --gpu 1
```

默认用 Nsight Systems 采集单层 FP16 KV、KV4 的 decode、verify 和分块 prefill，保存 `.nsys-rep`和前十项内核耗时摘要。热身、编译和 Graph 捕获在采集前完成。

`--target engine` 改为完整 W4 模型，`--context 2048` 改变上下文。`--tool compute` 使用 Nsight Compute 采集一个内核的硬件计数器，需要服务器开放 GPU 性能计数器权限。工具通过 PATH 查找，也可以用 `--nsys`、`--ncu` 指定程序。

Nsight 采集会增加执行开销；请求延迟看其他 benchmark，不能用采集时间替代。GPU 忙碌时间与墙钟间隙分别看，不能把间隙全部算成 CPU 运算时间。

Systems 结果按采集目标写入 `results/runtime/profile/layer/` 或 `engine/`，包括 `results.json`、`report.md` 和 `.nsys-rep`。Compute 报告放在对应的 `compute/` 子目录。

测量结果见 [RESULTS.md](RESULTS.md)。

采集后会更新 [单层结果](LAYER.md)或[整模结果](ENGINE.md)，与对应 JSON 一起生成。
