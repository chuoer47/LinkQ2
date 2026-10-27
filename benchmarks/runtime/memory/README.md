# 整模峰值显存

比较相同 W4 模型下 FP16 KV 与 KV4 的全流程峰值显存。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m pip install -e ".[benchmark]"
.conda/bin/python -m benchmarks.runtime.memory.run --gpu 1
```

默认输入 512/2048 token、输出 32 token，固定 4096 token KV 容量、Graph 和 64 token 分块 prefill。覆盖加载、预热、KV 分配、Graph 捕获、prefill 和 decode；完成后关闭模型。

NVML 每约 2ms 采样目标进程；PyTorch 同时记录张量分配峰值和内存池预留峰值。这三个指标不能相加。KV 池大小单独列出，不能代替整模峰值。为排除生成分叉，两个配置使用相同的固定后续 token；此入口不评估生成质量。需要项目环境中的 `nvidia-ml-py`。

结果写入 `results/runtime/memory/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
