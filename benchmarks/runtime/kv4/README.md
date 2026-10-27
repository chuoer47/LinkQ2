# KV4 整模质量

在完整模型中观察 KV4 对 PPL 的影响。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.runtime.kv4.run --gpu 1
```

固定 Qwen3-8B FP16 权重、同一份 SmoothAttention 校准和 1024 token KV 容量。使用 8×512 token、共 4088 个预测目标。第一个 token 使用 prefill，后续以 32 行因果验证读取缓存；每层误差继续影响后续层。

只测质量和缓存字节；执行速度见引擎与 Kernel benchmark。这样不会用一次完整浮点 prefill 的结果冒充 KV4 质量。

结果写入 `results/runtime/kv4/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
