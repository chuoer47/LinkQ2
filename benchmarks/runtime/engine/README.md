# 推理引擎

比较 FP16/W4 权重与 FP16/KV4 缓存的四种组合。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.runtime.engine.run --gpu 1
```

使用 Qwen3-8B、已有 RTN 权重包与 KV4 校准。默认单请求，输入 128/512 token、输出 32 token，预热后测三次，开启 decode Graph 和 64 token 分块 prefill。关闭前缀复用，重复请求均完整计算输入。

TTFT 是请求入队到首 token 提交的时间；TPOT 是首 token 之后每个输出 token 的平均时间。总耗时包含调度、输入组装、模型、KV、采样和提交，不含加载与分词。三次时间样本和中位数均保留。

结果写入 `results/runtime/engine/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。

输入来自一段 512 token 的固定文本；`--contexts` 可选择其中的输入长度。
