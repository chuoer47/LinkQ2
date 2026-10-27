# KV4 分布与量化误差

用真实 Qwen3-8B 激活比较 K/V 的分布，以及静态逐通道量化、动态分组量化和平滑变换带来的误差。

```bash
python -m benchmarks.compression.kv4.run --gpu 3
```

默认使用 8 段、每段 512 token 做校准，再用 2 段、每段 256 token 评估。只量化 K 或 V，分别观察张量和 Attention 输出的相对 L2、截断比例和零值比例。整模累计误差由 [KV4 PPL](../../runtime/kv4/README.md) 衡量。

结果写入 `results/compression/kv4/`。表格见 [RESULTS.md](RESULTS.md)，量化方式说明见 [KV4 文档](../../../docs/quantization/02-kv4.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
