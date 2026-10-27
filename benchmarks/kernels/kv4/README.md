# KV4 Attention

比较 FP16 KV 与 KV4 的单层计算时间，并拆出写入、Attention 和前缀物化。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.kernels.kv4.run --gpu 1
```

默认 Hq=32、Hkv=8、head_dim=128，页大小 256、V 分组 64；batch=1/4，上下文 128/512/2048，decode 查询 1 行、verify 查询 16 行，分块 prefill 追加 64 行。

使用固定随机输入，不需要模型。每项预热后进行三次 GPU Graph 测量，以微秒记录。各分项独立计时，不能直接相加；缓存驻留 L2 的单层结果不等同于整模 TPOT。JSON 同时记录各自缓存值下的数值误差和 KV4 相对 FP16 的输出差异。

结果写入 `results/kernels/kv4/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
