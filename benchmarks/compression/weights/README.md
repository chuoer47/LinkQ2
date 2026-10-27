# 离线压缩

比较 FP16 与 RTN、RTN-Clip、AWQ、AWQ-Affine。记录 PPL、权重包大小和在线权重存储量。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.compression.weights.run --gpu 1 --prepare
```

默认使用 Qwen3-8B 和固定 WikiText-2 的 8 段×512 token。四种算法分别运行 fake 与真实 W4；fake 恢复为浮点 Linear，real 由 GemLite 打包执行。AWQ-Affine 默认 group size 为 64，其余为 128。

`--prepare` 只生成尚不存在的权重包。校准输入为 `datasets/w4a16/checkpoint_calibration.pt`，与评测文本分离。PPL 越低越好；权重包使用 INT8 容器保存逻辑 W4，磁盘大小与在线 int4 存储量需要分别看。

结果写入 `results/compression/weights/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
