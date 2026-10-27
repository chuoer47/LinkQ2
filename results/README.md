# 测量结果

结果目录与 `benchmarks/` 的实验目录一致。例如：

```text
results/compression/weights/results.json
results/kernels/w4a16/results.json
results/kernels/kv4/results.json
results/runtime/engine/results.json
results/runtime/memory/results.json
results/speculation/dflash/results.json
```

每个实验主要保留 `results.json` 与 `report.md`。JSON 中有模型、输入输出规模等必要条件，以及 PPL、延迟、吞吐或显存等指标；重复测量保留少量时间样本。

Nsight 的 `.nsys-rep` / `.ncu-rep` 用于打开时间线和硬件计数器，JSON 保存主要内核耗时。使用方式见[性能分析入口](../benchmarks/runtime/profile/README.md)。

`--out` 可以指定另一目录；不指定时更新该实验的默认结果。模型权重放在 `models/`，KV 校准放在 `datasets/`，不混入测量结果。

中文表格同时更新到对应实验目录的结果文档。`--out` 改变原始文件位置，结果文档随同更新。
