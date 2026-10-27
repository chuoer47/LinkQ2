# CUDA Graph

比较 eager 与 Graph 的请求延迟。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.runtime.cuda_graph.run --gpu 1
```

分别测 FP16 权重＋FP16 KV、RTN W4＋KV4；其他输入、缓存容量、输出预算和分块设置相同。默认输入 128/512 token，输出 32 token，三次测量。

Graph 捕获模型 forward 和 KV 读写，prefill、lm_head 与采样仍在图外。比较同一配置的 TPOT；测量延迟本身不证明不同计算形状产生完全相同的生成文本。

结果写入 `results/runtime/cuda_graph/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
