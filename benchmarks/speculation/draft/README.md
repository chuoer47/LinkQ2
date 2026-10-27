# 自回归草稿

比较普通解码与 Qwen3-1.7B 草稿投机。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.speculation.draft.run --gpu 1
```

目标为 Qwen3-8B RTN W4＋KV4，草稿为 Qwen3-1.7B RTN W4＋KV4；两侧各有分页缓存和 decode Graph。默认候选窗口为 4。需要已有草稿权重包与 `datasets/kv4/qwen3-1.7b.pt`。

问题、模板、停止条件与其他投机 benchmark 一致。默认两题、三次测量；`--limit 6` 使用全部题目。草稿更小不保证总耗时更低，需要同时看草稿开销、接受长度和两模型峰值显存。

结果写入 `results/speculation/draft/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
