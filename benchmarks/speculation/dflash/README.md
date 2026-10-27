# DFlash 投机推理

比较普通解码与 DFlash 的端到端生成性能。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.speculation.dflash.run --gpu 1
```

目标为 Qwen3-8B RTN W4＋KV4；官方 b16 草稿保持 BF16 权重与 KV，一次最多提出 15 个候选。目标使用分页 KV、Graph 与分块 prefill；草稿使用 StaticCache Graph。非 thinking、greedy，遇 EOS 停止。

默认数学和代码两题、三次测量、最多 256 token。`--limit 6` 使用两道数学、两道代码和两道聊天题。`--compare-draft-graph` 增加 eager 草稿，单独观察草稿 Graph 的影响。

`python -m benchmarks.speculation.dflash.alignment --gpu 1 --limit 6` 比较官方 BF16 和项目四种权重/KV 配置，各自运行普通解码与 DFlash。HF 的首阶段截止于首次 forward，Runtime 截止于首 token 提交；跨后端的 TPOT 是近似比较。

输出长度可能不同，因此不能只看总秒数。这里不计算题目正确率，也不把运行变快等同于逐 token 严格一致。

结果写入 `results/speculation/dflash/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

W4＋KV4 结果见 [RESULTS.md](RESULTS.md)，不同权重与缓存组合见 [配置对照](CONFIGURATIONS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
