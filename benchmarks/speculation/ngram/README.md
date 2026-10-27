# N-gram 投机推理

比较不使用草稿模型时，候选窗口 1/2/4 的实际收益。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.speculation.ngram.run --gpu 1
```

与普通解码使用同一目标：Qwen3-8B RTN W4＋KV4、Graph、64 token 分块 prefill、单请求。默认两题分别为数学和代码；候选来自已生成序列中的 n-gram 匹配，不额外下载草稿。

`--limit 6` 使用完整六题；`--windows 2` 只测一个窗口。每题预热后测三次，上限 256 token，遇 EOS 停止。JSON 保存实际输出数、TTFT、TPOT、总耗时、接受率和每次验证提交数。低命中率可能使投机变慢。

结果写入 `results/speculation/ngram/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
