# 分块预填充

观察长请求到达时，混合调度对短请求延迟和整体吞吐的影响。

在仓库根目录使用项目 `.conda`：

```bash
.conda/bin/python -m benchmarks.runtime.chunked_prefill.run --gpu 1
```

首请求输入 32 token、输出 16 token；执行两步后再加入输入 512/32 token、各输出 8 token 的两个请求。分别测 FP16＋FP16 KV 和 W4＋KV4 的 prefill-first、混合调度。Graph 均开启，查询预算为 64 token。

每组预热一次、测三次。除整批耗时和吞吐外，JSON 记录三个请求的测量中位数，包括 TTFT、TPOT 和最大 token 间隔。这是三请求场景，不代表高并发下的容量或服务质量。

结果写入 `results/runtime/chunked_prefill/results.json` 和 `report.md`。JSON 保存测量条件、主要指标和少量重复测量数据。

测量结果见 [RESULTS.md](RESULTS.md)。

运行后会同时更新本目录的结果文档，表格与 JSON 来自同一组测量。
