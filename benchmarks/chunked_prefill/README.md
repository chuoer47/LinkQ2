# 混合分块预填充验收

```bash
.conda/bin/python -m benchmarks.chunked_prefill.run --gpu 1
```

需要现有 Qwen3-8B、RTN 完整包、KV4 校准与 WikiText 小样本，路径与第一版引擎 benchmark 相同；不重新量化或校准。

来源为 nano-vLLM [PR #276](https://github.com/GeeeekExplorer/nano-vllm/pull/276)，固定提交 `dbf934fcb44cbc6b763517106153767d97719bf3`，尚未合并上游 main。本项目适配其 decode 优先、统一 token 预算、混合 varlen 和 should_sample 路径，未自行设计新调度策略。

默认只测三个请求：第一个输入 32 token、输出 16 token；执行两步后，同时加入输入 512/32 token、各输出 8 token 的请求。固定 token 预算 64、最多 4 条序列、8×256-token 的 KV 池。四个独立进程分别测 FP16+KV16、W4+KV4 的 prefill-first 和 mixed。各预热一轮，测三轮取中位数，关闭前缀复用；纯 decode 均使用 CUDA Graph。

指标包括整批耗时、输出吞吐、原短请求 TPOT 和最大 token 间隔、长请求 TTFT、实际混合步骤、decode 推迟步数、graph replay、页释放和显存。保存每步调度轨迹；mixed 必须实际混合、遵守预算，且这个无资源压力的工作负载不能推迟原短请求的 decode。

FP16 的分块、混合及 graph 与 HF 对齐；KV4 另验证混合 decoder 读取包括当前 token 的量化 KV，而 prefill 新分块仍使用浮点 K/V。中间分块不调用 lm_head 或采样。这些数值回归在 `tests/runtime/engine/test_fp16_engine.py` 和 `tests/runtime/model/test_kv4_mixed.py`。

开启方式：

```python
LLM(path, enable_chunked_prefill=True, max_num_batched_tokens=64,
    enforce_eager=False, max_num_seqs=4)
```

`prefill_reserve_blocks` 为新 prefill 准入后保留的空闲页数，默认 0。当前只接普通生成，不与投机推理组合。KV4 在混合批仍需展开缓存前缀；本轮不增加 attention kernel，不把三请求结果外推为高并发、长上下文或抢占压力结论。

结果写入 `results/chunked_prefill/<时间>/`，含提交、参数、四组日志和原始 JSON、摘要与中文 `report.md`。
