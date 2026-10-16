# nano-vLLM 接入来源

上游：<https://github.com/GeeeekExplorer/nano-vllm>。
本轮 clone 固定提交：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`；许可证为 [MIT](LICENSE.nano-vllm)。

- `qslab/kernels/fp16_kv.py` 从 `nanovllm/layers/attention.py` 提取 `store_kvcache_kernel` 和 `store_kvcache`，两段函数保持原样。
- FP16 attention 沿用上游的“按 slot_mapping 存储，再读 paged cache”方式。Graph 填充行的 slot 为 -1，不写入真实缓存。prefill 使用官方 FlashAttention varlen，decode 使用只读 `flash_attn_with_kvcache`；缓存形状保持项目的 `[page, token, head, dim]`。
- `ModelRunner.capture_cudagraph/run_model` 沿用上游的固定输入缓冲区、batch bucket、共享 graph pool、decode replay。适配项目配置：bucket 不超过 max_num_seqs；捕获时 slot=-1；replay 清空未使用的元数据；lm_head 与采样仍在 graph 外。
- 现有 KV4 和 GemLite 计算实现不变，只复用同一套 graph 管理。

## 混合 Chunked Prefill

来源：[nano-vLLM PR #276](https://github.com/GeeeekExplorer/nano-vllm/pull/276)，固定提交 `dbf934fcb44cbc6b763517106153767d97719bf3`。该生态实现尚未合并进上游 main，不能标为正式发行功能。

适配其 `scheduler.py` 的 decode 优先、剩余 token 预算接纳 prefill、KV 空闲块 watermark，及 `model_runner.py/sequence.py` 的 `should_sample`：中间分块只写 KV，不执行 lm_head/采样；最终块和 decode 正常提交。decode 在混合批中表示为长度 1 的 chunk，纯 decode 仍走 graph。

项目保留二元调度返回值，布尔量统一表示“该批是否含 prefill”，不另加仅用于日志的统计位。`enable_chunked_prefill` 控制混合策略，关闭时为 prefill 优先对照；`prefill_reserve_blocks` 默认 0。当前混合策略不与投机推理组合。

FP16 使用上游 paged varlen 接口；KV4 接入现有反量化前缀+FlashAttention varlen 路径，计算算子不变。混合批中的 decode 连同刚写入的当前 token 一起从量化池读取，prefill 新分块保持浮点 K/V，避免改变原 decode 的量化语义。`run_model` 返回 hidden states，lm_head 只处理实际采样行；项目内 Draft 调用点同步改为显式取 logits。
