# nano-vLLM 接入来源

上游：<https://github.com/GeeeekExplorer/nano-vllm>。
本轮 clone 固定提交：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`；许可证为 [MIT](LICENSE.nano-vllm)。

- `qslab/kernels/fp16_kv.py` 从 `nanovllm/layers/attention.py` 提取 `store_kvcache_kernel` 和 `store_kvcache`，两段函数保持原样。
- FP16 attention 沿用上游的“按 slot_mapping 存储，再读 paged cache”方式。Graph 填充行的 slot 为 -1，不写入真实缓存。prefill 使用官方 FlashAttention varlen，decode 使用只读 `flash_attn_with_kvcache`；缓存形状保持项目的 `[page, token, head, dim]`。
- `ModelRunner.capture_cudagraph/run_model` 沿用上游的固定输入缓冲区、batch bucket、共享 graph pool、decode replay。适配项目配置：bucket 不超过 max_num_seqs；捕获时 slot=-1；replay 清空未使用的元数据；lm_head 与采样仍在 graph 外。
- 现有 KV4 和 GemLite 计算实现不变，只复用同一套 graph 管理。
