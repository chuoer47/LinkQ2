# nano-vLLM 接入来源

上游：<https://github.com/GeeeekExplorer/nano-vllm>。
本轮 clone 固定提交：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`；许可证为 [MIT](LICENSE.nano-vllm)。

- `qslab/kernels/fp16_kv.py` 从 `nanovllm/layers/attention.py` 提取 `store_kvcache_kernel` 和 `store_kvcache`，两段函数保持原样。
- FP16 attention 沿用上游的“按 slot_mapping 存储，再读 paged cache”方式。Graph 填充行的 slot 为 -1，不写入真实缓存。prefill 使用官方 FlashAttention varlen，decode 使用只读 `flash_attn_with_kvcache`；缓存形状保持项目的 `[page, token, head, dim]`。
- `execute/cuda_graph.py` 的 `CUDAGraphs` 沿用上游固定输入缓冲区、batch bucket、共享 graph pool 和 replay。decode 与固定 M verify 使用相同的捕获/重放逻辑，各自保存 GraphFamily。bucket 不超过 max_num_seqs；捕获时 slot=-1；replay 清空未使用的元数据；目标 lm_head 与采样仍在 graph 外。
- 现有 KV4 和 GemLite 计算实现不变，只复用同一套 graph 管理。

## 混合 Chunked Prefill

来源：[nano-vLLM PR #276](https://github.com/GeeeekExplorer/nano-vllm/pull/276)，固定提交 `dbf934fcb44cbc6b763517106153767d97719bf3`。该生态实现尚未合并进上游 main，不能标为正式发行功能。

适配其 `scheduler.py` 的 decode 优先、剩余 token 预算接纳 prefill、KV 空闲块 watermark，及 `model_runner.py/sequence.py` 的 `should_sample`：中间分块只写 KV，不执行 lm_head/采样；最终块和 decode 正常提交。decode 在混合批中表示为长度 1 的 chunk，纯 decode 仍走 graph。

项目保留二元调度返回值，布尔量统一表示“该批是否含 prefill”，不另加仅用于日志的统计位。`enable_chunked_prefill` 控制混合策略，关闭时为 prefill 优先对照；`prefill_reserve_blocks` 默认 0。混合策略支持 n-gram 和草稿模型投机推理。

FP16 使用上游 paged varlen 接口；KV4 接入现有反量化前缀+FlashAttention varlen 路径，计算算子不变。混合批中的 decode 连同刚写入的当前 token 一起从量化池读取，prefill 新分块保持浮点 K/V，避免改变原 decode 的量化语义。`run_model` 返回 hidden states，lm_head 只处理实际采样行；项目内 Draft 调用点同步改为显式取 logits。

## N-gram 投机接入

`qslab/speculation` 只处理 token 历史和概率张量，不导入 runtime。引擎持有 proposer，调度器按候选与当前 token 的实际查询数量分配预算，执行器负责组装输入和选择 logits，接受结果由调度器提交并更新 KV 有效长度。

纯 verify 捕获固定 `M=spec_num_drafts+1` 查询的 Graph，保留请求数 bucket；有效候选不足时填充行不写 KV，也不计算其输出 logits。没有候选时走普通 M=1 decode。混合 prefill/verify 及固定填充会超过上下文范围的情况，按实际查询数走 varlen；其中 verify 的所有当前查询 KV 均从量化池读取并采用因果 mask。

decode/verify 的 Graph 重放分别计数。接受率按实际提出并提交的候选统计，`mean_len` 为每请求 verify 轮次的平均提交 token 数。固定 Graph 填充成本与实际查询预算分别记录，不将理论减少的轮数当作实测加速。

## 草稿模型接入

`runtime/speculation/draft.py` 复用 ModelRunner 和 BlockManager，独立维护草稿权重、KV 池及请求状态；使用相同 token-ID 空间，不共享目标模型 KV。草稿通过 `draft_kv_cache_dtype` 与 `draft_smooth_kv` 显式选择 KV16 或 KV4，默认 FP16 KV，不推导校准路径、不强制编译。权重按草稿 checkpoint 的算法配置加载；共享 embedding/lm_head 的量化模型在 meta 初始化与权重加载后恢复绑定。候选数量在执行草稿前按目标剩余输出和查询预算限制，概率按请求和有效候选排列，接受算法仍位于 `qslab/speculation`。

`runtime/speculation/dflash.py` 按官方 checkpoint auto_map 加载草稿，复用官方 block diffusion forward；不复制修改官方模型计算实现。`runtime/speculation/dflash_graph.py` 使用 Transformers StaticCache，按请求复用固定地址的草稿 Graph 槽位；补齐特征写入 masked scratch 区，有效缓存长度单独维护。长提示词首次提议走官方 eager forward，关闭草稿 Graph 时使用 DynamicCache。目标按需输出指定层的完整残差特征，decode/verify Graph 复制到固定特征缓冲区；引擎仅传递实际确认且已计算的前缀。DFlash 复用目标 embedding/lm_head，草稿 BF16，目标 FP16 激活交界显式转换。当前仅 greedy，块长由 checkpoint 决定；DFlash 禁用缺少中间特征的目标前缀缓存复用，保留分页 KV、chunked prefill 和目标验证 Graph。固定形状与动态 BF16 路径存在已记录的数值差异，见 benchmark 结果。

每轮只保留草稿中与目标已提交历史一致的已计算 KV，拒绝分支回退，全部接受时正确补算最后一个草稿 token 和 bonus。仅缺两个 token 时可复用 M=1 Graph 补算，更长历史使用有预算的 varlen 分块，候选逐步生成使用草稿自身 M=1 Graph；目标沿用固定 M=k+1 Graph 和混合 varlen。被抢占、完成、取消及异常的草稿页面随请求释放。两模型预留显存并分别建池，生命周期复用 runner.exit。
