# 调度、分页 KV 与分块 prefill

本篇按一次请求的推进过程说明 Runtime 状态。关键是区分已提交 token、已经算出 KV 的 token、以及当前为候选预留的容量。

## 1. Sequence 与页映射

Sequence 保存完整 token 历史、prompt/输出长度、`num_cached_tokens`、`num_scheduled_tokens`、block_table 及投机候选。最后一个已提交 token 通常尚未完成下一步所需 KV 计算，因此提交长度和有效 KV 长度不总相等。

BlockManager 把逻辑页映射到物理页，维护引用计数、空闲队列和前缀哈希。若页大小为 $P$、逻辑位置为 $t$：

$$
\mathrm{slot}(t)=\mathrm{block\_table}[\lfloor t/P\rfloor]P+(t\bmod P).
$$

完整块的哈希还依赖前序块哈希，并核对 token 内容，防止只因当前块文本相同就在不同前缀下复用 KV。最后一块保留重新计算空间。共享页增加引用计数，归还后只有零引用页能重新分配。

当运行队列需要更多页时，调度器可以抢占其他运行请求：释放其页、转回 waiting，之后基于已提交 token 重算。抢占并非删除已生成输出，也不能继续信任被释放的草稿或特征状态。

## 2. 两种调度顺序

未启用 chunked prefill 时，先安排等待队列的 prefill；当前没有 prefill 才安排 decode。首个长 prompt 可以在查询预算内分段推进，不必一次算完整个 prompt。

启用 `enable_chunked_prefill=True` 后，先安排运行中的 decode / verify，再用剩余查询预算安排 prefill。设一轮预算为 $Q_{\max}$：

$$
\sum_{s\in\mathrm{batch}} Q_s\le Q_{\max},\qquad
Q_s=\begin{cases}1&\text{decode},\\k_s+1&\text{verify},\\c_s&\text{prefill chunk}.\end{cases}
$$

候选数量还受输出余额、上下文余额及页容量限制。预算按实际查询计数，不按“请求数乘名义窗口”计费；Graph 内的填充行则是另一层计算成本。

部分 prefill 完成前不采样输出；只有真正处理到 prompt 尾部才产生首 token。当前混合分块配置支持无投机、ngram、draft、dflash；lookahead 与该配置组合会被拒绝。

## 3. ModelRunner 如何组织一次执行

| 路径 | 输入与元数据 | 执行 |
|---|---|---|
| prefill / 混合批次 | 拼接真实查询，cu_seqlens、位置、slot、页表与前缀计划 | 实际长度 varlen |
| 普通 decode | 每请求最后一个 token、位置与有效上下文 | M=1 Graph 或 eager |
| 纯 verify | 最后一个已提交 token 加候选 | 固定形状 Graph 或实际长度 fallback |

FP16 KV 的分块读取使用官方分页接口；KV4 要物化已缓存前缀并与新块浮点 K/V 拼接。KV4 纯 decode / verify 才直接从 packed 池读取，详见[Kernel 专题](../kernels/02-kv4-attention.md)。

执行上下文通过 `set_context/reset_context` 传递给 Attention；它是每轮构建的元数据，不应被当成跨请求持久缓存。

## 4. 调度结果怎样进入 Graph 或实际长度路径

纯 decode / verify 可以使用固定查询行数的 Graph，含 prefill 的批次按实际长度执行。调度器按有效查询行数分配预算，Graph 的窗口和 batch 档位则可能引入额外填充计算，两者需要分开计数。

捕获范围、输入更新、填充隔离、上下文末端的路径选择以及显存取舍，单独见 [CUDA Graph 设计](04-cuda-graph.md)。

## 5. 投机提交为什么要单独处理

verify 前先为全部候选查询预留页；本次计算可能写出后来拒绝的 token 的 KV。接受结果返回后，调度器只提交接受前缀与一个纠正/bonus token：

$$
L_{\mathrm{KV,valid}}=L_{\mathrm{committed}}-1.
$$

最后交付的纠正/bonus 尚未作为输入完成 KV。被拒绝位置通过有效长度隐藏，多余尾页归还，后续计算覆盖旧内容；无需逐个把旧槽位清零。停止边界会进一步截短交付，不能把整个验证窗口都写成已提交历史。

DFlash 还要求目标特征连续且只涵盖已确认、已计算的前缀。它的请求特征不在普通前缀缓存中，因此引擎在 DFlash 模式关闭目标前缀复用。完整流程见[投机模型接入](../speculative/02-models-and-state.md)。
