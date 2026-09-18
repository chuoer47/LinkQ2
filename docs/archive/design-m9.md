> **[已归档]** 本文档为历史设计稿，结论以 [ARCHITECTURE.md](../ARCHITECTURE.md) 为准。

# design-m9 — n-gram 投机解码接入 runtime

日期：2026-09-18。前置：M8（W4A16 + KV4 + nano-vllm 引擎，CUDA Graph，30 测试绿）。

## 0. 业界依据（不是自拟方案）

- **vLLM V1 n-gram spec decode**（PR #5345 起源，V1 `v1/spec_decode/ngram_proposer.py`）：
  proposer 全程 CPU（numpy 滑窗匹配），提案 token 的 KV **直接 append 进 paged cache**，
  验证 = 一次 "flash forward"（context + 全部提案一起过 target 模型），
  验证后未接受部分回收（vLLM V0 做 retract/restore page table）。
- **TGI 1.4 / FlashInfer（SPOQ）**：append-only——避免 vLLM V0 的 page-table 回滚机制，
  未接受的 KV 标记为垃圾、由后续写入自然覆盖。
- **SpecInfer**：tree attention 把 token tree 序列化成一维、position_ids=tree depth；
  链式提案是其平凡情形（每序列 M 个连续 query）。
- **vLLM V1 graph 策略**：提案数固定 γ，不足 pad 并在采样侧 mask。

我们的选择：**vLLM 的 append→verify 模式 + TGI 的 append-only 回收**（回收 =
释放尾部块，见 §3），proposer 走 vLLM V1 的 CPU numpy 滑窗。

## 1. 分层（自底向上）

| 层 | 文件 | 职责 |
|---|---|---|
| L0 kernel | `runtime/paged_decode.py` | 现有 decode kernel 通用化：加 `M: constexpr`，grid `(N_Q, bs*M)`，row→seq=row//M，ctx_len=L+row%M。**M=1 时逐指令等价于原 kernel** |
| L1 attention | `runtime/attention.py` | `ctx.verify_m>1` 分支：q/k/v 天然是 [bs*M,...]，store 路径不变（本来就是逐 token）；verify kernel 出 [bs*M, H, D] |
| L2 runner | `runtime/model_runner.py` | `prepare_verify`（positions/slot_mapping/context_lens）+ graph 按 (bs, M=γ+1) 捕获一族 |
| L3 proposer | `runtime/ngram.py`（新） | `NGramProposer.propose(token_ids, gamma)`：取末 n 个 token 为 key，在已提交序列里找**最后一次**出现，返回其后继（≤γ 个）；无匹配返回 []。numpy sliding_window_view |
| L4 engine | `llm_engine.py` / `scheduler.py` | step() 分叉：decode 步变 verify 步（propose→reserve→forward→accept→commit→trim） |

config：`spec_method="ngram"|None`、`spec_num_drafts=γ=4`、`spec_ngram_size=n=3`。

## 2. 验证前向（核心机制）

greedy 下投机解码**数学等价**于非投机 greedy（Leviathan et al. 的接受律），
这是正确性依据。每次 verify：

- M = γ+1 个 query = `[t_{L-1}(最后已提交), d_1 … d_γ]`，positions = `L-1 … L+γ-1`
- 行 m 的读取上界 = `L+m`（store-before-attention，含自身槽位）——
  **因果性就是槽位下界**，后继 draft 的 KV 虽已写入，但每行的 mask 天然排除它们，不需要显式 mask
- logits 行 m 给出位置 L+m 的 target 分布。接受：`d_{m+1} == argmax(row m)` 的最长前缀 a；
  **bonus = argmax(row a)**。提交 `d_1..d_a + bonus`（a+1 个 token）
- 一次 verify 只有一次前向；γ+1 行的 GEMM 在 decode 尺度下仍近似带宽受限（M 小）

**Padding 中立性**：proposer 没提案（k=0，或该序列 temperature>1e-3 不参与投机）的序列，
pad 满 γ 个 dummy。接受循环把 a 钳制在 k（真实提案数）以内——dummy 永不可能被接受，
该步退化为普通 decode（bonus = row 0 的采样输出，与采样温度语义完全一致）。
因此整批统一走 verify graph，无混合模式。温度 > 1e-3 的序列不参与投机（拒绝采样的
分布保持需要概率比接受律，留作后续），但输出仍正确。

## 3. KV 槽位：预留 + trim（append-only 精神）

- verify 调度时把 block_table 预留到覆盖 index L+γ-1（至多 1 个新块，γ≪bs=128）
- 全部 γ+1 个 query 的 KV 先写入（地址纯由 position→block_table 决定，graph 兼容）
- 验证后按接受数 a 提交 token，`block_table` **trim 回 ceil(新长度/bs)**——被拒槽位
  所在的尾部块直接还回 free list，池里的垃圾数据无人读（行读取上界 = 各自 ctx_len）
- 为什么不用"不 trim、等自然覆盖"：`prepare_decode` 的槽位公式是
  `block_table[-1]*bs + last_block_num_tokens-1`，要求 table 长度恰为 num_blocks
  （block_table[-1] 假设）。trim 维持这个全局不变量，改一个地方不如维持一个不变量。
- 接受的 draft KV 是本次前向算出的真值，永不重写；这与 TGI append-only 的动机一致
  （不做 vLLM V0 的 page-table 备份/恢复）

## 4. CUDA Graph

- 第二族 graph：key = (bs, M=γ+1)，bs 桶复用现有 ladder；graph_vars 尺寸按 bs*M
- slot_mapping/context_lens/block_tables 是 graph 静态 buffer，verify 每步填
  （值由 position 纯函数决定，无数据依赖——这是 K 用静态 scale 的同一设计约束的延续）
- M=1 族（原 decode graph）保持不动

## 5. 测试口径（吸取"逐 token 无损"教训）

不断言"spec 输出 == 非投机输出"的逐 token 相等——两条路径的 GEMM M 维不同，
cuBLAS tiling 漂移会让近并列 argmax 翻转（M8 已确认 HF 自己也漂 0.02~0.03）。
断言的是**数学结构**：

1. kernel 对拍（严格）：verify 各行 vs 逐步 decode 的 logits（同池同权重），
   argmax 相等 + max|Δ| 紧界；因果性：只改最后一个 draft 槽的内容，前 γ 行输出不变
2. proposer 单元（纯 CPU，精确）：重复模式/无匹配/γ 截断/取最后一次出现
3. padding 中立性：k=0 stub proposer 的引擎输出 == 关闭投机引擎的输出（同一引擎
   两次运行，确定性可复现，无 tiling 混淆）
4. replay-stub：注入引擎自己的 greedy 输出作为提案 → 接受率 ≈100%（接受机制的结构性验证）
5. 复读 prompt（构造重复 token 模式）：n-gram 真接受 > 0 且步数严格减少
6. graph vs eager 一致（同输入同权重重放）
7. 边界：prompt 长度 < n / < γ、跨块（L ≈ 127/128/129）、eos 出现在提案中途、
   max_tokens 在提案中途截断

## 6. 验收

- 正确性：上述测试全绿；既有 30 测试无回归（kernel M=1 等价 + 调度器非投机路径零改动语义）
- 性能（`benchmarks/bench_spec_ngram.py`，1.7B+8B，greedy）：
  复读文本（预期显著 >1×）/ 自然文本 / 随机 token（预期 ≤1×，如实报告开销）
  指标：吞吐 tok/s、平均接受长度 a+1、接受率
- PPL 不受影响：投机解码 greedy 下数学等价，不改分布（无需重测；bench 采样
  口径 greedy）

## 7. 范围外（记录不做的事）

- 树形提案（EAGLE/Medusa/tree attention）——需训练/更大 kernel 改动
- temperature 采样的概率比接受律（Leviathan full rejection sampling）
- 旧 `qslab/engine/spec/`（chained 三模式）不迁移——n-gram 直接落 runtime，
  draft-model 模式将来复用同一 verify 机制（proposer 接口就是扩展点）
