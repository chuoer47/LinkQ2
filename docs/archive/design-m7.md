> **[已归档]** 本文档为历史设计稿，结论以 [ARCHITECTURE.md](../ARCHITECTURE.md) 为准。

# design-m7: 量化 paged attention

> 目标：让 KV4 在**分页存储**上直接工作，消除当前 KV4Cache 的两个代价。
> 路线：用户已定 **B + C** —— 自己写 Triton kernel，结构抄 paged-attention 骨架，
> 核心抄 KIVI 的 tile 反量化思路。

## 要解决的代价（M2 遗留）

当前 `KV4Cache.update()` 每次调用都做三件事：

1. 量化新 token
2. 存进 packed buffer
3. **把整段有效长度反量化回 fp16 稠密张量并返回**

代价一：反量化是 O(context)，decode 第 1000 步时每层都要解 1000 个 token，
只为算一个新 token 的 attention。
代价二：那个 O(context) 的瞬时 fp16 副本与 packed 存储**同时存在**，
长上下文下峰值显存可能反超纯 fp16（M2 报的 3.5× 只算了 packed 部分）。

## 方案

**新 add 一条存储 + 一个 kernel，老的 dense 路径保留**（作为 correctness 参照）。

```
KV4PagedCache (L1)          块式存储 + block_table 间接寻址
  └─ kv4_paged_attention.py (L0)   Triton kernel，tile 内反量化
```

### 块布局（关键设计决策：块对齐）

K 的分组沿 token 轴（per-channel），分页的块也沿 token 切——**两者必须对齐**，
否则一个 group 会跨块。取 `BLOCK_SIZE = 128`、`KV_GROUP = 64` →
每块恰好含 2 个 group，**分组永不跨块**（不需要"跨块 scale"的复杂方案）。

打包布局（每块独立，`num_blocks` 由 cache 容量决定）：

```
K（per-channel，沿 token 分组）：
  k_q  [num_blocks, H, D, BLOCK_SIZE/8]        uint32   每 8 个 token 一 word
  k_s  [num_blocks, H, D, BLOCK_SIZE/GROUP]    fp16

V（per-token，沿 D 分组）：
  v_q  [num_blocks, H, BLOCK_SIZE, D/8]        uint32
  v_s  [num_blocks, H, BLOCK_SIZE, D/GROUP]    fp16
```

对比老布局：老的是把整条序列当一个连续 buffer（`[B,H,D,L/8]`），
新的是同样内容**按 128-token 切成物理块**，逻辑位置经 `block_table` 映射。

### 为什么 K 存成「[D, BLOCK_N] 通道优先」

kernel 里算 score：`scores[n] = Σ_d q[d]·K[n,d]`。K 按 `[D, N]` 存，
这个式子就是 `Σ_d q[d]·K[d,n]`——**沿 d 轴归约，不需要转置**。
转置在 Triton 里是显式开销，能省则省。

### kernel 设计（decode-only）

引擎是 decode-only，所以只写 M=1 的流式版本（不需要 flash attention 的
M 分块那套）。每个 program 负责一个 (query token, kv_head)：

```
q ← load [D]
for blk in block_table:                  # 流式扫过上下文
    K ← dequant(load k_q[blk], k_s[blk])   # [D, BLOCK_N]
    s ← Σ_d q[d]·K[d,n]                    # [BLOCK_N]
    s ← where(pos < context_len, s, -inf)  # 尾部掩码
    m_new ← max(m, max(s)); p ← exp(s - m_new)
    V ← dequant(load v_q[blk], v_s[blk])   # [BLOCK_N, D]
    acc ← acc·exp(m-m_new) + Σ_n p[n]·V[n,:]
    l ← l·exp(m-m_new) + Σ_n p[n]; m ← m_new
out ← acc / l
```

online softmax 是标准写法；**新的部分只有 `dequant`**——把 tile 里的
uint32 解包成 int4、符号扩展、乘 scale，大约 15 行。

## 接口（不动 L2/L3）

`KV4PagedStrategy` 实现已有的 `KVCacheStrategy` 协议。但 paged cache 的
`update()` 契约要扩展：attention 需要拿到 **block_table + context_len**，
而不是稠密张量。做法：cache 暴露 `attention(q, head_idx)` 直接返回结果，
由 `PatchedQwen3Attention` 在检测到 paged 时走这条分支。

> 这是 M0 接口设计的第一次真正扩展：dense 路径 `update() → (k, v)` 不变，
> paged 路径走新方法。engine 只需一处 `if isinstance(cache, PagedCache)`。

## 验收

1. **数值**：paged 路径 vs dense KV4 路径，attention 输出逐元素误差 < 1e-2
   （同为 int4 量化，差异只该来自求和顺序）
2. **端到端**：8B + KV4-paged，PPL 与 dense KV4 相差 < 0.05
3. **显存**：长上下文下峰值不再有 O(context) fp16 副本
4. **无回归**：原有 19 个测试全过；dense KV4 路径不受影响

## 不做

- prefill 的 paged 版本（引擎是 decode-only）
- 与 nano-vllm 的对接（那是 M8，本阶段只交付能用的 paged attention）
- 多序列 batching（batch=1 定位不变）
