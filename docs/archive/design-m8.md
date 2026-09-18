> **[已归档]** 本文档为历史设计稿，结论以 [ARCHITECTURE.md](../ARCHITECTURE.md) 为准。

# design-m8: nano-vllm runtime 整合

> 目标：把 nano-vllm 的 L3 runtime（paged KV + CUDA Graph + 连续批调度）接进 qslab，
> 保留我们的 L1 量化栈（W4 权重 + KV4）与 L2 模型定义路线。
> 源：`.cache/nano-vllm-main`（MIT，GeeeekExplorer/nano-vllm）。

## 整合策略：vendor + 定向改写（不是 import 依赖）

nano-vllm 的 933 行 runtime 是**围绕它自己的模型定义**写的。我们**保留它的
调度/分页/图捕获逻辑**（这些是调好的轮子），但做两处结构性替换：

1. **模型定义**：用我们的（transformers Qwen3 + W4Linear backend 体系），
   替换它的 `models/qwen3.py` + `layers/`。这保留"transformers 当算子库"的
   既有决策，且 W4 量化栈直接生效。
2. **Attention 层**：它的 `Attention`（flash-attn + slot_mapping 写入）保留，
   但 decode 路径**接我们的 kv4_paged kernel**（M7 交付物）。flash-attn 处理
   prefill（varlen），我们的 Triton kernel 处理 decode 的 int4 读取。

## 分层落位（沿用 R1 的五层）

```
qslab/runtime/          ← 新 L3 子层（vendored nano-vllm，MIT 归属注释保留）
├── __init__.py
├── config.py           ← 改：合并我们的 EngineConfig 语义（w4/kv4 路径）
├── llm_engine.py       ← 原样（轻改：去 TP 多进程，单卡即可）
├── model_runner.py     ← 改：加载我们的模型构建路径 + W4 swap
├── scheduler.py        ← 原样（连续批 + 抢占逻辑不动）
├── block_manager.py    ← 原样（前缀缓存哈希不动）
├── sequence.py         ← 原样
└── context.py          ← 原样（attention 的全局上下文，graph 必需）
```

## 关键适配点（按顺序）

### 1. KV 块大小对齐（265 → 128）

nano-vllm 硬编码 `kvcache_block_size=256` 且 `assert %256==0`。我们的量化
group 是 64、块是 128。**改 config 默认为 128 + 去掉 256 断言**（改为 %128）。

### 2. KV cache 分配（fp16 → int4 packed）

`model_runner.allocate_kv_cache()` 分配 `[2, layers, blocks, block_size, H, D]`
的 fp16 张量再挂到各层。改为分配我们的 int4 五维组（`KV4PagedCache` 的布局），
并给每层 attention 挂 k/v/scale 四个 buffer。

### 3. Attention 层的双路径

nano-vllm 的 `Attention.forward`：prefill 用 `flash_attn_varlen_func`（读
dense cache + block_table），decode 用 `flash_attn_with_kvcache`。

改写为：
- **prefill**：flash-attn varlen 不变（此时 K/V 是刚算出的 fp16，还没量化）
- **写入**：量化进 paged 块（我们的 `_write_*` 逻辑，slot_mapping 驱动）
- **decode**：我们的 `kv4_paged_attention`（int4 直读）

### 4. 权重加载与 W4 swap

nano-vllm 的 `load_model` 走 `weight_loader` 协议。我们的 W4 swap 是事后
替换（`swap_w4_linears`）。顺序：先按 nano-vllm 的模型定义加载 fp16 权重
（含 packed_modules_mapping 的 QKV 合并），再跑 `swap_w4_linears` 换成
W4 backend。**QKV 合并**（q/k/v 拼成一个矩阵）与我们的 per-layer 量化兼容：
swap 时把 qkv_proj 当三个独立 Linear 分别量化（`packed_modules_mapping` 已
给出映射）。

### 5. CUDA Graph

`capture_cudagraph()` 不动——它依赖 slot_mapping/context_lens/block_tables
都是**静态地址的张量**（nano-vllm 已做到）。我们的 W4Linear/attention 只要
在 capture 时是同一 kernel 序列即可。**KV 写入地址 = slot_mapping 内容**，
已是数据驱动 ✓。

## 验收

1. 单请求生成与现有 dense/paged 引擎**输出一致**（无损性复用）
2. **连续批吞吐**：256 序列的 bench 对比单请求循环（nano-vllm README 声称与
   vLLM 同级——我们不求同级，求的是"我们的量化栈在分页 runtime 上工作"）
3. CUDA Graph 生效：`enforce_eager=False` 时的 tok/s vs eager
4. W4 权重生效：显存占用与 M4 的 2.49GB 口径一致

## 风险

- **flash-attn 版本**：nano-vllm 要 flash_attn_varlen_func（2.7.4 有）✓
- **QKV 合并量化的 group 对齐**：q/k/v 拼接后按行量化不影响（group 沿输入维）
- **graph capture 与 W4Linear 的兼容**：v1 kernel 是纯 CUDA 无 Python 状态 ✓；
  marlin 的 workspace 是固定 buffer ✓；**marlin 的 dispatch 依赖 M**——capture
  时 M 固定为 graph_bs ✓ 无动态分派风险
- **transformers 版本**：nano-vllm 用 transformers 的 Config/模型类名，我们
  4.57.6 ✓

---

## 追加（M8-S2 实测后）：K/V 的量化方案定版

原设计沿用 `attention_store.py` 的 **per-token per-head** 方案。实测**推翻**：

- Qwen3 的 K 在 `k_norm + RoPE` 后有**固定离群通道**（layer0 amax 313 / 均值 2.8）
- per-token 量化下步长被离群拉大，99% 通道被压成 0，相对误差 **23%**，decode 即崩
- 同精度下 per-channel 只有 3.6%

**定版方案**（对齐 KIVI / QServe，见 `notes/M8-整合.md` 的调研表）：

| | 量化 | scale | CUDA Graph |
|---|---|---|---|
| **K** | int4，per-channel | **静态**（离线校准冻结 `[H,D]`） | ✅ 写入纯 slot 驱动 |
| **V** | int4，per-token group=64 | 动态（`[slots,H,D/64]`） | ✅ 离群是 token 局部的 |

理由：per-channel 的动态 scale 必须在旧 token 到达新 token 时重算，写入地址即依赖
数据，graph 捕获不成立。用**离线校准**（SmoothAttention 压平离群 + 冻结 scale）
换取"写入纯 slot 驱动"。V 无此约束，保持动态分组。

前置步骤：`scripts/build_smooth_kv.py` 产出 `λ[H,D]` 与 `kscale[H,D]`；
λ 在 attention 内做 `q*=λ, k/=λ`（norm 权重按头共享，折不进去）。

**验收**：1.7B e2e 与 HF oracle 逐 token 一致；KV 池 6.92 GiB（fp16 需 13.84）。
**未覆盖**：CUDA Graph 尚未实测（设计上兼容）、连续批吞吐、W4 权重进 runtime。
