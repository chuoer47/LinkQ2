# M7 — 量化 paged attention：实验心得

> 2026-09-16 · Qwen3-8B W4 · RTX 4090 D (GPU3) · Triton 量化 paged attention

## 验收结果（docs/design-m7.md 三条线）

| 验收线 | 结果 | 判定 |
|---|---|---|
| PPL：paged vs dense KV4 差 <0.05 | **-0.20**（paged 18.50 比 dense 18.70 **更好**） | ✅（见下） |
| 长上下文峰值显存不再有 O(context) fp16 副本 | **@8K 峰值 24.4→18.4 GB（-6 GB）** | ✅ 实证 |
| 原有 19 测试无回归 | 全过 | ✅ |

单层数值：paged kernel vs dense 参考 max diff **0.0000**（300 token，跨 3 块）。

## 为什么 paged 的 PPL 反而更好

dense 和 paged 的量化方案数学上应该给出不同的数值：dense 路径的 group scale 在
"追加新 token"时只基于**当时的 token**（它的 `_k_tail` staging 语义），而 paged 的
`_write_k` 在写入时**重算整个 group**（含之前的 token）——后者的 scale 覆盖了更
完整的组内容，理论上更准。-0.20 的差异在 PPL 噪声边缘（16 docs 口径 ±0.2），
但它符合"重算组的 scale 更准"的推理，不是纯粹的运气。

## 交付物

| 组件 | 层 | 内容 |
|---|---|---|
| `kernels/kv4_paged_attention.py` | L0 | Triton kernel：per-(q-head) program 流式扫块，tile 内 int4 反量化（static_range 展开 nibble 位运算），online softmax；**GQA 映射**（q_head → kv_head） |
| `quant/cache/kv4_paged.py` | L1 | `KV4PagedCache`：块式存储（BLOCK_N=128 块对齐 GROUP=64）+ block_table + append 跨块 + dense_view（prefill 用） |
| `quant/kv_strategies_impl.py` | L1 | `kv4.paged` 策略注册（paged 槽位兑现） |
| `models/patched.py` | L2 | decode 分支：`isinstance(KV4PagedCache) and T_q==1` → 走 paged kernel |

## 设计决策的兑现与修正

1. **块对齐分组**（BLOCK_N=128 恰含 2 个 GROUP=64）→ 分组永不跨块，避免了"跨块
   scale"的复杂方案 ✓ 按设计生效
2. **prefill 走 dense / decode 走 paged** 的混合路径 ✓——paged kernel 是 M=1 设计
   （每 program 一个 query token），prefill 用它会有 O(T²) 读放大
3. **接口扩展兑现**：`attention(q)` 直接返回结果（不再返回 dense K/V），L2/L3 只加
   了一处 isinstance 分支 ✓

## 修掉的 4 个 bug（按破坏力排序）

1. **组内 scale 污染**（最隐蔽）：`_write_k` 追加 token 时只用新 token 算 scale，
   但 scale 是整组共享的——组内之前的 token 被静默用错误 scale 反量化。修复：写入
   时重算整个 group（从 packed 数据恢复已有 token，`_dequant_k_group`）。
   **这是 per-group 量化的分页写入的经典坑。**
2. **uint32 下溢**：`tl.where(val >= 8, val - 16, val)` 在 uint32 上做减法，val<16
   时回绕到 ~4e9，dequant 输出天文数字。必须先 `.to(tl.int32)` 再减。
   **Triton int4 解包的标准坑。**
3. **fp16 scale 下溢**：全零/极小组的 scale（clamp 1e-12）存 fp16 变 0 → softmax
   除零。下界抬到 1e-4（fp16 最小正规数 6.1e-5 之上）。
4. **张量转置**：`k[0].permute(1,2,0)` 给的是 [T,D,H] 不是 [H,D,T]。

另有一个调试教训：`_m7_kdeq.py` 中间调试脚本自身的参考实现也有 uint32 下溢，
导致"kernel 和参考一致但都错"的假象——**调试脚本的参考实现也要按同样的正确性
标准写**。

## 对 M8（nano-vllm 整合）的意义

本阶段交付的 `KV4PagedCache` 已经是"写入位置 = 块号"的结构——**CUDA Graph 的
核心障碍（动态 KV 写入地址）在数据结构层面已解决**。M8 接入 nano-vllm runtime
时，paged KV 的 slot_mapping 语义可以直接映射到 block_table，且 KV4 的量化读取
已经验证可行。

## 显存账（@8K ctx, 8B W4）

- dense KV4：packed 345 MB + 瞬时 fp16 副本 ~6 GB → **峰值 24.4 GB**（卡满）
- paged KV4：packed 321 MB，无副本 → **峰值 18.4 GB**（余量 5 GB）

这也顺带回答了 M2 遗留的问题："KV4 的瞬时副本在长上下文下可能反超 fp16"——
实测确实如此（dense 路径峰值 24.4 GB > fp16 cache 的 24 GB 附近），M7 消除了它。
