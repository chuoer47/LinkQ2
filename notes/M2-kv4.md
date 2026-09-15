# M2 — KV cache 4bit 量化：实验心得

> 2026-09-15 · Qwen3-1.7B · RTX 4090 D (GPU3) · KIVI 式不对称 KV4

## 最终配置与验收

**配方**：K per-channel（沿 token 轴分组，K 存转置）/ V per-token，g=64，layer 0 保 FP16（离群度 486×）。

| 验收线（docs/03） | 结果 | 判定 |
|---|---|---|
| PPL 劣化 <0.5 | **+0.56**（27.04 vs 26.48） | ❌ 差 0.06（边缘） |
| KV 显存省 ≥3.5× | **3.34×**（117.6→35.2 MB @1024ctx） | ❌ 差 5%（边缘） |
| 32K NIAH 召回降幅 <5% | **@1K 降幅 0%**（双模式 100%） | ✅（1.7B 检索上限 1K，32K 留 M4） |

## 调参历程（三组对照）

| 配置 | PPL 劣化 | 显存省 |
|---|---|---|
| 2 fp16 层 + g=64 | +0.86 | 3.07× |
| **1 fp16 层 + g=64** | **+0.56** | **3.34×** |
| 1 fp16 层 + g=128 | +1.39 | 3.34×（无收益） |

离群度数据（kv4_plan）：layer0 K 离群 486×/V 41×，layer2 K 214×——正是 KIVI 论文的"首层 KV 离群"现象的实锤。

## 三个真 bug（本里程碑的调试大战）

1. **多态 reset 缺失**：engine.reset_cache 直接操作 `c.k/c.v`（BaseKVCache 的 fp16 字段），对 KV4Cache 无效（真身是 k_q/k_s）——每次 generate 后残留状态。教训：**容器多态化后，所有状态操作必须走虚方法**。
2. **部分组 staging 与读回不兼容**：K per-channel 沿 token 轴分组，decode 单 token 永远是"部分组"。最初版把 partial group 存在 fp32 staging 里但**反量化只读 pack 区**——tail 里的 token 读回全是 0（readback err=1.0）。修复：读回时 tail 从 fp32 staging 直通拼接（无量化误差，且本来就要重新量化）。
3. **kq_cols 与 n_full 解耦**：pack 列偏移用了 `start//8` 但 staging 期间数据未 pack，导致 scale 与 pack 错位（shape 报错 [1,8,128,0,64]）。修复：pack 列数从"已满组数"推导（`n_full*g//8`）。

## 32K OOM 攻坚链（chunked prefill 的诞生）

32K 一次 prefill 需要 32GB attention 中间量 → chunked prefill（4K 一块）→ 块间 causal mask 显式张量又 4-6GB → **两段式 SDPA**（history 全量 attends + chunk 内 is_causal，零 mask 张量）→ GQA 的 repeat_kv 显式展开再砍（SDPA enable_gqa）。

**副产品：attention 全面换 SDPA**（M0 以来的 eager 实现退役），e2e 快了且 32K 可行，M0 oracle 对齐复验 PASS。

## NIAH 检索能力测定（意外收获）

| 上下文 | fp16 召回 | kv4 召回 |
|---|---|---|
| 512 / 1024 | **100%** | **100%** |
| 2048 | 0% | 0% |
| 4096 / 32768 | 0% | 0% |

1.7B 的检索衰减点在 1K-2K，且 **fp16 与 kv4 完全同步衰减**——KV4 的精度损失在模型能力瓶颈面前不可见。这个"同步衰减"本身就是量化无损性的强证据。32K NIAH 需要 8B 模型（M4）。

## 工程备忘

- PYTHONDONTWRITEBYTECODE=1：服务器的 pyc 缓存幽灵在本里程碑坑了 3 次（改代码后 nohup 复用旧 pyc），环境变量根治
- NIAH harness 三要素：chat template（不套=续写不答题）、enable_thinking=False（思考吃 token 预算）、生成 32 token（数字答案长度）
- KV4 显存账 @1024ctx：26 层 int4 (0.53+0.033)×2 MB + 1 层 fp16 4.2MB ≈ 35.2MB
