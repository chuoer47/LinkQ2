# design-m2: KV cache 4bit 量化

> M2 设计半页。定稿后动工。

## 目标

KV cache 从 FP16 → 4bit：KV 显存省 ≥3.5×（验收 ≥3.5 是给 zero/开销留量后的实际值），PPL 劣化 <0.5（vs FP16 KV），32K NIAH 召回率降幅 <5%。

## 方案（KIVI 式不对称 + 前置层 FP16）

### 量化粒度（不对称，K 和 V 结构不同）

- **K per-channel**：K 的离群按 head 通道分布（RoPE 后 key 的某些 dim 恒定偏大），
  per-token 量化会把这些离群摊进每个 token 的 scale → 精度崩。按 channel 分组量化
  等价于把 K^T 按 token 维分组（KIVI 的核心 trick：K 存转置）
- **V per-token**：V 无系统性离群结构，per-token（行内 group）即可
- 两者都是 group-wise + 对称（沿用 M1 的 scale 语义，zero=0）

### 精度分层（kv4_plan）

- **前 2 层 KV 保 FP16**（KIVI/QServe 共识：首层 KV 离群最重）
- 其余层 KV4，group=64（KV 数值范围比权重平缓，组可以更小）
- plan 离线生成：校准集前向，测每层 K/V 的 absmax 分布，离群度 = amax/median(amax)，
  离群度 > 阈值的层保 FP16（预期就是前 1~2 层）

### 读时反量化（M2 先行策略）

- cache 存储 int4 打包 + scale；attention 读取时整块反量化回 FP16 再走 matmul
- 不做 attention kernel 内反量化（那是 M2 可选优化 / M2b）——理由：decode 单步
  attention 是小 GEMV，反量化开销可接受，先把"量化-存储-精度"链路做对
- 显存账（Qwen3-1.7B, GQA 8 KV head, head_dim 128, 28 层, 32K ctx）：
  - FP16: 2(K,V) × 2B × 8 × 128 × 32768 × 28 ≈ 3.7GB
  - KV4: 权重部分 0.5B + scale 每 64 元素 2B（≈1/32 开销）→ ~0.55GB → **~6.7× 省**
  - 前 2 层 FP16 影响: +2/28 ≈ 忽略

## 数据流

```
engine.generate
  → prefill: K/V 计算后逐层 cache.update(k,v)（quantize=True 的实现里先量化再存）
  → decode: cache.update 追加；attention 读 k_full/v_full 时反量化整段
```

接口不变（BaseKVCache.update 返回 fp16 K/V）——attention 代码零改动，精度切换
只换 cache 实现。这是 M0 接口设计的回报。

## 实现文件

- quantizer/kv4_plan.py：离线 plan 生成（输出 json 进打包 config）
- qslab/cache/kv_cache.py：KV8Cache（int8 验证管线）、KV4Cache（主目标）
- scripts/m2_s4_ppl.py：PPL 对比（FP16 KV vs KV8 vs KV4）
- scripts/m2_s5_niah.py：NIAH 自实现（随机 key-value，深度扫描）

## 里程碑顺序

S1 plan → S2 cache 实现 → S3 接 attention（按 plan 分层）→ S4 PPL/显存 → S5 NIAH → S6 笔记

## 风险与对策

- K per-channel 的转置存储：cache 布局 K 存 [B, H, D, T]（转置），attention 用
  时再转回来（或 matmul 直接吃转置形式）
- 反量化整段的显存峰值：一次只反量化当前需要的窗口（attention 本来就要全段 K/V，
  峰值=一段 KV 的 fp16，可接受）
- PPL 超标：先查 group=64 → 32；再查前 N 层 → 4；最后 per-channel K 的 group 沿
  channel 维分组方式
