# design-m3: draft-model 投机推理

> M3 设计半页。定稿后动工。

## 目标

Qwen3-0.6B draft + Qwen3-1.7B target（M4 换 8B target），经典拒绝采样（Leviathan & Chen 2023，无损）。验收：固定 seed 下输出分布与 target-only 一致率 >99%；acceptance rate ≥2 时端到端 ≥1.5×。

## 算法（greedy 变体先行，采样版跟上）

### Greedy 投机（M3 主路径，与 M0 验收口径一致）

```
循环：
  1. draft 自回归生成 γ 个 token（用 draft 自己的 KV cache，从上次截断点继续）
  2. target 一次前向吃这 γ 个 token（target cache 追加 γ 格）
     → 得到 target 对每个位置的 argmax：t_1..t_γ 及下一个分布
  3. 逐位置比较：draft 的 d_i vs target 的 t_i
     - 全部相同：全接受，下一个 target argmax 作为第 γ+1 个 token（bonus token）
     - 第 i 个不同：接受 d_1..d_{i-1}（+ bonus t_i），回滚 target cache 到接受长度，
       draft cache 同步回滚，继续循环
```

- Greedy 下无损性显然：接受的每个 token 都被 target argmax 背书
- **bonus token**：每轮至少白赚 1 个 target token（这就是加速的来源下限）
- 回滚：target cache 的 update 已按位置写入，回滚 = len 回调（数据不用清，后续覆盖）

### Stochastic 投机（S1 实现数学，S5 用固定 seed 验证）

拒绝采样：draft 给 p(x)，target 给 q(x)，接受概率 min(1, q(x)/p(x))；
拒绝时从归一化的 max(0, q-p) 重采样。单测：跑 10k 步分布对齐 target-only（KS 或 chi2）。

## 调度细节

- **draft 模型加载**：复用 QslabEngine（kv_mode=fp16，max_len 小），engine 生成返回 logits 而非仅 token
- **KV 对齐**：draft 和 target 的 cache 各自独立维护；回滚时两者 len 同步
- **γ 扫描**：1/2/3/4/5/6，量 acceptance length（每轮平均接受 token 数）与端到端 tok/s
- **显存**：0.6B fp16 ~1.2G + 1.7B ~3.4G + caches——24G 无压力（M4 时 8B ~16G + 0.6B，也够）

## 何时加速（理论）

decode 是带宽受限：draft 每步读 0.6B 权重（~1.2GB/s·step≈1.2ms），
target 每步读 3.4GB（~3.5ms）。γ=4、接受率 α=平均接受 2.5 tokens/轮：
- 无投机：2.5 token = 2.5×3.5ms = 8.75ms
- 投机：draft 2.5 步（3ms）+ target 1 轮（3.5ms）= 6.5ms → ~1.35×
- ar=3.5 时 ~1.9×；ar≥2 是 1.5× 线的必要条件（α 高时成立）

风险：1.7B↔0.6B 的同族 draft 接受率预期 ar 1.5~2.5（EAGLE 论文同规模数据）。
若 ar<1.5，加速不达标 → 记录分析（γ 调优/换 draft 策略），不上 EAGLE（训练超范围）。

## 文件

- qslab/sampler.py：采样数学（含拒绝采样）
- qslab/spec/draft.py：draft engine 包装
- qslab/spec/verify.py：greedy 验证 + 回滚
- qslab/engine.py：spec_generate() 主循环
- benchmarks/bench_spec.py：γ 扫描 + ar + tok/s
- tests/test_sampler.py：无损性单测

## 不做

tree attention（单链即可）、EAGLE/Medusa（需训练）、draft 量化（M4 可选）
