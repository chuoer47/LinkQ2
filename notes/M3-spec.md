# M3 — draft-model 投机推理：实验心得

> 2026-09-16 · Qwen3-1.7B (target) + Qwen3-0.6B (draft) · RTX 4090 D (GPU3) · greedy 投机

## 最终成果

| 验收线（docs/03） | 结果 | 判定 |
|---|---|---|
| 无损性（与 target-only 一致率 >99%） | **100%**（3 prompts × 64 tok 全一致） | ✅ |
| acceptance rate ≥2 时端到端 ≥1.5× | AR=4.96@γ4 但 e2e **0.74×** | ❌ 负加速（分析见下） |

正确性成果扎实：拒绝采样数学单测（TV<0.012）+ greedy 投机 100% token 一致。

## γ 扫描（512 in / 128 dec，5 轮中位数）

| γ | AR | tok/s | speedup |
|---|---|---|---|
| — (baseline) | — | 59.03 | 1.00× |
| 1 | 2.00 | 29.76 | 0.50× |
| 2 | 3.00 | 32.96 | 0.56× |
| 4 | **4.96** | 39.31 | 0.67× |
| 6 | 6.79 | 43.83 | 0.74× |

## 为什么 AR 极高却负加速（本轮最重要的分析）

1. **AR≈γ+1 几乎全接受**：512 输入是重复性校准文本，draft 预测极准。真实文本 AR 预期 1.5~2.5。
2. **Python 调度开销主导**：每轮 = γ 次 draft Python 循环（各一次 model forward）+ 1 次 target 批量前向 + 2 次 commit forward + 2 次 cache 回滚。γ=4 时每 5 个 token 要 ~7 次 Python 级 forward 调用。1.7B 的 GPU 前向只需 ~17ms/token 基线，**CPU dispatch（每次 ~1ms）占比过高**——投机推理在"模型小 + Python 引擎"场景下天然吃亏。
3. **基线偏快**：重复文本对 cuBLAS/cache 友好，59 tok/s 高于 M0 的 42（真实文本基线会更低，加速比会更接近 1 但仍难转正）。
4. **理论转正条件**：target 越大 GPU 时间占比越高（8B 时 target 前向 ~5ms，draft 便宜，dispatch 被摊薄）+ batch 内多请求摊调度成本 + CUDA Graph 消 dispatch。**这正是 M4 的实验点**。

## 算法实现的三个真 bug（投机推理的经典坑，全踩了一遍）

1. **验证对齐错误**：`a_i = target 看完 g_1..g_i 后的预测` 验证的是 **g_{i+1}**，不是 g_i。g_1 的验证值必须来自**上一轮**（prev_target_pred 贯穿循环）。初版把 a[i] 对 g[i] 比对 → 全盘错位。
2. **bonus 选择错误**：accept=k 时 bonus 是"第一个被拒绝的预测"= prev_target_pred（k=0）或 a[k-1]（k>0），不是 a[k]。a[k] 是错误世界（g_{k+1}=被拒 token）的预测。
3. **cache 与 generated 失步**：bonus token 的 KV 未写入 cache 就进入下一轮（下一轮验证前向会把它写到错位）。修复：回滚后 bonus 走一次真实 decode_step 提交 KV，顺带产出下一轮的 prev_target_pred。

## 理论速记（greedy 投机 = 无损）

接受的每个 token 都等于 target 在该位置的 argmax（由 prev_target_pred / a 数组背书），bonus 同理。所以输出序列 ≡ target-only greedy。实测 100% 一致验证了实现。

## 工程备忘

- 服务器的 debug 心得：**先用小 γ（2）+ 短 N（16）找分叉点**，再看长序列是否衰减——分两类 bug（逻辑错 vs 状态错）
- `c.len = keep` 式回滚对 fp16 cache 安全（数据残留无害，后续覆盖）；KV4Cache 同样成立（部分组 staging 在 reset 时清理）
- speculative 引擎的显存：1.7B(3.4G) + 0.6B(1.2G) + caches ≈ 5G，24G 富余
