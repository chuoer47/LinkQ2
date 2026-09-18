# 05 — n-gram 投机（新 runtime，M9）

## 证据链

底稿 results/m9_spec_1.7b.txt、results/m9_spec_8b.txt（γ=4，graph，greedy）：

| 模型 | 文本 | off (基线) | on | 加速 | 接受 |
|---|---|---|---|---|---|
| 1.7B fp16KV | 复读 | 145.5 | 455.9 | **3.13×** | 3.54/步 |
| 1.7B | 自然 | 145.5 | 212.3 | 1.46× | 1.62 |
| 1.7B | 随机 | 140.6 | 180.8 | 1.29× | 1.44 |
| 8B W4A16KV4 | 复读 | 97.2 | **310.4** | **3.19×** | 3.54 |
| 8B | 自然 | 98.0 | 93.2 | 0.95× | 1.05 |
| 8B | 随机 | 95.0 | 89.5 | 0.94× | 1.05 |

- 复读输出 128/128 与非投机逐 token 一致（greedy 数学等价）
- **诚实口径**：8B 自然/随机 ~5% verify 开销（vLLM ngram 同样报告）
- 底稿里的 `offM` 行 = Marlin 强制路径对照（当时验证 crossover 修正用）
- 8B off=97.2 即现行主线 decode 基线（1.75× vs M4 时代 55.8）

结论：n-gram 提案免费，统治复读/抽取式负载；自然文本收益低但 1.7B 仍 1.46×。

## 复现

```bash
# 1.7B
python benchmarks/05-spec-ngram/bench_spec_ngram.py
# 8B W4（m9_spec_8b.txt 的复现）
MODEL=models/Qwen3-8B W4=models/Qwen3-8B-qslab-w4-awq TOKENS=192 GAMMA=4 TAG=Qwen3-8B \
  python benchmarks/05-spec-ngram/bench_spec_ngram.py
# γ 可用 GAMMA= 换；需要 results/smooth_kv4_*.pt 校准（已在 results/）
```
