# 07 — 前缀缓存 TTFT（M9，物化路线修复后）

## 证据链

底稿 results/m9_prefix_cache.txt（3800-token 前缀 + 11 后缀，max_tokens=1 取首 token 墙钟）：

| 模型 | 冷（turn-1） | 命中（turn-2） | 加速 |
|---|---|---|---|
| 1.7B | 136.2ms | 49.9ms | **2.73×** |
| 8B | 558.3ms | 63.0ms | **8.86×** |

背景：M8 发现静默错答 bug（命中前缀只在 int4 池、prefill 不读池），曾禁用；
M9 物化路线修复后默认开（ENABLE_PREFIX_CACHE=True）。正确性由 tests/test_prefix_cache.py
锁定（hit==hit 精确确定、命中确实发生、共享前缀+新后缀、chunked prefill 同路径）。

## 复现

```bash
python benchmarks/07-prefix-cache/bench_prefix_cache.py                       # 1.7B
MODEL=models/Qwen3-8B python benchmarks/07-prefix-cache/bench_prefix_cache.py  # 8B
# CTX= 换前缀长度；需要 results/smooth_kv4_*.pt
```
