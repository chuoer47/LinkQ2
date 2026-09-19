# 09 — batch-size 吞吐扫描（当前 runtime）

脚本：`bench_batch.py`。底稿：`results/m12_batch_1.7b.txt`、`m12_batch_1.7b_spec.txt`、
`m12_batch_8b.txt`、`m12_batch_8b_spec.txt`（2026-09-19，4090_public GPU3）。

## 为什么缺这张表

仓库里唯一的 decode 吞吐基准 `benchmarks/01-decode-baseline/bench_throughput.py`，
按它自己的 docstring 就是 **batch=1** 口径，而且驱动的是已冻结的 M0–M7 `QslabEngine`。
M8 之后的 live runtime 基准（05/06/07）全都配了并发（`max_num_seqs=8`）却只调一次
`add_request`（`bench_spec_ngram.py:60`）。所以 `results/` 里**从来没有 bs>1 的读数**：
调度器、CUDA graph 分桶、paged KV 池在并发下的表现一直没被定价。

## harness 的三个设计点

1. **口径与出厂逐字相同**：decode-window 公式抄 `bench_spec_ngram.py:63-78`——只累加
   `step()` 返回负数的步（prefill 不计）。另加一列 `tok/s(wall)`（含 prefill 的总墙钟），
   因为 vLLM 的离线 API 只有这一种口径，它是 `benchmarks/10-vllm-compare/` 唯一能对齐的列。
2. **自检门（默认开，`SELF=1`）**：报任何 bs>1 之前，先用**出厂 prompt + 出厂 token 数**
   复测 bs=1；偏差 > `TOL`(5%) 就 `sys.exit(1)`，bs>1 一格都不出。四格全部通过：
   1.7B off Δ0.4%、1.7B ngram Δ3.2%、8B off Δ0.7%、8B ngram Δ0.1%。
   规则同项目惯例：**扫描工具必须先证明自己量的就是主线量的东西**。
3. **prompt 去前缀共享**：每条 prompt 前面塞一个互不相同的随机 token id（`seed=11`）。
   `BlockManager.compute_hash` 链式哈希父块（`block_manager.py:43-50`），首块不同 ⇒ 整条
   序列后续块全不命中。不这么做的话 N 条相同 prompt 会共享绝大部分 KV 块，扫到的是
   缓存而不是批量。vLLM 对照脚本用**同一个 RNG、同一批 token id**（`bench_vllm.py`）。

## 复现

环境前置（必须 `conda activate qslab`，纯 fp16 感觉不到但 W4 格子会跑挂）见 `../README.md`。
跑前先看 `nvidia-smi`，0/2 号是别人的。

```bash
# 1.7B fp16 权重 + KV4，关投机 / 开 n-gram γ=4
MODEL=models/Qwen3-1.7B UTIL=0.5 TOKENS=256 BATCHES=1,2,4,8,16,32 TAG=Qwen3-1.7B \
  python benchmarks/09-batch-throughput/bench_batch.py
SPEC=ngram GAMMA=4 ...同上...

# 8B W4A16KV4（W4=1 触发 pack 切换，工具链环境变量必须齐）
W4=1 MODEL=models/Qwen3-8B UTIL=0.5 TOKENS=192 BATCHES=1,2,4,8,16 TAG=Qwen3-8B \
  python benchmarks/09-batch-throughput/bench_batch.py
SPEC=ngram GAMMA=4 ...同上...
```

## 读数（decode-window tok/s；括号内为 tok/s(wall)）

| bs | 1.7B off | 1.7B ngram | 1.7B 投机/关 | 8B off | 8B ngram | 8B 投机/关 |
|---|---|---|---|---|---|---|
| 1 | 146.2 (143.6) | 278.1 (268.6) | 1.90× | 97.9 (92.0) | 313.5 (260.7) | 3.20× |
| 2 | 266.3 (261.6) | 373.9 (365.1) | 1.40× | 184.9 (171.7) | 583.9 (490.2) | 3.16× |
| 4 | 527.5 (518.9) | 732.3 (711.3) | 1.39× | 361.6 (341.2) | 971.2 (810.3) | 2.69× |
| 8 | 1031.7 (1014.9) | 1137.3 (1116.4) | 1.10× | 687.8 (651.3) | 1577.6 (1356.9) | 2.29× |
| 16 | 1902.9 (1874.2) | 1709.0 (1684.8) | **0.90×** | 1271.9 (1208.5) | 1912.6 (1772.9) | 1.50× |
| 32 | 3572.8 (3516.0) | 2736.2 (2703.0) | **0.77×** | — | — | — |

批量加速比（相对本表 bs=1，decode-window 口径）：
1.7B off **24.43×**（bs32，理想 32×）、1.7B ngram 9.84×、8B off **12.99×**（bs16，理想 16×）、8B ngram 6.10×。

## 结论

1. **关投机时批量近似线性**：1.7B 到 bs=8 拿到 7.06×（理想 8×），8B 到 bs=8 拿到 7.03×。
   之后开始掉：每请求吞吐 1.7B 从 146 → 112 tok/s、8B 从 98 → 80 tok/s。掉的是**算力/带宽**，
   不是调度（`min/max` 一列显示每步都吃满 bs 个 token，步数恒定，无抢占抖动）。
2. **投机与批量抢的是同一份算力，且这一抢就把投机的收益吃光**：1.7B 的 n-gram 收益从 bs=1
   的 1.90× 一路降到 bs=16 的 **0.90×**、bs=32 的 **0.77×**——**高并发下开 n-gram 是净负收益**。
   8B 在同一区间仍为正（bs=16 时 1.50×），因为它每步的算力余量更大。
   这否证了"投机和批量各自都能拿一半收益"的直觉，也是本项目目前**没有**并发-投机联动策略的理由
   （已登记 TODO 12）。
3. **KV 池容量就是并发上限**（TODO 11 的实测顺带产出）：8B 池只有 **431 块 = 55,168 tokens**，
   按 `max_model_len=4096` 只容得下 **13 条**满长序列；1.7B 池 1062 块 = 135,936 tokens。
   本扫描每条只用 289 tokens（33 prompt + 256 生成），离触顶很远 ⇒ 表内没有 preempt 影响。
4. **over-billing 从"归属未验证"变成实测事实**：`allocate_kv_cache` 按
   `slot_bytes = 2*head_dim`（`model_runner.py:123-124`）计费，而真实缓冲是
   kq(int4)+vq(int4)+vs(fp16) = **132 B/head/layer**（`:139-147`）⇒ 两个模型都是
   **1.94× 超订**。8B 的 3.79 GiB 预算里只有约 2.05 GiB 真被分配，等于凭空少 1 倍的池。

## 口径边界（别越界解读）

- **扫描的 bs=1 投机读数 ≠ 出厂 bs=1 读数**：1.7B 278.1 vs 自检 441.3（−37%），
  8B 313.5 vs 310.6（+0.9%）。两次 prompt 除首 token 外完全相同，所以差异只能来自生成序列
  本身对 n-gram 命中面的影响；**机制未证**。⇒ 本表的加速比只在表内自洽，不可与
  `results/m9_*.txt` 互比，也不可拿"1.7B 投机只有 1.90×"去推翻出厂的 3.13×。
- **1.7B 每序列接受率随 bs 掉**（`tok/step ÷ bs` = 2.16 → 1.50 → 1.53 → 1.28 → 1.18 → 1.19），
  而 8B 恒在 3.54（= γ+1 上限内）。`NGramProposer` 是逐序列查自己的 `token_ids`、无跨序列状态
  （`ngram.py:51`），所以下降不来自提议器共享。两个模型的其他差异（权重精度、token 数、
  接受窗口是否饱和）**未逐一定责** ⇒ 登记为 TODO 13 的开放问题。
- 1.7B 未扫 bs>32：`graph_bs` 分桶在 32 之后是 16 一格，且并发继续涨就要开始吃 prefill 重算。
- 8B 未扫 bs=32：单卡 W4A16KV4 下 32×289 tokens 仍在池内，但 32 路 verify 的中间激活
  会顶到显存上限，属于另一个实验设计（要改 `max_model_len`），没有顺手跑。
