# 08 — 长上下文：YaRN 分档 NIAH（M11）

## 这个目录为什么存在

TODO 从 M4 起挂着一条"128K YaRN demo 未跑"。M11 去跑之前先撞到一个更基本的事实：
**runtime 没有 YaRN**——`rope_scaling` 在 `Qwen3Attention` 被接收后直接丢弃，配置写了也不生效，
而且不报错（静默无效）。所以本目录的证据链包含两件事：把 rope 外推真正实现（并有逐位对照锁），
以及在真实的 128K 位置上看这套 W4A16 + KV4 还能不能读写出结果。

## 证据链

底稿 `results/m11_yarn_niah.txt`（两轮原始输出 + 答案全文探测 + 读数），模型
`models/Qwen3-8B` + W4A16 + KV4，`UTIL=0.8`，每档 4 深度 × 2 次 = 8 次，greedy，投机 off。

| 档位 | rope | 冷 prefill | decode | NIAH recall | peak | 池对当档余量 |
|---|---|---|---|---|---|---|
| 32768 | native | 6.7 s | 13.3 tok/s | 6/8 = 75% | 12.97 GiB | 4.8× |
| 32768 | yarn 3.2 | 6.7 s | 13.3 tok/s | 8/8 = 100% | 12.89 GiB | 4.7× |
| 65503 | yarn 1.6 | 17.8 s | 7.1 tok/s | 7/8 = 88% | 12.92 GiB | 2.3× |
| 131039 | yarn 3.2 | 53.2 s | 3.7 tok/s | **3/8 = 38%** | 13.38 GiB | **1.2×** |

三条能读的、两条不能读的，说清楚：

* **能读**：① 这套 runtime 在 131K 位置上端到端跑通了，且速度曲线是产品级数字——
  decode 每次长度翻倍约减半（13.3→7.1→3.7），冷 prefill 每次 ×2.7/×3.0；
  ② 128K 档明显比其他三档差（3/8 vs 其余合计 21/24，Fisher 单侧 p≈0.012）；
  ③ **128K 的漏是"找到了但抄错"**——5 次 miss 里 4 次是数字前缀全对、尾部丢
  （`92142` vs 921426、`74403` vs 744039、`3341` vs 334130），1 次把键里的数字抄成了答案，
  且每次都自己吐终止符收尾（ntok 22–25，远小于生成预算）⇒ 塌点在**转写**，不在注意力查表。
* **不能读**：① **YaRN 对召回的贡献：未证**。32K 那两列是同一份代码、同一批 needle，
  native 6/8 vs yarn 8/8——换一下 haystack 相位就能动整格（64K 列两轮 8/8→7/8），
  所以这 2 例差不记账给 rope；② "128K 差"也不等于"YaRN 不行"，因为本轮**没有** 128K 的
  可比基线（不缩放 rope 跑到 131K 是越界外推，不是一个基线数字）。
* **顺带暴露的评分口径缺陷**：`value in answer` 把"定位对、抄错末位"和"完全没找到"记成同一个
  miss。本轮不改脚本（改了就没有可比的两轮），已进 TODO：长上下文评分需要逐位/编辑距离口径。

## 正确性锁（不靠这份底稿）

YaRN 的实现与 transformers 4.57.6 逐位一致，锁在 `tests/runtime/model/test_rotary_yarn.py`（12 条）：
`inv_freq` `atol=1e-9`、cos/sin 缓存 `rtol=0, atol=0`、`attention_factor = 0.1·ln(f)+1.0`
（f=3.2 → 1.1163150809805682）严格相等，位置取 0/1/7/400/5000/20480/40959/131071。
另有两条**防静默**的锁：① native 路径的 rope 缓存与实现 YaRN 之前**逐位相同**
（`attention_scaling == 1.0`），所以这次重写不会悄悄改动其他里程碑的所有数字；
② 非 yarn 的 `rope_type` 现在**抛 `NotImplementedError`** 而不是被忽略——
一次没生效的长度外推会跑完、给出好看的数字、然后什么都不能证明。

配置侧：`Config.rope_scaling` 是新的入口（`{"rope_type": "yarn", "factor": 3.2}`），
它在 `__post_init__` 里把 `hf_config.max_position_embeddings` 改写成 `original × factor`
**再**过原有的 `max_model_len` clamp，所以"天花板"只有一处定义，bench 和 engine 不会各说各话。
factor 必须 > 1.0（这个字段只用于外扩，不外缩）。

## 刻意不测的两件事

1. **未缩放 rope 在 64K/128K 的负对照**——越界外推不是基线，见上。
2. **长上下文 PPL**——需要逐位置 logits，131072 × vocab 151936 是 40 GB 量级，
   24 GB 卡上这套 runtime 结构上给不出（PPL 主线在 `03-kv4-quality/`，那是短序列口径）。

## 显存预算（想复现前先读这段）

KV 池按每 token `36 层 × 8 KV头 × (2·head_dim=256) B = 72 KiB` 计费
（`qslab/runtime/execute/model_runner.py:123-124`），而 int4 载荷本身只要 36 KiB/token
⇒ 当前计费约为真值的 2×，多出的一半归属（scales/对齐）**未逐字节拆开验证**。
后果：128K + 32 生成 ≈ 9.0 GiB 池，`UTIL=0.8` 给到 10.56 GiB，`max_num_seqs=2` 已贴着上限
（表中"1.2×"就是它）。再往上加长度或并发，要么抬 `UTIL`，要么先修池计费口径。
engine 建好后还有一笔**逐层** fp16 前缀物化（`_materialize_prefix`，只在命中前缀缓存时走），
128K 档实测 peak 13.38 GiB。

## 复现

```bash
# 只有一张空卡时才跑；0/2 号是别人的作业
CUDA_VISIBLE_DEVICES=3 \
  MODEL=models/Qwen3-8B W4=models/Qwen3-8B-qslab-w4-awq CALIB=results/smooth_kv4_qwen3-8b.pt \
  TIERS="32768:native 32768:yarn:3.2 65536:yarn 131072:yarn" \
  DEPTHS="0.25,0.5,0.75,0.9" TRIALS=2 GEN=32 UTIL=0.8 \
  python benchmarks/08-long-context/bench_niah_yarn.py
```

env 旋钮：`TIERS`（`<ctx>[:native|:yarn[:factor]]`，空格分隔多档；不给 factor 时按
`ceil(ctx/native*100)/100` 自动取，64K→1.6、128K→3.2）、`MODEL/W4/CALIB`、`DEPTHS/TRIALS/GEN`、
`UTIL`、`SEED`。超过该 rope 配置天花板或池装不下的档位会打印 `SKIPPED` 并继续。

## 本目录立下的两条口径（下次写 bench 前先看）

1. **同列 trials 的 haystack 必须相位错开**，否则前缀缓存会命中：第一轮每列 prefill 从
   6.7 s 衰减到 1.0 s，列均值全是假数（第二轮改 `build_prompt(..., phase=)` 后才稳成常数）。
2. **打印答案要宽到能看见数字**（≥60 字符）并单独抽 `got=` 与 `exp=` 并排：
   这族 prompt 开头 24 字符恒定，只印 24 字符时所有 miss 长得一模一样、无法归因。
3. 附一条与 M10 对称的教训：M10 说"不要用带随机字的 prompt 评投机"，本轮补
   "不要用逐位转写长数字评长上下文检索"——它会把转写通道的脆性记在检索头上。
