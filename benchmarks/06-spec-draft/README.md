# 06 — 投机：draft 提案、γ 扫描、温度税（M9 建，M10 复现）

## 证据链状态：脚本已入库（M10 闭合缺环①，M10 续补扫参台）

- `bench_spec_draft.py` — 8B+0.6B draft 的 γ 扫描，可选 adaptive 窗口（逐 verify 步轮询
  `scheduler.proposal_gamma` 并打印窗口轨迹）。
- `bench_spec_temperature.py` — 同一家族 off/on 对照，量化 T>0 概率比接受的吞吐税；
  提案器（ngram/lookahead/draft）× 提示家族（copy/natural/random）× 温度。
- `bench_spec_adaptive_tune.py` — 动态 γ 控制器的扫参台（monkeypatch `_adapt_gamma`）：
  窗口长度 W ∈ {1,2,3,4,6} × 折半落点 {γ//2 并停住 / 当前//2 级联 / γ−1 阶梯 / 一步到 1} ×
  阈值 {1.0, 1.5}，外加"手工把 γ 直接配成 2"的对照；`FAMILY=natural|copy` 两族各对自己的平解码基线。

M9 的三张底稿（`m9_spec_draft.txt`、`m9_gamma_sweep_draft.txt`、`m9_draft_fused.txt`）
当时是临时脚本跑的。M10 用 `bench_spec_draft.py` 重跑了一遍 **m9_draft_fused.txt**：

| γ | natural M9 → M10 | copy M9 → M10 |
|---|---|---|
| 1 | 105.2 → 99.5 | 123.4 → **122.9** |
| 2 | 114.4 → **116.5** | 159.2 → 158.8 |
| 3 | 98.0 → 103.9 | 176.5 → 175.5 |
| 4 | 97.5 → 89.0 | 184.1 → 183.4 |
| 6 | 81.7 → 86.6 | 202.4 → 202.7 |
| 8 | 78.6 → 73.2 | 225.1 → **224.7** |

**copy 六格全部在 ±0.6% 内复现；natural 抖动 ±6–9%。** 后者不是代码问题而是 4-bit 贪心
的混沌性——一次 argmax 翻转就换一条轨迹。所以 natural 列只能读成"γ≥3 都在 ≈1.0× 的噪声
带里"，不能读成"γ=4 显著差于 γ=3"。本项目一律用吞吐/接受率/PPL 做证据，不用逐 token 相等。
原始底稿：`results/m10_draft_sweep.txt`。

## 结论汇总（M9 建 + M10 复测）

**未融合基线**（results/m9_spec_draft.txt）：draft 步 4.53ms，CUDA-event 归因
100% GPU-bound（CPU 0.25ms 完全重叠）：GEMV 2.50ms@352GB/s + 小 kernel 1.70ms
+ lm_head 0.32ms。8B+0.6B：复读 143.5（1.47×）、自然 71.6（0.73×）。

**W4 draft 证伪**（results/m9_gamma_sweep_draft.txt）：GEMV 2.50→1.19ms
（840→210MB）但**步时不变**——2241 kernel/步，~1500 个未融合 elementwise 占 ~2.6ms。
**bs=1 时 kernel 数是税不是字节。**

**未融合 γ 扫描**（同底稿）：copy 峰 γ=6 156.0（1.59×）；natural 全 γ 劣化（γ=1 0.94×）。

**torch.compile 融合后**（results/m9_draft_fused.txt）：步时 4.53→**2.59ms（1.75×）**。

| γ | natural | copy |
|---|---|---|
| 2 | **114.4（1.17×）** ← natural 最优 | 159.2（1.62×） |
| 8 | 78.6（0.80×） | **225.1（2.30×）** ← copy 最优 |

⚠ M10 复测把"natural 最优"收窄到 **γ=2**（1.19×），γ≥3 一律回落到 1.0× 附近甚至以下。
"n-gram 统治复读、draft 补自然语言"这个分工依然成立，但 draft 的自然语言收益是**薄**的。

踩坑：dynamo 守卫编码 inference_mode → 首提案 5s 重编译 → `DraftProposer._warm()` init 预热。

## 动态 γ（M10 建，M10 续扫完两个先验）

`Scheduler._adapt_gamma`：WINDOW=3，**平均接受 ≤1.0 就把窗口折半到 γ//2**，其他情况不动；
只裁不涨，且只对按条付费的 draft 生效（ngram/lookahead 提案免费）。γ 上限被
`spec_num_drafts` 锁死——verify CUDA 图族按固定 `M=γ+1` 捕获，所以窗口不可能越过配置值。

8B natural、γ 上限 4、每策略 4 次重复（`results/m10_draft_sweep.txt`，原始读数保留）：

| 策略 | tok/s（4 次） | 倍率区间 | mean | 窗口轨迹 |
|---|---|---|---|---|
| 关闭自适应 | 89.0 | 0.91× | 0.91× | 恒 4 |
| **只裁不涨（=代码现状）** | 102.7 / 116.1 / 103.6 / 98.2 | 1.00–1.18× | 1.07× | 恒 [2, 2] |
| 曾试验的回升分支 | 90.3 / 109.1 / 88.3 / 97.0 | 0.90–1.11× | 0.98× | [2, 4] |

⚠ **这张表的因果读法已被 M10 续的 n=5 重测推翻**（见下节第 3 条）。回升分支的证伪仍然成立：

回升判据（`avg >= proposal_gamma - 0.1`）**被自己的复测量证伪并已回退**：4 次里 3 次跌破
1.0×。注意 HEAD 原本写的是 `avg >= self.gamma - 0.1`——阈值对着**上限**，而 `avg` 最大只到
**当前窗口**，折半后那条分支按定义不可达，所以旧代码事实上早就是棘轮；本次量的是可达写法，
它输了，于是**死分支删除而不是修复**，代码/测试/文档三处把这件事说明白。
机制解释（未证但自洽）：接受是**最长前缀**判据，一轮落在 2/2 对"第 3、4 条会不会被
接受"零信息，于是回升读的是噪声。要真正回收窗口需要**逐位置**接受率，当前 `avg` 只是逐序列
标量。copy 侧三种策略零差异（183±0.5，acc 0.76）——高接受场景控制器不介入，这是设计意图。

### 续扫：WINDOW 与"折半落点"（`results/m10_adaptive_tune.txt`）

`_adapt_gamma` 的 WINDOW=3 与 γ//2 都是旧引擎 M6 直移的先验，新 runtime 上从没扫过。
这次用 `bench_spec_adaptive_tune.py` 把三组轴一次扫完（γ 上限一律配 4，n=5/n=2 重复）。
先做了一件必要的事：**穷举 780 条接受长度流验证扫参台 `w3h` 与出厂代码判定完全一致**（0 处不同），
否则整张表没有基线。

1. **折半落点是这次唯一有决定性的轴**，任何"继续往下切"都净亏 20% 上下、区间极紧：
   出厂 `γ//2 并停住` 1.06 [1.01,1.14]；`当前窗口//2 级联` **0.79 [0.78,0.81]**；
   `一次只放弃一位` **0.81 [0.80,0.84]**；`一步裁到 1` **0.83 [0.78,0.90]**。
   机制算得出来：配置 γ=4 的步价 21.5ms、平解码 10.2ms ⇒ **盈亏平衡要求 tok/step ≥ 2.11**；
   窗口停在 2 实测 2.2（越过线），继续切到 1 只剩 1.68–1.76（跌回线下）。
2. **WINDOW 在 natural 上是一条平台**（W=1/2/3/4/6 → 1.05/1.05/**1.06**/1.04/1.02，互相覆盖），
   但**在 copy 上是悬崖**：W=1 第 4 步误触发 → **1.35× vs 1.85×（−27%）**，W=2 第 14 步误触发
   −24%，阈值放宽到 1.5 第 23 步误触发 −19%；而 W=3/4/6 各 10 次抽取**全部从不触发**。
   ⇒ 窗口的真实身份**不是反应速度旋钮，而是一次不可逆动作的误触发滤波器**（γ//2 落点幂等，
   所以窗口宽度只能把"那一次切割"推迟几步）。出厂的 3 恰好站在悬崖边沿——保持不动。
3. **裁窗口不省时间**（本文件的头号更正）。`llm_engine.py:102-104` 的注释自己写着 adaptive
   window 只缩短提案列表、`M` 仍是 `spec_num_drafts+1`；`DraftProposer` 的前向循环用的是**配置** γ
   （`draft.py:292`），全仓 grep 确认 `proposal_gamma` 从未流进提案器。实测形式：步价只有
   **21.5ms（配置 γ=4）** 和 **16.3ms（配置 γ=2）** 两档，**与运行时窗口停在哪无关**。
   所以"只裁不涨把 0.91× 救到 1.07×"这句撤掉：n=5 重测是钉上限 **0.98 [0.90,1.05]** vs 棘轮
   **1.06 [1.01,1.14]**，+18% 实为 +8%（Welch t=2.47，弱信号），且切割在固定轨迹上只可能变差
   （丢掉已付费的提案），那 8% 只能来自轨迹分叉，**未证**。
4. **缺的那块对照：手工把 γ 配成 2**（adaptive off）。natural **1.08 [1.02,1.17]**——与棘轮打平；
   但 copy 只有 **1.61**，比跑在 γ=4 上限上的 1.85–1.88 低 13%。⇒ "γ=4 + 只裁不涨"仍是更好的配置，
   但理由换成"它让一个配置值同时保住 copy 的上限，又不像在 natural 上那样一路跌到盈亏线以下"。
5. **混沌带首次量化**：同配置、同贪心、无随机源，natural 投机跑 5 次 tok/step 1.91–2.22（±8%）、
   步数 86–100；平解码 5 次全 191 步、0.9% 抖动；copy 投机连步数都次次相同（48/48，10 次抽取）。
   ⇒ **natural 上任何 <5% 的格间差不该读成结论**（上面第 1 条的平台判据就按这条给）。
   copy 稳而 natural 飘的原因：**未证**。
6. **一行模型**（25 格全部吻合，最大偏差 0.003）：`x = tok/step × 平解码 ms/step ÷ 该格 ms/step`。
   ⇒ 这次扫参里所有倍率差异都只是 tok/step 的差异，而 ms/step 只取两档。

**决定：代码一个数字都不改**（W=3、γ//2、阈值 1.0、只裁不涨）——扫参没给出可替换的更优格子，
只给出一条被证伪的解释（第 3 条）和一块真空间：**让早停真的省钱**（提案器按 `cap` 提前结束串行
解码，或为每个 M 各录一族 verify 图）。这条的收益**没测**，只有外推上限（窗口 2 的 tok/step 2.25
若配上 16.3ms 的步价，natural 会到 ~1.4×；但 21.5→16.3 那 5.2ms 里"多 2 次提案前向"与"更宽
verify"各占多少没拆开，**未证**）。

## 温度税（M10 实测，缺环④）

T>0 走 Leviathan 概率比接受（分布无损，正确性见 `tests/runtime/engine/test_spec_acceptance.py`，
N=20000 实测 TV<0.06）。代价首次量化（`results/m10_spec_temperature.txt`）：

| 提案器 | 家族 | greedy | T=0.7 | 税 |
|---|---|---|---|---|
| ngram | copy | 3.06× | 2.68× | +12.4% |
| ngram | natural | 1.45× | 0.95× | +34.4% |
| lookahead | copy | 3.54× | 1.31× | +63.0% |
| draft | copy | 1.10× | 1.05× | +4.4% |
| draft | natural | 1.13× | 0.70× | +38.1% |

税的来源**不是**那份 `[bs·(γ+1), V]` float32 提案分布（lookup 类提案是一-hot，走不到分配，
peak 5.66→5.67 GB），而是**接受率塌了**：T>0 时"逐位等于贪心前缀"不再成立，lookahead 的
自然语言接受率 0.62→0.00（tok/step 退化成 1.00，等于没有投机）。
随机提示家族上 T=0.7 出现的 3.3×/3.8× 是**轨迹假象**——探针显示该温度下模型输出周期为 1 的
重复 token（56/64 个 195），任何提案器都能命中；不要用带随机字的 prompt 评投机。

结论：**T>0 时免费提案器（ngram/lookahead）全线劣于 greedy 同配置**，只有 draft 勉强保住
1.0× 附近；真要温度解码，γ 应该配小（≤2）或者干脆关投机。

## 复现

```bash
# 必须先 activate：只 export PATH 时 CUDA_HOME 缺失，Marlin 扩展二次 JIT 失败
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qslab

# γ 扫描（8B W4A16KV4 + 0.6B draft，两 int4 池共卡：UTIL=0.62 / DRAFT_UTIL=0.9）
CUDA_VISIBLE_DEVICES=1 GAMMAS="1 2 3 4 6 8" ADAPTIVE=off \
  python -u benchmarks/06-spec-draft/bench_spec_draft.py

# 同上，打开自适应窗口（会打印窗口轨迹区间）
CUDA_VISIBLE_DEVICES=1 GAMMAS="4 8" ADAPTIVE=on \
  python -u benchmarks/06-spec-draft/bench_spec_draft.py

# 温度税（全部 1.7B 规模；免费提案器走 fp16 权重，draft 行走 1.7B W4 + 0.6B draft）
CUDA_VISIBLE_DEVICES=1 TEMPS="1e-6 0.7" METHOD="ngram lookahead" TOKENS=256 \
  python -u benchmarks/06-spec-draft/bench_spec_temperature.py
CUDA_VISIBLE_DEVICES=3 MODEL=models/Qwen3-1.7B W4=models/Qwen3-1.7B-qslab-w4-awq2 \
  METHOD=draft PROMPTS="copy natural" TEMPS="1e-6 0.7" TOKENS=192 \
  python -u benchmarks/06-spec-draft/bench_spec_temperature.py

# 动态 γ 扫参（γ 上限配 4；三组轴一次扫完，约 19 分钟 / 5 分钟）
CUDA_VISIBLE_DEVICES=1 REPEATS=5 FAMILY=natural \
  python -u benchmarks/06-spec-draft/bench_spec_adaptive_tune.py
# 误触发对照：copy 上 W=1/2 会掉 24–27%，W>=3 十次抽取从不触发
CUDA_VISIBLE_DEVICES=1 REPEATS=8 POLICIES="w3h w4h w6h" FAMILY=copy \
  python -u benchmarks/06-spec-draft/bench_spec_adaptive_tune.py
# 缺的那块对照：手工把 γ 直接配成 2（adaptive off）
CUDA_VISIBLE_DEVICES=1 GAMMA=2 POLICIES=off REPEATS=5 \
  python -u benchmarks/06-spec-draft/bench_spec_adaptive_tune.py

# 单测（纯 CPU，无模型）：接受律无偏 + 窗口是单向棘轮 + 窗口是误触发滤波器
python -m pytest tests/runtime/engine/test_spec_acceptance.py -q
```

DraftProposer 硬编码 `compile=True`，因此本目录所有表都对应 **融合**构建，
与 `m9_gamma_sweep_draft.txt`（未融合）不可直接比。
