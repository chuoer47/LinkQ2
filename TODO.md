# TODO — 证据链缺环与收尾遗留（2026-09-18 整理；同日 M10 复核后更新；09-19 补测后再次更新；09-19 M11 闭合遗留 7 并新账 10/11）

## 缺环（证据链不完整处）

1. ~~**draft 投机 bench 脚本未入库**（06-spec-draft/）：
   results/m9_spec_draft.txt、m9_gamma_sweep_draft.txt、m9_draft_fused.txt 无对应脚本。
   建议：按 benchmarks/05-spec-ngram/bench_spec_ngram.py 结构补 bench_spec_draft.py
   （spec_method="draft" + GAMMA 扫描 + 融合开关对照）。~~ — **09-19 入库并复现**：
   `benchmarks/06-spec-draft/bench_spec_draft.py`（γ 扫描 + `ADAPTIVE` 开关 + 逐 verify
   步轮询 `scheduler.proposal_gamma` 打印窗口轨迹）+ 底稿 `results/m10_draft_sweep.txt`。
   **copy 列六格在 ±0.6% 内复现** m9_draft_fused.txt（最大 184.1→183.4）；natural 列
   抖动 ±6–9%（γ=3 98.0→103.9、γ=4 97.5→89.0），是 4-bit 贪心的混沌轨迹而非代码差异，
   因此 natural 只读成"γ≥3 在 ≈1.0× 噪声带里"。
2. ~~**m9_prefix_cache.txt 只有 1.7B 数字**，8B 的 558.3→63.0ms 只在 notes/PROGRESS
   里（当时跑过）~~ — **09-19 重跑落盘**：该文件追加了两段带命令的原始输出，8B
   cold 555.9ms → hit 62.3ms（**8.92×**），与 M9 口录的 558.3/63.0 差 0.4%/1.1%。
   顺带记下一条此前没写清的事实：bench 无 W4 开关，8B 走 fp16 权重，需
   `UTIL=0.9` 才放得下（`peak=16.91GiB`，KV 池只剩 3.69GiB）。
3. ~~m2_niah.json 只覆盖 @1K/@2K，32K 无独立底稿~~ — **M10 复核后不成立**：
   `results/m2_niah.json` 本身就是 ctx=32768 的落盘（fp16 与 kv4 各 3 深度 ×3 次，
   overall_recall 均 1.0），产出脚本 benchmarks/01-decode-baseline/bench_niah.py 在库。
   真正偏弱的只是探针密度（9 次/配置），不算缺环。

### M10 带来的新缺环

4. ~~**温度投机只有正确性证据，没有吞吐证据**：接受律已换成 Leviathan 概率比
   （tests/test_spec_acceptance.py 实测分布无偏：N=20000、T=0.9 时 TV<0.06），
   但概率比路径每步要多一份 `[bs·(γ+1), V]` float32 提案分布和一次 softmax，
   **加速比从未测过**。建议：06-spec-draft 下补 T>0 的 n-gram/draft 吞吐对照。~~
   — **09-19 已实测**：`benchmarks/06-spec-draft/bench_spec_temperature.py` +
   `results/m10_spec_temperature.txt`。**税不在那份 float32 分布**——lookup 类提案是
   one-hot，走不到分配（peak 5.66→5.67 GB）；税在**接受率塌**：ngram copy 3.06→2.68×
   （+12.4%）、lookahead copy 3.54→1.31×（+63.0%）、lookahead natural acc 0.62→0.00
   （tok/step 退化成 1.00，等于没投机）、draft copy 1.10→1.05×（+4.4%）。
   随机提示家族上 T>0 的 3.3×/3.8× 经探针（`/tmp` 一次性脚本，未入库）证实是**轨迹
   假象**（该温度下输出周期为 1 的重复 token），已从结论剔除。**口径教训：不要用带
   随机字的 prompt 评投机。**
5. ~~**动态 γ 的阈值未在新 runtime 上调过参**：WINDOW=3、"平均接受 ≤1.0 折半 /
   ≥γ-0.1 回升"是旧引擎 M6 直移的先验；γ 上限被 `spec_gamma` 锁死（verify 图族
   M=γ+1 固定），所以"高接受率时窗口能不能再放大"这件事当前无从验证。~~
   — **09-19 已实测并据此改判**：`results/m10_draft_sweep.txt`（8B natural，γ 上限 4，
   每策略 4 次重复）——只裁不涨 **1.00–1.18×（mean 1.07×）**、固定窗口 0.91×、把旧引擎
   的可达写法 `avg >= proposal_gamma - 0.1` 接回来则 0.90–1.11×（mean 0.98×，4 次里 3 次
   跌破 1.0）。**回升被自己的复测量证伪**：HEAD 原分支写的是 `avg >= self.gamma - 0.1`，
   阈值对着上限而 `avg` 最大只到当前窗口，折半后**按定义不可达**——旧代码事实上已是棘轮，
   只是挂着死分支和一段不成立的"滞回防抖"说辞；本次把死分支**删除而非修复**，
   `tests/test_spec_acceptance.py` 改为断言单向棘轮。机制（未证但自洽）：接受是**最长
   前缀**判据，一轮落在 2/2 对"第 3、4 条会不会被接受"零信息，回升读的是噪声；要回收窗口
   需要**逐位置**接受率，当前 `avg` 只是逐序列标量。
   两处更正/遗留：γ 上限的**运行时字段**是 `config.spec_num_drafts`（`spec_gamma` 只是 L4
   门面的参数名，旧文两者混用）；
   ~~WINDOW=3 与折半系数没扫过~~ — **09-19（M10 续）已扫完**：
   `benchmarks/06-spec-draft/bench_spec_adaptive_tune.py` + `results/m10_adaptive_tune.txt`
   （先穷举 780 条接受流证明扫参台与出厂代码判定 0 处不同，再扫）。**代码一个数字都不改**，
   三条结论：①**折半落点是唯一决定性轴**，任何"继续往下切"净亏 20%（级联 0.79、γ−1 阶梯
   0.81、一步到 1 的 0.83），因为配置 γ=4 的步价 21.5ms vs 平解码 10.2ms ⇒ 盈亏平衡要
   tok/step ≥2.11，窗口停在 2 有 2.2、切到 1 只剩 1.7；②**WINDOW 不是反应速度旋钮而是
   误触发滤波器**（γ//2 落点幂等 ⇒ 窗口只推迟那一次切割），natural 上 W=1..6 是一条平台，
   但 copy 上 W=1/2 分别因第 4/14 步误触发掉 27%/24%，阈值放宽到 1.5 掉 19%，而 W≥3
   各 10 次抽取从不触发 ⇒ 出厂的 3 恰在悬崖边沿；③**本项最大的更正**：裁窗口**不省时间**
   （`DraftProposer` 用配置 γ 循环，`proposal_gamma` 从未流进提案器；步价只有 21.5/16.3ms
   两档，与运行时窗口无关），所以上面"0.91× 救到 1.07×"那句因果解释作废——n=5 重测为
   钉上限 0.98×[0.90,1.05] vs 棘轮 1.06×[1.01,1.14]（+8%，Welch t=2.47），且切割在固定
   轨迹上只可能变差，那 8% 归属轨迹分叉、**未证**；手工把 γ 配成 2 的对照显示 natural 打平
   （1.08）但 copy 掉 13%（1.61），这才是"γ=4 + 只裁不涨"真正的理由。
   "窗口能否越过配置值"结构上仍无从验证（verify 图族按固定 M=γ+1 捕获）。
   真正的优化空间变成一条新任务：**让早停真的省钱**（提案器按 `cap` 提前结束串行解码，
   或为每个 M 各录一族 verify 图）；收益未测，外推上限 natural ~1.4×。
6. ~~**Marlin 路径双份 int4 常驻只有一句口算**（M10 修门面测试时暴露）：
   `w4.auto` 名义上按 M dispatch，实际 `CROSSOVER_M=0`，v1 pack 全程闲置，
   而 `w4.marlin` 另存一份 Marlin 排布的 `_B/_s`。口算给的"8B ≈ 1.9 GB"从未
   进测。~~ — **09-19 已实测**：`benchmarks/02-w4-quant/bench_w4_residency.py` +
   `results/m10_w4_residency.txt`，1.7B 死重量 0.677 GB（+15% KV 池）、8B
   **3.335 GB**（1.52→4.85 GB，可换出约 2.1 万 → 6.6 万 token 总容量）。口算
   低了近一倍，已更正。**注意**：`w4.auto` 与 `w4.marlin` 逐列相同——省不掉
   这一份是 Marlin 后端本身的性质，与 dispatch 策略无关。同日该改动**已实现并
   复测**（后端 `uses_v1_pack` + `W4Linear` 按需注册 buffer）：8B 常驻
   6.674→3.339 GB（0.52×→0.26× fp16），"只有走 `w4.v1` 才省得下"随之作废；
   而 "+220% / 4.85 GB" 的换算也错了——腾出的字节进的是分配预算
   （2.94→6.14 GiB），池子只兑现其中 ~51%，实测 1.52→**3.16 GB（+108%）**。

> **缺环 1–6 于 2026-09-19 全部闭合**（底稿：`results/m10_w4_residency.txt`、
> `m10_draft_sweep.txt`、`m10_spec_temperature.txt`，以及追加 8B 段的
> `results/m9_prefix_cache.txt`）。收尾遗留 7 于同日 M11 闭合（`results/m11_yarn_niah.txt`）；
> 剩下的 9/10/11 是功能与新账，不是证据链问题。

## 收尾遗留（功能，非证据链）

7. ~~**128K YaRN demo 未跑**（M4 起遗留）~~ — **M11（09-19）已跑，且前提先修了一遍**：
   runtime **没有 YaRN**——`rope_scaling` 在 `Qwen3Attention` 收到后被直接丢弃，配置写了
   不生效也不报错。本轮先实现（`rotary.py`/`config.py`/`qwen3.py`，与 transformers 4.57.6
   **逐位相同**，12 条锁在 `tests/test_rotary_yarn.py`），再按 32K/64K/128K 分档跑 NIAH：
   底稿 `results/m11_yarn_niah.txt`、脚本 `benchmarks/08-long-context/bench_niah_yarn.py`。
   读数：128K 端到端跑通，decode **13.3 → 7.1 → 3.7 tok/s**、冷 prefill **6.7 → 17.8 → 53.2 s**、
   peak 13.38 GiB；召回 3/8（其余三档合计 21/24，Fisher p≈0.012），但**漏的形态是"数字前缀
   全对、尾部丢"**（`92142` vs 921426）⇒ 塌点在转写不在检索；**YaRN 对召回的贡献未证**
   （32K 两列同代码同 needle，换 haystack 相位就能整格移动）。
8. ~~models/Qwen3-0.6B-qslab-w4-awq2~~ 已删（A 档清理 11795fd）；tests/ 4 个辅助脚本已移 scripts/archive/m-verification/。
9. **让动态 γ 的"早停"真的省钱**（M10 续扫参后新账，见缺环⑤第③条）：当前裁窗口只丢弃
   已付费的提案，控制器没有成本通道。两条路——①`DraftProposer` 按 `cap` 提前结束它的 γ 次
   串行解码（本来就只取前若干条，天然可截）；②为每个 M 各录一族 verify 图，让窗口能真的换图。
   ①便宜且不需要重录图族。**收益未测**：只有外推上限（窗口 2 的 tok/step 2.25 若配上 γ=2
   那档 16.3ms 的步价，natural 会到 ~1.4×；21.5→16.3 的 5.2ms 里"多 2 次提案前向"与
   "更宽 verify"各占多少没拆开）。落地后"回升"才重新值得测（需要逐位置接受率，见缺环⑤）。
10. **长上下文评分口径**（M11 新账，底稿 `results/m11_yarn_niah.txt` §5③）：`value in answer`
    把"从 12 万 token 外把 6 位数字捞回来、末位抄错"和"完全没找到"记成同一个 miss，
    于是每次底稿都要人工重读一遍答案才敢下结论。需要逐位/编辑距离口径（或至少分开统计
    "前缀对长度"）。本轮**没改脚本**——改了就没有可比的两轮；128K 档 38% 这个数因此是
    严格匹配下界，不是"检索成功率"。
11. **KV 池按 72 KiB/token 计费，是 int4 载荷的约 2×**（M11 读代码发现，
    `model_runner.py:123-124` 的 `slot_bytes = 2 * head_dim`）：36 层 × 8 KV 头 × 128 × 0.5 B
    × 2(K,V) = 36 KiB 就够。多出的一半归属（per-channel scales / 对齐）**未逐字节拆开验证**。
    后果直接可见：128K + 32 生成要吃 9.0 GiB 池，`UTIL=0.8` 只剩 1.2× 余量、
    `max_num_seqs=2` 贴着上限 ⇒ 修掉计费口径约等于再翻一倍上下文/并发。**收益未测**。

## M10 已完成（2026-09-18，功能遗留全部落地）

- [x] **L4 门面接新 runtime**：`api/llm.py` 的 `LLM` 改为包 `LLMEngine`，
      `api/cli.py` 重写为 runtime 词表（--w4/--smooth-kv/--spec/--draft/--stats-only）；
      6 个门面测试（tests/test_api_facade.py）+ CLI 两条实跑冒烟（greedy ngram、T=0.8 lookahead）
- [x] **温度采样投机**：`ModelRunner.rejection_verify` 概率比接受 + 拒绝后
      `norm(max(0,q-p))` 重采样；draft proposer 下发提案分布（float32，防 fp16 下溢把
      拒绝变成必受）；greedy 路径逐行不变（15 个既有 spec/draft e2e 测试原样绿）
- [x] **lookahead 迁移**：`runtime/ngram.py::LookaheadProposer`（持久逐序列索引、
      链式延伸、首次出现优先），`--spec lookahead` 可达
- [x] **动态 γ**：`Scheduler._adapt_gamma`——只裁剪提案条数、不动 verify 图的 M，
      故无需新图族；仅对按条付费的 draft 生效（ngram/lookahead 提案免费）
- [x] **8B+draft e2e 固化**：`tests/test_8b_acceptance.py::test_8b_draft_spec_acceptance`
      （0.62/0.9 显存分配，两 int4 池共卡；实跑 33.41s 通过）
- [x] 旧 `qslab/engine/spec/`（lookahead/dynamic）**逻辑已迁移**，旧包保留为冻结参照

## M10 补测（2026-09-19，缺环 1/4/5/6 与 2 的 8B 段全部落盘）

- [x] **缺环①**：`bench_spec_draft.py` 入库，copy 列 ±0.6% 复现 M9 融合表；顺带确认
      natural 列不可复现是 4-bit 贪心混沌，不是代码差异（底稿已写明）
- [x] **缺环④**：`bench_spec_temperature.py` + 底稿，把"无损"这件事的吞吐价签贴上；
      并纠正了一个此前的口头归因（税不在 float32 提案分布，在接受率）
- [x] **缺环⑤**：三种窗口策略 × 4 次重复对照，**回升分支证伪并回退**，
      `tests/test_spec_acceptance.py` 改断言单向棘轮；
      GPU e2e `tests/test_draft_spec.py::test_adaptive_gamma_shrinks_when_proposals_keep_failing`
      重跑通过
- [x] **缺环⑥ / ②**：常驻字节 + 8B 前缀缓存（见上文两条）
- [ ] **新认账的产品限制（非缺环）**：T>0 时免费提案器（ngram/lookahead）全线劣于同配置
      greedy，只有 draft 勉强守在 1.0× 附近 → 温度解码建议 γ≤2 或直接关投机。已写入
      `benchmarks/06-spec-draft/README.md`，README/docs 同步
- [x] **释放 v1 pack 死重量**（原"未实现的优化"）：后端新增 `uses_v1_pack`，
      Marlin repack 后丢 `qfp/scale`，`W4Linear` 只在需要时注册这两个 buffer，
      `W4Linear.to()` 的设备推导改成不依赖"一定有 buffer"。实测 8B 常驻
      6.674→3.339 GB、KV 池 1.52→3.16 GB（+108%，不是估的 +220%）；1.7B graph
      decode 同运行内 `w4.auto`≡`w4.marlin`（≤0.6%），省显存没付速度。PPL 未复测
      （`_B/_s` 逐字节不变）。全量测试 84/84（新增 3 条 CPU 常驻契约测试）。
      底稿同一文件新增复测 + 点测两节

## M10 续（2026-09-19，动态 γ 两个先验扫参）

- [x] **缺环⑤的剩余半项**：`bench_spec_adaptive_tune.py` 入库 + `results/m10_adaptive_tune.txt`
      （GPU 约 31 分钟，只用 1、3 号空卡）。三组轴扫完：**折半落点**是唯一决定性轴
      （级联/阶梯/一步到 1 一律 0.79–0.83×，机制=盈亏平衡 tok/step ≥2.11）；
      **WINDOW** 在 natural 上是平台、在 copy 上是悬崖（W=1/2 误触发掉 27%/24%，
      W≥3 各 10 次抽取从不触发）⇒ 它的身份是"一次不可逆动作的误触发滤波器"；
      **阈值**保持 1.0。**决定：`_adapt_gamma` 一个数字都不改**，扫参换来的是优化空间⑨。
- [x] **推翻并撤销 M10 为这条控制器写的因果解释**：裁窗口不省时间（提案器循环用配置 γ，
      `proposal_gamma` 从未流进它；步价只有 21.5/16.3ms 两档，实测）。n=5 重测
      钉上限 0.98× vs 棘轮 1.06×，原"0.91 救到 1.07"降级为弱信号 + 轨迹分叉（未证）。
      补上原设计缺的对照：手工 γ=2 → natural 打平（1.08×）但 copy 掉 13%。
- [x] 顺带量化混沌带（natural 投机重复 ±8%、平解码与 copy 投机可复现）⇒ 立了条口径：
      natural 上 <5% 的格间差不读成结论。`tests/test_spec_acceptance.py` 加一条 CPU 锁
      （窗口是误触发滤波器：单轮 1/4 不触发、连续三轮 ≤1.0 才触发）。全量测试 85/85。

## M11（2026-09-19，128K YaRN：先实现，再分档跑）

- [x] **实现 runtime 的 YaRN**：`rotary.py` 加 `_yarn_inv_freq`/`_yarn_attention_factor`
      （从 transformers 4.57.6 转写：`find_correction_dim`、truncate→floor/ceil、clamp、
      `linear_ramp_factor` 的 `high += 0.001` 退化处理、按 `1-ramp` 混合、
      `attention_factor = 0.1·ln(f)+1.0` 同乘 cos 与 sin），`qwen3.py` 停止丢弃 `rope_scaling`，
      `config.py` 新增 `rope_scaling` 字段并在 `__post_init__` 里把
      `hf_config.max_position_embeddings` 改写成 `original × factor` **再**过原有 clamp
      （天花板只有一处定义）。`lru_cache(1)` → 按 rope 设置建键的 dict（36 层共享）。
- [x] **正确性锁**：`tests/test_rotary_yarn.py` 12 条 —— 与 HF 的 `inv_freq` 差 ≤1e-9、
      cos/sin 缓存 `rtol=0, atol=0`（**逐位相同**，位置 0…131071）、attention factor 严格相等；
      另两条防静默：native 路径的缓存与实现前**逐位相同**且 `attention_scaling == 1.0`
      （这次重写不会改动其他里程碑的数字）、非 yarn 的 `rope_type` 抛 `NotImplementedError`
      而不是被忽略（一次没生效的长度外推会跑完、数字好看、什么都不能证明）。
      全量测试 **97/97**。
- [x] **分档 NIAH 落盘**：`benchmarks/08-long-context/bench_niah_yarn.py` +
      `results/m11_yarn_niah.txt`（8B W4A16+KV4，util 0.8，GPU3）。速度曲线是本轮最硬的
      产品数字（见上第 7 条）；召回读数只允许说"128K 更差 + 差在转写"，
      YaRN 的贡献**未证**（无 128K 基线：不缩放是越界外推而非基线）。
- [x] **两条 bench 口径教训**（已写进 `benchmarks/08-long-context/README.md`）：
      ① 同列 trials 的 haystack 必须**相位错开**，否则前缀缓存会把 prefill 从 6.7s 打到 1.0s
      —— 第一轮整张 prefill 表因此作废并重跑；② 答案片段必须宽到看得见数字（≥60 字符）+
      单独抽 `got=`，否则 24 字符套话让所有 miss 长得一样、无法归因。
- [ ] 新账 10（评分口径）与 11（池计费）见上。
