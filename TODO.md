# TODO — 证据链缺环与收尾遗留（2026-09-18 整理；同日 M10 复核后更新；09-19 补测后再次更新）

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
   WINDOW=3 与折半系数**没扫过**，且"窗口能否越过配置值"结构上仍无从验证（verify 图族按
   固定 M=γ+1 捕获）。这两点降级为优化空间，不再算证据链缺环。
6. ~~**Marlin 路径双份 int4 常驻只有一句口算**（M10 修门面测试时暴露）：
   `w4.auto` 名义上按 M dispatch，实际 `CROSSOVER_M=0`，v1 pack 全程闲置，
   而 `w4.marlin` 另存一份 Marlin 排布的 `_B/_s`。口算给的"8B ≈ 1.9 GB"从未
   进测。~~ — **09-19 已实测**：`benchmarks/02-w4-quant/bench_w4_residency.py` +
   `results/m10_w4_residency.txt`，1.7B 死重量 0.677 GB（+15% KV 池）、8B
   **3.335 GB**（1.52→4.85 GB，可换出约 2.1 万 → 6.6 万 token 总容量）。口算
   低了近一倍，已更正。**注意**：`w4.auto` 与 `w4.marlin` 逐列相同——省不掉
   这一份是 Marlin 后端本身的性质，与 dispatch 策略无关。释放 `qfp/scale`
   的改动**尚未实现**，底稿只证明空间存在。

> **缺环 1–6 于 2026-09-19 全部闭合**（底稿：`results/m10_w4_residency.txt`、
> `m10_draft_sweep.txt`、`m10_spec_temperature.txt`，以及追加 8B 段的
> `results/m9_prefix_cache.txt`）。下列 7–8 为功能性收尾，不是证据链问题。

## 收尾遗留（功能，非证据链）

7. **128K YaRN demo 未跑**（M4 起遗留）。
8. ~~models/Qwen3-0.6B-qslab-w4-awq2~~ 已删（A 档清理 11795fd）；tests/ 4 个辅助脚本已移 scripts/archive/m-verification/。

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
- [ ] **未实现的优化**：`w4.auto`/`w4.marlin` 的 v1 pack 死重量（8B 3.335 GB）可在装换后
      释放 `qfp/scale`，底稿只证明了空间存在，改动未做
