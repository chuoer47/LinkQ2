# TODO — 证据链缺环与收尾遗留（2026-09-18 整理；同日 M10 复核后更新）

## 缺环（证据链不完整处）

1. **draft 投机 bench 脚本未入库**（06-spec-draft/）：
   results/m9_spec_draft.txt、m9_gamma_sweep_draft.txt、m9_draft_fused.txt 无对应脚本。
   建议：按 benchmarks/05-spec-ngram/bench_spec_ngram.py 结构补 bench_spec_draft.py
   （spec_method="draft" + GAMMA 扫描 + 融合开关对照）。
2. **m9_prefix_cache.txt 只有 1.7B 数字**，8B 的 558.3→63.0ms 只在 notes/PROGRESS 里
   （当时跑过）。建议：重跑 `MODEL=models/Qwen3-8B` 一次落盘，或把当时输出补录。
3. ~~m2_niah.json 只覆盖 @1K/@2K，32K 无独立底稿~~ — **M10 复核后不成立**：
   `results/m2_niah.json` 本身就是 ctx=32768 的落盘（fp16 与 kv4 各 3 深度 ×3 次，
   overall_recall 均 1.0），产出脚本 benchmarks/01-decode-baseline/bench_niah.py 在库。
   真正偏弱的只是探针密度（9 次/配置），不算缺环。

### M10 带来的新缺环

4. **温度投机只有正确性证据，没有吞吐证据**：接受律已换成 Leviathan 概率比
   （tests/test_spec_acceptance.py 实测分布无偏：N=20000、T=0.9 时 TV<0.06），
   但概率比路径每步要多一份 `[bs·(γ+1), V]` float32 提案分布和一次 softmax，
   **加速比从未测过**。建议：06-spec-draft 下补 T>0 的 n-gram/draft 吞吐对照。
5. **动态 γ 的阈值未在新 runtime 上调过参**：WINDOW=3、"平均接受 ≤1.0 折半 /
   ≥γ-0.1 回升"是旧引擎 M6 直移的先验；γ 上限被 `spec_gamma` 锁死（verify 图族
   M=γ+1 固定），所以"高接受率时窗口能不能再放大"这件事当前无从验证。

## 收尾遗留（功能，非证据链）

6. **128K YaRN demo 未跑**（M4 起遗留）。
7. ~~models/Qwen3-0.6B-qslab-w4-awq2~~ 已删（A 档清理 11795fd）；tests/ 4 个辅助脚本已移 scripts/archive/m-verification/。

## M10 已完成（2026-09-18，功能遗留 4-8 全部落地）

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
