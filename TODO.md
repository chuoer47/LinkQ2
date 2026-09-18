# TODO — 证据链缺环与收尾遗留（2026-09-18 整理）

## 缺环（证据链不完整处）

1. **draft 投机 bench 脚本未入库**（06-spec-draft/）：
   results/m9_spec_draft.txt、m9_gamma_sweep_draft.txt、m9_draft_fused.txt 无对应脚本。
   建议：按 benchmarks/05-spec-ngram/bench_spec_ngram.py 结构补 bench_spec_draft.py
   （spec_method="draft" + GAMMA 扫描 + 融合开关对照）。
2. **m9_prefix_cache.txt 只有 1.7B 数字**，8B 的 558.3→63.0ms 只在 notes/PROGRESS 里
   （当时跑过）。建议：重跑 `MODEL=models/Qwen3-8B` 一次落盘，或把当时输出补录。
3. **m2_niah.json 覆盖 @1K/@2K**，32K NIAH（M4 双 100%）无独立 json 底稿（在
   bench_engine_matrix 的会话输出里）。建议：M7_acceptance 式的独立验收脚本可选。

## 收尾遗留（功能，非证据链）

4. L4 门面（api/llm.py）仍指旧引擎 qslab/engine；新 runtime 无 LLM/CLI 包装。
5. 温度采样投机（Leviathan 概率比接受律）未做——温度>1e-3 序列投机时静默退化普通 decode。
6. 动态 γ（按接受率自适应）未做；最优 γ 强依赖负载（自然 2 / 复读 8）。
7. 旧投机栈 qslab/engine/spec/（lookahead/dynamic）冻结未迁移。
8. 8B+draft e2e 测试未固化（有 bench 数据无测试）。
9. 128K YaRN demo 未跑（M4 起遗留）。
10. ~~models/Qwen3-0.6B-qslab-w4-awq2~~ 已删（A 档清理 11795fd）；tests/ 4 个辅助脚本已移 scripts/archive/m-verification/。
