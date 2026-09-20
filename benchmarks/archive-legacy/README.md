# archive-legacy — 旧引擎 spec bench（M3/M6）

对应**旧引擎** `qslab/engine/spec/`（SpeculationMode：chained/lookahead/dynamic），
已被新 runtime（ngram + DraftProposer）取代。保留原因：M3/M6 笔记的数字出自这里，
且 lookahead 模式在新栈尚未迁移。

## 证据链

- results/bench_spec.json（M3）：AR 4.96@γ4 但 e2e **0.74×**——Python 调度开销归因
- results/m6_spec_modes.json（M6）：lookahead **1.44×** 转正 / chained 0.72× / dynamic 0.76×
- results/m5_spec_retest.json（M5）：W4 target 下 spec 0.53×→0.72×（autodetect 修正）

## 复现

两个脚本已于 2026-09-20 随旧引擎从 main 摘除，完整保留在存档分支
`archive/legacy-engine`（= 标签 `legacy-engine-final` = `8af4b1e`）：

```bash
git checkout archive/legacy-engine
python benchmarks/archive-legacy/bench_spec.py
python benchmarks/archive-legacy/bench_spec_modes.py
```

新 runtime 的投机基准在 `benchmarks/05-spec-ngram/` 与 `benchmarks/06-spec-draft/`。
