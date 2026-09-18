# 06 — draft 投机（新 runtime，M9）

## ⚠ 证据链状态：结果底稿完整，入库脚本缺环

底稿 results/m9_spec_draft.txt、results/m9_gamma_sweep_draft.txt、results/m9_draft_fused.txt
**没有对应脚本入库**——当时的 bench 是以临时脚本/内联方式跑的（draft γ 扫描 + 融合对照），
未像 bench_spec_ngram.py 那样固化。已记入根目录 TODO（见 benchmarks/README.md 同级 TODO.md）。
按 bench_spec_ngram.py 的结构补一个 bench_spec_draft.py 即可闭环。

## 证据链（底稿数字）

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

vs n-gram（copy 3.19× / natural 0.95×）：**n-gram 统治复读，draft 统治自然文本**。
踩坑：dynamo 守卫编码 inference_mode → 首提案 5s 重编译 → DraftProposer._warm() init 预热。

## 复现（待脚本固化后可用）

```bash
# 现状：通过引擎直接跑（spec_method=draft），γ 用 config.spec_num_drafts
python - << 'PY'
from qslab.runtime.llm_engine import LLMEngine
from qslab.runtime.sampling_params import SamplingParams
eng = LLMEngine(model="models/Qwen3-8B", w4="models/Qwen3-8B-qslab-w4-awq",
                spec_method="draft", draft_model="models/Qwen3-0.6B",
                spec_num_drafts=2, max_model_len=4096, max_num_seqs=8)
eng.add_request("The capital of France is", SamplingParams(temperature=1e-6, max_tokens=192, ignore_eos=True))
# ... 计时循环同 bench_spec_ngram.py
PY
```
