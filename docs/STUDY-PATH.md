# STUDY-PATH — 项目学习与整理路径

> 按一个 token 的旅程组织：从进入引擎到吐出来，每停一站下钻一层。
> 每层四步法：① 读 ARCHITECTURE.md 对应段带着问题进代码 → ② 画调用链 →
> ③ 跑对应测试看断言（测试 = 模块契约）→ ④ 用自己的话写 ≤10 行小结。
> ⚠ = 该站的过时物清单（开发过程遗留，见文末处置建议）。

## 第 0 步：背地图（半天）

只读 ARCHITECTURE.md §2（速览 + 7 条结论）+ §3（分层图 + 投机 step 数据流）。
验收：能默写数据流、能回答"8B 97.2 tok/s 怎么来的"、能说清两代引擎谁在哪个目录。

## 第 1 步：L3 runtime — 主线下钻（1-2 天，最重）

按调用顺序，每个文件回答一个问题：

| # | 文件 | 问题 | 过时物标注 |
|---|---|---|---|
| 1 | `runtime/llm_engine.py` | step() 三分支怎么转？ | 全部现役 |
| 2 | `runtime/scheduler.py` | 序列生命周期；γ 从哪来 | 全部现役 |
| 3 | `runtime/model_runner.py` | prepare_* 造张量；(bs,M) graph 捕获 | 全部现役 |
| 4 | `runtime/block_manager.py` | 分块/哈希命中/预留与 trim | 全部现役 |
| 5 | `runtime/sequence.py` `context.py` | 数据结构（短，扫一眼） | 全部现役 |

测试：`test_spec_runtime.py`(7)、`test_prefix_cache.py`(4)、`test_ngram.py`(7)、`test_draft_spec.py`(4)。
⚠ 本站无过时物——runtime 是 M8 起的纯主线。

## 第 2 步：L2/L1 — 量化数据路径（1 天）

| 文件 | 问题 | 状态 |
|---|---|---|
| `quant/packfmt.py` | uint32 装 8 个 int4、scale 排布 | ✅ 现役（新旧两代共用） |
| `quant/w4_backends.py` | 三后端分派；为什么 CROSSOVER_M=0 | ✅ 现役（runtime loader 经 W4Linear 用它） |
| `models/w4linear.py` | Linear 调用怎么转给 backend | ✅ 现役（新 runtime/loader.py 直接 import） |
| `runtime/attention.py` | prefill 物化拼接 / decode Triton paged | ✅ 现役 |
| `runtime/paged_decode.py` | tile 内反量化；M 通用化与因果性 | ✅ 现役 |

⚠ 本站过时物（旧引擎的量化路径，新 runtime 不经此）：
- `quant/cache/kv_cache.py`（dense fp16/kv8/kv4）——仅 test_gpu_kernels 与旧引擎用
- `quant/kv_strategies.py` / `kv_strategies_impl.py`——旧引擎策略层（kv4.paged 注册在此）
- `models/patched.py` / `models/loader.py`——旧引擎 L2（patched.py 的 oracle 对拍思路仍有教学价值）

## 第 3 步：L0 kernel — 两个灵魂文件（1 天）

| 文件 | 问题 | 状态 |
|---|---|---|
| `kernels/csrc/w4a16_gemm.cu` | 寄存器解包；coalescing 为何是一切 | ✅ 现役 |
| `kernels/marlin_backend.py` | repack 管线（置换表逐行照抄上游） | ✅ 现役 |
| `kernels/ops.py` | 绑定与流语义（M8 默认流 bug 的现场） | ✅ 现役 |

⚠ 本站过时物：
- `kernels/kv4_paged_attention.py`——**M7 交付物，但只服务旧引擎**；新 runtime 的
  paged kernel 是 `runtime/paged_decode.py`（另一个、更新的 Triton kernel）。
  两者别混：M7 kernel 是"旧引擎的 paged"，现役的是 paged_decode。

测试：`test_verify_kernel.py`(4，对拍+因果性，值得精读)、`test_gpu_kernels.py`(4，测的是旧 dense KV)。

## 第 4 步：离线量化链（半天，独立于引擎）

`quantizer/` → AWQ 校准 → 打包 → SmoothAttention（`scripts/build_smooth_kv.py`）。
动手：对 0.6B 走一遍 quantize → load → 推理。
⚠ 过时物：无——这条链是"简历最硬证据"（自有打包格式），新旧引擎共用。

## 第 5 步：横切专题收口（1 天）

- **投机三代对照读**：notes/M3（旧引擎，负加速）→ M6（三模式，lookahead 转正）→
  M9（新 runtime，n-gram + draft 各有主场）。同一条问题看方案怎么演进。
- **前缀缓存一个 bug 的完整生命周期**：notes/M8 §七（发现+禁用）→ M9 §九（复现+修复+复活）。
- ARCHITECTURE.md §4 的全部 ⚠ 弯路逐条过，每条问"怎么发现的、怎么证的"。

## 第 6 步（可选）：旧引擎考古（半天）

`qslab/engine/`（6 文件）是 M0-M7 的完整引擎：单请求 decode 循环 + transformers 前向。
读它的价值：oracle 对拍方法论、M3 投机三 bug 现场、M6 lookahead 实现（新栈未迁移）。
⚠ `engine/spec/graph_decoder.py` 是未完成的 CUDA Graph 骨架——注释里有完整方案分析
（M6 中止决策），是"为什么换 runtime"的最好注脚。

---

## 过时物总清单与处置建议（三档）

### A 档：可删（无引用、实验证伪产物）
| 项 | 依据 |
|---|---|
| `models/Qwen3-0.6B-qslab-w4-awq2`（227M） | W4 draft 证伪实验的产物（TODO #10）；证伪结论在 results/m9_gamma_sweep_draft.txt，模型本身无保留价值 |
| `tests/` 下 4 个非 pytest 辅助脚本（spec_lossless / oracle_alignment / kernel_path_oracle / kv_ppl_compare） | M1/M3/M8 时代的一次性验证脚本，pytest 不执行，全走旧引擎；移 `scripts/archive/` 即可 |

### B 档：冻结保留（叙述/教学价值 > 删除收益）
| 项 | 为什么留 |
|---|---|
| `qslab/engine/`（52K，6 文件） | M0-M7 全部笔记数字的出处；oracle 对拍方法论载体；M6 lookahead 待迁移参考 |
| `qslab/models/patched.py` + `loader.py` | 同上（旧引擎 L2）；patched.py 是"替换子类绕过重分配权重"的教学样本 |
| `quant/cache/kv_cache.py`（dense 三实现） | test_gpu_kernels 锁定的精度分级仍有效；M2 笔记数字的出处 |
| `kernels/kv4_paged_attention.py` + `quant/cache/kv4_paged.py`（M7 套件） | 简历叙事物（Triton 量化 paged attention）；test_kv4_paged 锁定；**读代码时注意区分它与现役 paged_decode.py** |
| `qslab/sampler.py`（根级） | M3 拒绝采样数学，test_unit_cpu 锁定分布无损性 |
| `benchmarks/04-runtime-m8/`（engine_matrix、m7_acceptance 等） | M4/M7 结果的证据链脚本，README 已归位 |

### C 档：标注即可（防误读）
| 项 | 标注 |
|---|---|
| `qslab/api/llm.py` 指旧引擎 | 已在 ARCHITECTURE §3 + README 注明；L4 接新 runtime 是 TODO #4 |
| `third_party/`（marlin，1.8M，gitignore） | vendored 依赖，repack 表对照上游用 |
| M7 kernel vs paged_decode 重名问题 | STUDY-PATH 第 3 步已标注 |
