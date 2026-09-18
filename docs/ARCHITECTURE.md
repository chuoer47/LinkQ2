# qserve-lab 架构与项目主线（唯一入口）

> 生成于 2026-09-18。本文是项目的**唯一权威入口**：架构分层、M0→M9 演进主线、
> 当前最终状态的数字与结论。所有数字取自 `results/` 原始文件与 `notes/` 里程碑笔记，
> 与代码现状交叉核对过。旧设计文档已移入 `docs/archive/`（结论以本文为准）。

---

## 1. 项目是什么

**自研轻量 LLM 量化推理引擎**（简历项目，非学术）。目标：搞清"W4 量化推理快在哪、
准在哪、边界在哪"，全部自己动手 + 每步实测对标。

- **平台**：单张 RTX 4090 D（sm_89, 24GB），服务器 `4090_public` GPU3
- **模型**：Qwen3-8B（主）/ 1.7B（迭代）/ 0.6B（draft），Apache 2.0
- **技术栈**（每件都有行业参照物，不自己发明）：

| 部件 | 方案 | 参照 |
|---|---|---|
| 权重量化 | W4A16 group-wise + AWQ（自实现 scaling+clip） | Marlin / AWQ |
| KV cache | KV4 不对称量化（K 静态 per-channel + SmoothAttention / V 动态 per-token g64） | KIVI / QServe / vLLM FP8 |
| attention | Triton 量化 paged kernel + flash-attn prefill + 手写 CUDA GEMV | vLLM / Marlin |
| runtime | paged KV 池 + 连续批 + CUDA Graph + 前缀缓存 | nano-vllm → vLLM V1 |
| 投机推理 | n-gram（lookup）+ chained draft model，greedy 无损验证 | vLLM V1 / TGI |

- **方法论**（贯穿全项目）：手写 → 对比官方 → 记笔记；每个里程碑有 notes 文件，
  **证伪的实验和成功的同等保留**；正确性测试必须有"零容差对拍物"。

---

## 2. 当前最终状态（速览表）

全部为 RTX 4090 实测，greedy。基线均指本引擎（非 vLLM）。

### 2.1 吞吐（batch=1，128 tok 生成）

| 配置 | tok/s | vs FP16 引擎基线 | 出处 |
|---|---|---|---|
| 8B FP16（旧引擎 M4 口径） | 42.4 | 1.00× | results/m4_matrix_fp16.json |
| 8B W4A16KV4 decode（新 runtime + Marlin） | **97.2** | **1.75×**（vs 旧 W4 14.2 → 6.8×） | notes/M9 §三 |
| 8B W4A16KV4 + n-gram γ=4 复读 | **310.4** | 3.19× | results/m9_spec_8b.txt |
| 8B W4A16KV4 + draft γ=8 复读（融合后） | 225.1 | 2.30× | results/m9_gamma_sweep_draft.txt |
| 8B + draft γ=2 自然文本（融合后） | 114.4 | 1.17× | results/m9_draft_fused.txt |
| 1.7B 新 runtime FP16 decode | 148.7 | 3.5×（vs M0 42.1） | notes/M8 §五 |

### 2.2 质量

| 指标 | 数字 | 口径 |
|---|---|---|
| 8B W4 PPL | 16.07 → 17.31（+1.24） | WikiText-2, AWQ 配方 |
| 8B KV4 PPL | 16.06 → 16.43（+0.369, +2.3%） | bench_ppl_kv.py（HF 模拟法） |
| 权重压缩 | 16G → 2.49G（**6.6×**） | qslab_w4_v1 打包 |
| KV 显存 | fp16 的 **2×** 节省（池口径）/ 3.5×（序列口径） | notes/M8 §四 |
| 32K NIAH | fp16 与 KV4 双 100% | results/m2_niah.json + M4 |
| 前缀缓存 TTFT（3800-token 前缀，8B） | 558.3 → 63.0ms（**8.86×**） | results/m9_prefix_cache.txt |
| CUDA Graph | batch1 6.5×、8 路连续批 1048 tok/s | notes/M8 §五 |

### 2.3 关键结论（一句话版）

1. **W4A16 的收益在 decode 是带宽账**：Marlin M=1 达 911 GB/s（带宽极限 1008）。
2. **自研 GEMV 输给 Marlin**，e2e 上 crossover 不是 M=8 而是 0——微基准不能外推。
3. **KV4 的 PPL 代价 +0.37 可接受**；"逐 token 无损"在 4-bit 是混沌不可达，PPL 才是仪器。
4. **静态 K scale（SmoothAttention）是 CUDA Graph 可捕获的前提**——写入与数据无关。
5. **投机推理两 proposer 各有主场**：n-gram 统治复读（3.19×，提案免费）、
   draft（torch.compile 融合后）统治自然文本（1.17× vs 0.95×）。
6. **bs=1 时 kernel 数是税不是字节**：W4 draft 步时不改（2241 kernel/步），
   融合（torch.compile）才是正解，4.53→2.59ms。
7. **批处理下单条输出漂移不是缺陷**：cuBLAS 换 tiling，HF 同样漂移——对比必须同批口径。

---

## 3. 架构（当前真实代码）

```
L4  qslab/api/          LLM 门面 + CLI —— ⚠ 仍指向旧引擎 qslab/engine
L3  qslab/runtime/      新 runtime（M8 起，主引擎）:
                          llm_engine.py   LLMEngine（add_request/step 连续批循环）
                          scheduler.py    调度（prefill/decode/verify 分支 + 投机提交）
                          model_runner.py prepare_prefill/decode/verify + CUDA Graph 捕获
                          draft.py        DraftProposer（chained：第二个完整 runtime）
                          ngram.py        NGramProposer（CPU 查表）
                          block_manager.py 分页块管理 + 前缀哈希缓存
                          attention.py    prefill(flash-attn+前缀物化) / decode(Triton paged)
                          paged_decode.py int4 池 Triton kernel（decode/verify/materialize）
                          qwen3.py primitives.py rotary.py sampler.py ...
L2  qslab/models/       Qwen3 backbone（旧引擎用，patched.py 替换子类）
                          w4linear.py     W4Linear → QuantBackend.linear()
L1  qslab/quant/        packfmt.py  qslab_w4_v1 打包格式
                          w4_backends.py  w4.v1 / w4.marlin / w4.auto（CROSSOVER_M=0）
                          cache/kv4_paged.py  M7 时代的 paged KV（旧引擎路径）
                          calibrate.py quantize.py kv4_plan.py  离线量化链
L0  qslab/kernels/      csrc/w4a16_gemm.cu（自研 GEMV）+ marlin vendor + ops 绑定
adapters/               唯一可 import transformers 的地方
```

**两代引擎并存**（当前最重要的结构事实）：

| | 旧引擎 `qslab/engine`（M0-M7） | 新 runtime `qslab/runtime`（M8-M9，主线） |
|---|---|---|
| 形态 | 单请求 decode 循环，transformers 前向 | nano-vllm 骨架：paged KV + 连续批 + CUDA Graph |
| KV | KVCacheStrategy（dense fp16/kv8/kv4 + kv4.paged） | int4 池（K 静态 channel / V 动态 group） |
| 投机 | SpeculationMode：chained/lookahead/dynamic | ngram + draft（proposer 接口统一） |
| 入口 | `qslab/api/llm.py`（LLM 门面/CLI 挂这里） | `qslab/runtime.llm_engine.LLMEngine` 直接用 |
| 状态 | **冻结**，oracle 对拍参照 + spec 三模式代码 | 主线，全部最终数字出自这里 |

规则不变：每层只 import 直接下层；量化决策在 L1（backend），模型不感知 kernel。
遗留：L4 门面未接到新 runtime（bench 与测试直接用 `runtime.LLMEngine`）。

**一次投机 decode step 的数据流**（新 runtime）：

```
LLMEngine.step()
  ├─ Scheduler.schedule() → prefill / decode / verify 三分支
  ├─ _propose(): ngram.propose_batch（CPU 查表）或 draft.propose_batch（GPU 紧循环）
  ├─ ModelRunner.run_verify(): M=γ+1 行一次前向（(bs,M) CUDA Graph 族）
  │    └─ attention decode 分支: Triton kv4_paged_decode_kernel(M)
  │         读取上界 = L+m —— 因果性=槽位下界，draft KV 天然被排除
  └─ Scheduler.postprocess_verify(): greedy 接受（最长 argmax 前缀 + bonus）→ 提交 → trim
```

---

## 4. 里程碑主线（动机 → 方案 → 结果 → 结论）

每节末尾是"如果只记一句话"。证伪/弯路用 ⚠ 标注。

### M0 — FP16 基线（1.7B，42.08 tok/s）
- 动机：先有能跑的 decode-only 引擎与 oracle 对齐链路。
- 结果：token 级与 HF 一致；PPL 26.48；单步 23.5ms。
- 坑：decode 显式传 `cache_position`（否则位置 0 分叉）；transformers 4.57 属性在 config 上。
- **一句话：oracle 对齐链路是所有后续里程碑的安全网。**

### M1 — W4A16：量化器 + 自研 GEMV kernel
- 方案：RTN→AWQ（scaling + per-group clip MSE 搜索）；w4a16_gemm.cu（nibble 寄存器解包 + FP16 dot）。
- 结果：AWQ PPL +1.24（16 docs 校准口径）；kernel 主 decode 形状 1.45~1.77×；8B 打包 4.6×。
- ⚠ 弯路：kernel 三轮"优化"（warp 分 K / LUT）全无效或负收益——**coalescing 敏感度 > 一切**；
  nibble 解码 bug（`q&0xF` 编码 vs `(q-8)` 解码）被 0.11 量级误差**掩盖四轮**，直到引入
  反量化 matmul 零容差对拍才暴露。
- ⚠ 测量教训：4090 的 72MB L2 会完整缓存 ≤25MB 权重，不冲刷 L2 的 GEMV 虚高 2~4×。
- **一句话：正确性测试必须有零容差对拍物；微基准必须冲 L2。**

### M2 — KV4（KIVI 式不对称）
- 方案：K per-channel（转置存）/ V per-token，g=64，layer0 保 FP16（离群 486×）。
- 结果：PPL +0.56（边缘）；显存 3.34×（1.7B）；NIAH @1K 双 100%。
- 关键产出：chunked prefill（两段式 SDPA，零 mask 张量）解锁 32K。
- 坑：多态 reset 必须走虚方法；部分组 staging 读回。
- **一句话：KV4 可行，离群保护（首层 FP16 + per-channel K）是命门。**

### M3 — draft 投机（旧引擎，负加速的教训）
- 方案：0.6B draft + 拒绝采样，greedy 无损验证。
- 结果：AR 4.96@γ4 但 e2e **0.74×**。归因：Python 调度开销主导（γ=4 时每 5 token 7 次 Python forward）。
- 三个经典 bug：验证对齐（g_i 由上一轮 target 预测背书）、bonus 选择、cache 失步。
- **一句话：小模型 + Python 引擎里投机天然吃亏；target 越大 dispatch 越被摊薄（M4 验证）。**

### M4 — 8B 组合拳
- 结果：四方矩阵（FP16 42.4 / W4 14.2 / +KV4 9.0 / +spec 10.8 tok/s）；32K NIAH 双 100%；
  KV4 显存 3.5× 达标；spec 8B 转正 1.19×。
- ⚠ 诚实结论：W4 在 8B 反而慢于 FP16——自研 kernel 在大形状只有 1.14×，cuBLAS 太强。
  这直接引出 M5。
- **一句话：8B 上"权重 4 倍、GEMV 占比 3 倍"的预言兑现了，kernel 短板也暴露了。**

### M5 — Marlin 接入 + hybrid dispatch
- 方案：CUTLASS mixed-input 在 sm89 无 collective（硬约束放弃）→ vendor Marlin（vLLM 同款）；
  repack 管线（置换表逐行照抄上游）。
- 结果：8B e2e 14.2 → **39.78** tok/s（fp16 的 94%）。三方对比：M=1 小形状 v1 仍赢（Marlin ~100µs 固定开销），M≥8 Marlin 全面碾压。
- 当时结论：crossover=8，autodetect M>1→M>8。
- **⚠ 该 crossover 后来被 M9 e2e 推翻（见 M9 §三）——微基准教训的全额学费。**
- **一句话：Marlin 是 Ada 工业标准；"持平 fp16"目标达成。**

### M6 — 投机三模式 + lookahead 转正
- 结果：lookahead（n-gram 自投机，提案零成本）**1.44×**；chained 0.72×；dynamic 0.76×。
- 架构回报：三模式共享验证/回滚核心，无损 100%——验证框架独立于提案来源。
- ⚠ CUDA Graph 中止决策：旧引擎 KV 写入地址依赖数据，正确方案（ring-buffer）工程量 1-2 周，
  让位给 lookahead。**这个障碍 M8 换 runtime 后自然消失。**
- **一句话：提案成本决定投机生死；graph 的问题留给数据结构去解决。**

### M7 — Triton 量化 paged attention（旧引擎内）
- 结果：paged vs dense PPL 差 -0.20（噪声边缘）；@8K 峰值显存 24.4→18.4 GB（O(context) fp16 副本消失）。
- 四个 bug：组内 scale 污染（写入时重算整组）、uint32 下溢、fp16 scale 下溢、转置。
- **一句话：paged 化解决长上下文显存峰值；块对齐分组（128 块恰含 2×64 组）让分组永不跨块。**

### R0-R6 — 工程化重构（插在 M7 与 M8 之间）
- 五层骨架、三策略接口（registry + factory）、统一入口 LLM/CLI、仓库清理、pytest 19 测试锁定。
- ⚠ 抓出重构自伤：sed 误伤 `_wrap_input_scale` 成静默 no-op，AWQ fold-back 全失效输出乱码
  ——由统一入口的 e2e 测试抓住。
- **一句话：重构的价值在接口（三策略插槽 M7/M8 直接兑现）和测试网（两次抓住静默回归）。**

### M8 — nano-vllm runtime 整合（分水岭）
- 动机：旧引擎三座大山——CUDA Graph 不可捕获、无连续批、KV 全量反量化 O(context) 副本。
- 方案：vendor nano-vllm 骨架（~1.3k 行，MIT），TP 剥离；KV 池换 int4；**K 改静态 per-channel
  scale（SmoothAttention 离线校准 λ_i=max|K_i|^0.5，RoPE 配对约束 λ_i=λ_{i+D/2}），
  V 保持动态 per-token**——写入纯 slot 驱动，graph 可捕获。
- 依据：vLLM FP8（离线校准）/ TRT-LLM NVFP4（offline）/ KIVI（残差窗口）/ QServe（平滑）——
  **没有一家生产系统回头重量化旧 token**。
- 结果：1.7B e2e 与 HF 逐 token 一致；CUDA Graph batch1 **6.5×**（148.7 tok/s）；
  8 路连续批 1048 tok/s；W4 接入（拆融合投影——AWQ per-shard scale 装不进融合 qkv，融合只值 3%）。
- 坑（高价值）：store/decode nibble 编码不对称（offset-binary vs two's-complement，差 8 个量化级）；
  V 非连续视图（qkv.split）store 读垃圾——prefill 无恙 decode 崩；
  lm_head 维度声明反了（tie-embedding 的 1.7B 侥幸掩盖，8B 即崩）；
  **w4a16_gemm 启动在默认流 → 不被录进 CUDA Graph → 输出全 0**（任何自定义 kernel 必须显式用当前流）。
- ⚠ 两次自我更正：per-token K 量化被实测推翻（离群通道 111×，误差 23%→per-channel 3.6%）；
  "逐 token 无损"降格为"混沌不可达，PPL 才是仪器"（4-bit greedy 近并列必翻）。
- ⚠ 遗留 bug（M9 修复）：前缀缓存静默错答——命中前缀只在 int4 池，prefill 不读池。
- **一句话：runtime 换骨后，M6 的 graph 障碍、M7 的副本问题一次全消；静态 K scale 是 graph 的钥匙。**

### M9 — 投机推理接入新 runtime（n-gram → draft → 融合 → 前缀缓存）
四波推进，全部有 results 文件：

**① n-gram + verify kernel 通用化**（4392dae）
- decode kernel 加 `M: constexpr`：verify = M=γ+1 行一次前向；因果性=槽位下界（draft KV
  在更高槽天然被排除）；append-only 回收（TGI 路线）；padding 中立（dummy 永不被接受，
  该步严格等价普通 decode）。
- 结果：8B 复读 **310.4 tok/s（3.19×）**，输出 128/128 一致；γ=8 达 5.77×（1.7B）。
- **⚠ Marlin crossover 修正**：M5 的 crossover=8 被 e2e 推翻——v1 GEMV 逐行重读权重，
  M=5 verify 65ms vs Marlin 11.5ms；M=1 decode 8B 54.1 vs 91.1。**M8 遗留"55.8 未查因"
  的答案就是它**。CROSSOVER_M=0，8B 主线 decode **97.2（1.75×）**。
  教训：单算子固定开销模型不能外推到图内 252 层前向。

**② draft model 迁移**（ad95b3e → baf9ab6）
- DraftProposer = 第二个完整 runtime（自有 runner+scheduler+int4 池+graph），
  vLLM V1 draft worker 形态；锁步协议：sync 截断（draft KV 前缀因果有效）+ prefill + γ 步紧循环。
- 接受率兑现：自然文本 2.08-2.48 tok/步（n-gram 仅 1.05）。
- ⚠ 瓶颈归因三连（全部实测，不猜）：CPU 不是（CUDA event 证明 0.25ms 完全重叠，
  修正"一半是 Python"的初判）→ W4 不是（GEMV 2.50→1.19ms 但步时不变）→
  **kernel 数是**（profiler：2241 kernel/步，~1500 个未融合 elementwise 占 2.6ms）。

**③ torch.compile 融合 + γ 帕累托**（c95c3a0 → e395b60）
- draft 编译（attention dynamo-disable 处 graph-break，层间 elementwise 交 inductor）
  → 步时 4.53→**2.59ms（1.75×）**，编译产物照常录进手写 CUDA Graph。
- ⚠ dynamo 守卫陷阱：inference_mode 被编码进守卫，init 预热与 propose 循环不同
  → 首提案 ~5s 重编译，γ=1 扫描点被污染成 0.30×（一度误判异常路径）。修复：init 时 dummy propose 预热。
- γ 定版：自然 **γ=2（1.17×）**、复读 **γ=8（2.30×）**——最优 γ 强依赖负载，静态配置必偏科。
- **分工格局：n-gram 统治复读（3.19×），draft 统治自然文本（1.17× vs 0.95×）——各有主场。**

**④ 前缀缓存修复复活**（9d7b91a → 6790029）
- M8 遗留的静默错答 bug（当场复现：同一 prompt 二次提交输出多出错误前导 0），
  物化路线修复：`materialize_kv`（池槽位→fp16 反量化）+ prepare_prefill 一次 slot 计划
  下发全层 + attention prefill 物化拼接；`ENABLE_PREFIX_CACHE=True` 默认开。
- ⚠ 第一版每层 tolist 重算 = 28 次 GPU 同步 ~14ms，比省的还贵——跨层计划绝不进层内。
- **顺手修掉同族 bug**：chunked prefill 第二块同款错位（prompt>16384 才触发，从未暴露）。
- 结果：8B 3800-token 前缀 TTFT 558.3→**63.0ms（8.86×）**；1.7B 2.73×。
- 投机序列补 hash_blocks（投机提交也贡献缓存条目；哈希边界不含 bonus 位，拒绝槽垃圾不入缓存）。
- **一句话：一次物化修复两个 bug；"先复现再修"的铁律再次兑现。**

### 测试现状（locks）

非 e2e 32 + e2e 27（含 spec 11、prefix 4、draft 4）+ 8B 2，全绿。
断言口径：**不逐 token 断言量化输出**（混沌），断言数学结构（接受机制/步数/确定性/
首位 token/前缀稳定性）。pytest: `pytest tests/ -m "not e2e"` 快门 / 全量 ~2min。

---

## 5. 已知限制与下一步

1. **L4 门面未接新 runtime**：`api/llm.py` 仍指旧引擎；新引擎无 LLM/CLI 包装。
2. **温度采样投机未做**：仅 greedy 接受律；温度>1e-3 的序列投机时静默退化普通 decode
   （padding 中立保证正确，但不加速）。Leviathan 概率比接受律是正解。
3. **动态 γ 未做**：最优 γ 强依赖负载（自然 2 / 复读 8），按接受率自适应是明确方向
   （旧引擎 M6 dynamic 有先例）。
4. **旧投机栈冻结**：`qslab/engine/spec/`（lookahead/dynamic 模式）未迁移未删除。
5. **8B+draft 的 e2e 测试未固化**（有 bench 数据无测试）。
6. **128K YaRN demo 未跑**（M4 起遗留）。

证据链与复现：见 `benchmarks/`（脚本）与 `results/`（原始文件）；
分类导读见 `benchmarks/README.md`（若此文件存在）。

---

## 6. 文档地图

| 文档 | 状态 | 用途 |
|---|---|---|
| **本文 ARCHITECTURE.md** | ✅ 权威 | 唯一入口：架构 + 主线 + 最终状态 |
| PROGRESS.md | ✅ 维护中 | 开发日志（按时间，最细粒度） |
| notes/M0..M9 | ✅ 保留 | 每里程碑的原始实验记录（数字出处） |
| docs/00-主线决策记录 | ✅ 保留 | 立项决策链（选型依据） |
| docs/03-评测协议 | ✅ 保留 | PPL/吞吐/NIAH 口径定义 |
| docs/design-m*.md | 📦 archive | 各里程碑设计稿（已被本文取代，留作设计依据引用） |
| docs/architecture.md（旧） | 📦 archive | M6 时代架构，**过时**（描述旧引擎） |
| docs/01-目录结构 / 02-开发环境 / 04-量化器设计 | 📦 archive | 立项期设计（环境篇 02 仍有效：工具链版本与约束） |
