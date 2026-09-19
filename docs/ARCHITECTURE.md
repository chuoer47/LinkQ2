# qserve-lab 架构与项目主线（唯一入口）

> 生成于 2026-09-18。本文是项目的**唯一权威入口**：架构分层、M0→M10 演进主线、
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
| 投机推理 | n-gram / lookahead（查表）+ chained draft model；接受律 greedy argmax 前缀 + 温度 Leviathan 概率比（两者都无损） | vLLM V1 / TGI / Leviathan et al. |

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
| 8B + draft γ=2 自然文本（融合后） | 114.4（M10 复测 116.5） | 1.17×；natural 列整体抖动 ±6–9% | results/m9_draft_fused.txt + m10_draft_sweep.txt |
| 8B + draft γ=4 自然，**开自适应窗口**（只裁不涨） | 98.2–116.1 | 1.00–1.18×（同配置固定窗口 0.91×） | results/m10_draft_sweep.txt |
| 1.7B + n-gram γ=4 复读，**T=0.7**（概率比接受） | 394.7 | 同格 greedy 449.0 → 少 12.4% | results/m10_spec_temperature.txt |
| 1.7B 新 runtime FP16 decode | 148.7 | 3.5×（vs M0 42.1） | notes/M8 §五 |

### 2.2 质量

| 指标 | 数字 | 口径 |
|---|---|---|
| 8B W4 PPL | 16.07 → 17.31（+1.24） | WikiText-2, AWQ 配方 |
| 8B KV4 PPL | 16.06 → 16.43（+0.369, +2.3%） | bench_ppl_kv.py（HF 模拟法） |
| 权重压缩 | 16G → 2.49G（**6.6×**） | qslab_w4_v1 打包 |
| KV 显存 | fp16 的 **2×** 节省（池口径）/ 3.5×（序列口径） | notes/M8 §四 |
| 32K NIAH | fp16 与 KV4 双 100% | results/m2_niah.json + M4 |
| 前缀缓存 TTFT（3800-token 前缀，8B） | 555.9 → 62.3ms（**8.92×**） | results/m9_prefix_cache.txt（M10 重跑；无 W4 开关，fp16 权重需 UTIL=0.9） |
| W4 主线权重常驻（释放 v1 pack 后） | 8B **3.339 GB = 0.26× fp16**（双份时 6.674 GB）、1.7B 0.678 GB；KV 池 1.52→**3.16 GB（+108%）** | results/m10_w4_residency.txt（改动前后两版表都在） |
| CUDA Graph | batch1 6.5×、8 路连续批 1048 tok/s | notes/M8 §五 |

### 2.3 关键结论（一句话版）

1. **W4A16 的收益在 decode 是带宽账**：Marlin M=1 达 911 GB/s（带宽极限 1008）。
2. **自研 GEMV 输给 Marlin**，e2e 上 crossover 不是 M=8 而是 0——微基准不能外推。
3. **KV4 的 PPL 代价 +0.37 可接受**；"逐 token 无损"在 4-bit 是混沌不可达，PPL 才是仪器。
4. **静态 K scale（SmoothAttention）是 CUDA Graph 可捕获的前提**——写入与数据无关。
5. **投机推理两 proposer 各有主场**：n-gram 统治复读（3.19×，提案免费）、
   draft（torch.compile 融合后）补自然文本——但 M10 复测把这份收益收窄到 **γ=2**
   （1.19×），γ≥3 一律回到 ≈1.0× 的噪声带里，所以"draft 统治自然文本"是**薄**结论。
6. **bs=1 时 kernel 数是税不是字节**：W4 draft 步时不改（2241 kernel/步），
   融合（torch.compile）才是正解，4.53→2.59ms。
7. **批处理下单条输出漂移不是缺陷**：cuBLAS 换 tiling，HF 同样漂移——对比必须同批口径。
8. **投机一旦服务温度采样，正确性判据就从 token 变成分布**：概率比接受 +
   `norm(max(0,q-p))` 重采样使输出与目标模型同分布（N=20000、T=0.9 时 TV<0.06，
   同代码对照组 q≠p 的 TV>0.3 证明该检验有功效）；greedy 是它 p=one-hot 的退化情形。
   **但这件事有价签**：T=0.7 时免费提案器全线劣化（ngram copy 3.06→2.68×、
   lookahead natural acc 0.62→0.00 即 tok/step 退化到 1.00），税来自**接受率塌**而不是
   那份 float32 提案分布（one-hot 路径根本不分配，peak 5.66→5.67 GB）。
   **结论：投机目前是 greedy 场景的功能**；要温度解码就 γ≤2 或关投机。
9. **窗口控制器是单向棘轮**：接受判据是**最长前缀**，一轮落在 2/2 对"第 3、4 条会不会被
   接受"零信息，所以"高接受率就把窗口涨回去"读的是噪声——实测回升 0.90–1.11×（mean 0.98×）
   劣于只裁不涨 1.00–1.18×（mean 1.07×），回升分支已回退、数据保留。顺带一条诚实注脚：被删
   的那条分支阈值对着 **γ 上限**写，折半后按定义不可达，所以旧代码**事实上**早就是棘轮——
   本次只是把这个事实从意外变成交代（代码/测试/文档三处一致）。要回收窗口需要
   **逐位置**接受率，不是逐序列标量。
10. **Marlin 的 repack 是"换掉"而不是"多加"一份 int4**：`uses_v1_pack` 让 `W4Linear`
    在 repack 后释放 v1 打包缓冲，主线常驻从 0.52× 降到 **0.26× fp16**（8B 6.674→3.339 GB）。
    与 dispatch 策略无关（`w4.auto` ≡ `w4.marlin`）。折成 KV 池是实测 **+108%**（1.52→3.16 GB），
    不是先前按"字节直接进池"估的 +220%。
11. **评投机不要用带随机字的 prompt**：随机家族在 T=0.7 出现过 3.3×/3.8× 的"高潮"，探针
    证明该温度下模型退化成周期为 1 的重复 token（56/64 个同一 id），任何提案器都能命中——
    测到的是轨迹形状，不是提案质量。

---

## 3. 架构（当前真实代码）

```
L4  qslab/api/          LLM 门面 + CLI —— M10 起包的是新 runtime（llm_engine.LLMEngine）
L3  qslab/runtime/      新 runtime（M8 起，主引擎）:
                          llm_engine.py   LLMEngine（add_request/step 连续批循环）
                          scheduler.py    调度（prefill/decode/verify 分支 + 投机提交 + 自适应 γ）
                          model_runner.py prepare_prefill/decode/verify + CUDA Graph 捕获
                          draft.py        DraftProposer（chained：第二个完整 runtime）
                          ngram.py        NGramProposer / LookaheadProposer（CPU 查表）
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

| | 旧引擎 `qslab/engine`（M0-M7） | 新 runtime `qslab/runtime`（M8-M10，主线） |
|---|---|---|
| 形态 | 单请求 decode 循环，transformers 前向 | nano-vllm 骨架：paged KV + 连续批 + CUDA Graph |
| KV | KVCacheStrategy（dense fp16/kv8/kv4 + kv4.paged） | int4 池（K 静态 channel / V 动态 group） |
| 投机 | SpeculationMode：chained/lookahead/dynamic | ngram / lookahead / draft（proposer 接口统一）+ 自适应 γ |
| 入口 | `qslab/engine/QslabEngine`（bench/对拍直用） | `qslab/api/llm.py`（LLM 门面 + CLI），或直接 `runtime.LLMEngine` |
| 状态 | **冻结**，oracle 对拍参照 + spec 三模式代码 | 主线，全部最终数字出自这里 |

规则不变：每层只 import 直接下层；量化决策在 L1（backend），模型不感知 kernel。
门面即新 runtime 的对外入口（M10 起）；bench 与部分测试仍直接构造
`runtime.LLMEngine`，为的是拿住 step 级循环与显存配比。

**一次投机 decode step 的数据流**（新 runtime）：

```
LLMEngine.step()
  ├─ Scheduler.schedule() → prefill / decode / verify 三分支
  ├─ _propose(): ngram.propose_batch（CPU 查表）或 draft.propose_batch（GPU 紧循环）
  ├─ ModelRunner.run_verify(): M=γ+1 行一次前向（(bs,M) CUDA Graph 族）
  │    └─ attention decode 分支: Triton kv4_paged_decode_kernel(M)
  │         读取上界 = L+m —— 因果性=槽位下界，draft KV 天然被排除
  └─ Scheduler.postprocess_verify(): 逐行接受（greedy=最长 argmax 前缀 / 温度=概率比
       + 重采样）→ 提交 → trim；接受数写进 spec_stats，自适应 γ 在这里被喂窗口
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

### M10 — 功能收尾（门面 / 温度投机 / lookahead / 动态 γ）

M9 之后主线够快但不够"全"：L4 门面还挂在旧引擎上、温度>0 的序列投机时静默退化成
普通 decode、旧引擎的 lookahead/dynamic 两模式未迁移、8B+draft 只有 bench 没有测试。
本里程碑收掉全部功能遗留（**本身不产性能数字**，产的是可达性与正确性；数字在下一节
"M10 续"）。

**① L4 门面接到 runtime**
- `api/llm.py`：`LLM(model, w4=, smooth_kv=, spec=, spec_gamma=, draft=, **Config)` 包
  `LLMEngine`；`generate/generate_batch` 返回 `{text, token_ids, stats}`；
  `kv_memory_bytes()` / `weight_memory_bytes()` 把 M8 的显存账直接摊到用户面前。
- 门面 `SamplingParams` 默认 `temperature=GREEDY=1e-6`：runtime 没有 greedy 分支，
  logits 除以 1e-6 让 softmax 成 argmax one-hot——沿用全部 runtime 测试的口径，
  而不是在门面里另造一条通路。温度 ≤0 也归一到 1e-6（0 不是"关闭采样"的暗号）。
- `cli.py` 重写为 runtime 词表（--w4/--smooth-kv/--spec/--draft/--stats-only）；
  `--device` 在 import qslab **之前**翻成 `CUDA_VISIBLE_DEVICES`（runtime 绑第一张可见卡）。
  旧引擎专属旋钮（kv_mode/kv_plan）刻意不暴露。
- 实跑冒烟：greedy+ngram、T=0.8+lookahead 两条命令各 16 token，输出连贯、stats 正常。

**② 温度投机：概率比接受律（Leviathan）**
- `ModelRunner.rejection_verify`：按 `min(1, q(x)/p(x))` 接受，拒绝后从
  `norm(max(0, q-p))` 重采样。greedy 行仍走原 argmax 最长前缀路（4-bit 近并列翻转是
  混沌，M8 口径），所以既有的 15 个 spec/draft e2e 测试一字未改照常绿。
- proposer 接口新增 `propose_probs`：`[bs·(γ+1), V]` 的**float32** 提案分布矩阵，行号与
  verify 批的展平顺序一一对应。用 float32 不是省心的缘故：低概率提案在 fp16 下溢成 0，
  而 p=0 会把"该拒的"变成"必受"。
- ⚠ **对齐陷阱**：draft 首次 prefill 吐出的那个 token 本身就是提案 0，所以它的分布必须
  跟着 prefill 一起捕获（走 `_prefill_capture`，`runner.run()` 那条路把分布丢了）。
  错了不报错——接受率照样好看，只是接受的是错位的分布。
- one-hot（查表类提案）与概率比是**同一条规则**：p 为 δ 函数时 `min(1,q/p)` 退化成
  "q 命中提案即受"，用 `scatter_` 置零 x 列实现，无需分支。
- 证据：`tests/test_spec_acceptance.py` — V=64、N=20000、T=0.9 下输出经验分布与目标
  分布 TV<0.06；同代码对照组（q≠p）TV>0.3，证明这条检验真的有功效。

**③ lookahead + 动态 γ 迁移**
- `runtime/ngram.py::LookaheadProposer`：逐序列持久索引 + 链式延伸（在上一步提案的
  延续里继续找）+ **首次出现优先**（n-gram 是最后出现优先）——同一 prompt 两种查表
  策略给出不同提案，这是设计差异不是 bug。
- `Scheduler._adapt_gamma`：WINDOW=3 滑动均值，**≤1.0 折半、其他一律保持**（单向棘轮），
  下限 1、上限即 `config.spec_num_drafts`。**只裁提案条数、不重建 verify 图族**：
  M=γ+1 已烤进 `graphs_verify`，所以动态上限就是配置上限。
  ⚠ 旧引擎 M6 版还带一条回升判据，但写的是 `avg >= self.gamma - 0.1`——`avg` 的上限就是
  **当前窗口**，折半后永远到不了 γ-0.1，所以那条回升**按定义不可达**：老代码其实已经是棘轮，
  只是挂着死分支和一段不成立的"滞回防抖"说辞。M10 把它改成可达写法（对着 `proposal_gamma`）
  在 8B 上复测：回升 0.90–1.11×（mean 0.98×） vs 只裁不涨 1.00–1.18×（mean 1.07×）——
  **证伪，于是删除死分支而不是修复它**。
  机制：接受是**最长前缀**判据，落在 2/2 的一轮对"第 3、4 条会不会被接受"零信息，
  回升读的是噪声（底稿 `results/m10_draft_sweep.txt`，测试
  `test_adaptive_gamma_shrinks_and_never_grows_back`）。
- 只对"按条付费"的 proposer 有意义：ngram/lookahead 提案是 CPU 查表，多提不花钱，
  所以开关对它们不生效（有测试锁住这条）。

**④ 8B+draft 测试固化**
- 一张 24GB 卡上两个完整 runtime、两套 int4 池：target `gpu_memory_utilization=0.62`、
  draft `0.9`、`max_num_seqs=4`（比单模型用例的 0.82 低，正是共卡的代价）。
  断言结构：步数、提交数、`spec_stats` 自洽、平均 ≥2.0 tok/步——不比 token。实跑通过。

### M10 续 — 证据链补测（2026-09-19）

M10 本体只补功能，于是留下三处"口算/口录/从未测"。本轮把它们全部落到 `results/`，
TODO 的 6 条缺环一次结清：

| 测什么 | 脚本 | 底稿 | 一句话结论 |
|---|---|---|---|
| draft γ 扫描可复现性 | `06-spec-draft/bench_spec_draft.py` | `m10_draft_sweep.txt` | copy 六格 ±0.6% 复现；natural ±6–9% 是 4-bit 贪心混沌，非代码差异 |
| 动态 γ 三策略对照 | 同上（`ADAPTIVE`） | 同上 | 只裁不涨 1.07× > 回升 0.98× ≈ 固定窗口 0.91× → 回升证伪、回退 |
| T>0 概率比接受的税 | `06-spec-draft/bench_spec_temperature.py` | `m10_spec_temperature.txt` | +12.4%（ngram copy）到 +63.0%（lookahead copy）；来源是接受率塌，不是 float32 分布 |
| Marlin 路径双份 int4 | `02-w4-quant/bench_w4_residency.py` | `m10_w4_residency.txt` | 8B 死重量 **3.335 GB**（口算 1.9 GB 低近一倍）；`w4.auto`≡`w4.marlin` |
| ↳ 由这条测量直接引出的改动 | 同脚本复测 | 同上（复测一节） | 释放后 8B 常驻 3.339 GB（0.26× fp16）、池 1.52→3.16 GB；**"+220%" 的换算被自己的复测证伪**（池只兑现预算 ~51%） |
| 8B 前缀缓存 | `07-prefix-cache/bench_prefix_cache.py` | `m9_prefix_cache.txt`（追加） | 555.9→62.3ms（8.92×），与口录差 0.4%/1.1% |

**方法学收获（比数字更值钱）**：① 一次非单调（natural tok/step γ=3 1.97→γ=4 1.91）
**没找到解释就写"未证"**，不编故事；② 随机提示家族在 T=0.7 冒出的 3.3×/3.8× 用一次性探针
证伪（模型退化为周期 1 的重复 token），这类"好得可疑"的格子必须探针复核；③ 我自己的第一版
读数是"T>0 就该换 draft"，数据出来发现 natural@T=0.7 三种提案器全输（0.95/0.89/0.70×），
底稿上传前改判；④ 常驻底稿被它所引出的改动复测了一次，结果**推翻了自己两条结论**
（"只有走 `w4.v1` 才省得下" 和 "+220% KV 池"）——原始读数不删，复测一节追加在后面。

### 测试现状（locks）

非 e2e 46 + e2e 38（含 8B 3、门面 6、投机接受律 7）= **84 全绿**
（2026-09-18 首次全量；09-19 因 `_adapt_gamma` 定稿 + 棘轮断言改名重跑 81/81，同日
释放 v1 pack 后再跑 **84/84**，新增 3 条 CPU 常驻契约测试，单卡串行）。相比 M9（61）净增 23：接受律/自适应 γ 7 + lookahead 查表 4（快门档）、门面 6、
spec runtime 温度路径 2、8B+draft 1、W4 常驻契约 3。自适应那条断言现在锁的是**单向棘轮**
（`test_adaptive_gamma_shrinks_and_never_grows_back`），别再按"能涨回去"写测试。
断言口径：**不逐 token 断言量化输出**（混沌），断言数学结构（接受机制/分布/步数/确定性/
首位 token/前缀稳定性）。pytest: `pytest tests/ -m "not e2e"` 快门（秒级）/
全量单卡串行 **3min24s**（81 只，2026-09-18 实测；09-19 的 84 只未单独计时）。

---

## 5. 已知限制与下一步

M10 收掉了原 1-5 号功能遗留（门面 / 温度投机 / 动态 γ / lookahead 迁移 / 8B+draft 测试），
其补测轮（2026-09-19）把 6 条证据链缺环全部落盘。下面这些不是"没测过"，而是**测出来的
结构性限制**，加一件确实没做的事：

1. **温度投机目前是净亏**：概率比接受在分布上无损（TV<0.06），但吞吐税实测
   +12.4%（ngram copy）到 +63.0%（lookahead copy），natural 上三种提案器 T=0.7 全部
   ≤1.0×（0.95/0.89/0.70）。**税不在 float32 提案分布，在接受率塌**。所以"投机"当前
   只在 greedy 场景成立；温度解码要么 γ≤2 要么关投机。要救回来需要**逐位置**接受率信号
   （`results/m10_spec_temperature.txt`）。
2. **窗口控制器上限是结构性的**：`_adapt_gamma` 只能在 `[1, spec_num_drafts]` 里裁，
   因为 verify 图族按固定 M=γ+1 捕获——越过配置值要为每个新 M 再录一族图，没做。
   而且 WINDOW=3 与"折半"系数至今**没扫过参**（只裁不涨这件事被验证了，阈值本身仍是
   M6 先验）。回升分支已证伪回退，别再往回加（`results/m10_draft_sweep.txt`）。
3. **~~只要走 Marlin 就多常驻一份 int4~~ 已闭合（09-19）**：`W4MarlinBackend` repack
   完成后不再持有 v1 `qfp/scale`，后端用 `uses_v1_pack` 声明这件事，`W4Linear` 据此
   决定要不要把它们注册成 buffer；`memory_bytes()` 改算各自实际常驻的那一份。
   复测：8B 常驻 **6.674 → 3.339 GB**（0.52×→**0.26×** fp16）、1.7B 1.355→0.678 GB，
   `v1 pack`/`dead v1` 两列归零。原来"想省只有走 `w4.v1`、代价是 decode 慢一半"
   **作废**——Marlin 主线现在与 `w4.v1` 同宽。同时纠正一条换算：腾出的 3.335 GB
   进的是**分配预算**（8B budget 2.94→6.14 GiB），KV 池只兑现其中 ~51%，实测
   1.52→**3.16 GB（+108%）**，不是先前估的 4.85 GB/+220%；token 容量 ≈2.1 万→≈4.4 万
   （`results/m10_w4_residency.txt` 复测一节）。池子为何只拿一半预算**未证**。
4. **128K YaRN demo 未跑**（M4 起遗留）。
5. **旧 `qslab/engine/spec/` 仍在库**：lookahead/dynamic 的逻辑 M10 已迁进 runtime，
   旧包留作冻结 oracle 参照，等旧引擎整体归档时一并处置。

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
