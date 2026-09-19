# PROGRESS.md — qserve-lab 唯一状态源

## 当前阶段：**R0-R6 工程化重构**

### R0 flash-attn 安装
- [x] flash-attn 2.7.4 (cu124/torch2.5/py3.11 预编译 wheel) 装入 qslab env
      来源：mjun0812/flash-attention-prebuild-wheels v0.3.18
      验证：flash_attn_func 在 sm_89 上实测通过（2026-09-16）
      坑：wheel 文件名必须合规（fa.whl → 全名重命名），pip 才接受

### R1 分层骨架
- [x] L0 qslab/kernels（csrc+ops+marlin）/ L1 qslab/quant（算法+packfmt+cache）/ L2 qslab/models / L3 qslab/engine（core+cache→quant+spec）/ L4 qslab/api（空壳待 R3）
- [x] 设计修正：KV cache 归属 L1（"数据怎么存"）而非 L3——解决了 L2↔L3 循环依赖（2026-09-16）
- [x] transformers 违规修复：loader 的 HF 构建移到 adapters/model_builder.py，L2 零 transformers（2026-09-16）
- [ ] 遗留：L2 w4linear 直连 L0 kernels（3 处）——R2 的 QuantBackend 正式消除
- [x] 验收：1.7B oracle 冒烟 PASS（输出与重构前一致）

### R2 策略接口
- [x] registry.py 通用注册表（装饰器注册 + 工厂查询）
- [x] QuantBackend(L1)：w4.v1 / w4.marlin / w4.auto 三后端，**M 分派内化到 backend**；W4Linear 只调 backend.linear()，L2->L0 直连彻底消除（2026-09-16）
- [x] KVCacheStrategy(L1)：fp16/kv8/kv4/kv4.plan 策略 + 注册表；engine 的内联 kv_mode 分派改为策略工厂；**paged 位已预留**（M7 用）
- [x] SpeculationMode(L3)：chained/lookahead/dynamic 策略类，验证/回滚逻辑与提案策略解耦；补回 lookahead 空提案回退（重构中丢失的路径）（2026-09-16）
- [x] 验收：三模式 lossless=True（ar 4.00/1.75/4.71）；QuantBackend 冒烟输出与基线一致

### R3 统一入口
- [x] qslab/api/llm.py：LLM 门面（model + w4 + kv_mode + draft 正交组合，SamplingParams，stats/kv_memory/weight_memory 查询）（2026-09-16）
- [x] qslab/api/cli.py：python -m qslab.api.cli generate --model ... [--w4 --kv-mode --draft --spec-mode]
- [x] 修复 R2 引入的真 bug：lookahead 空提案回退分支的 commit 位置算错（start_pos 少 1）导致 KV 错位、无损性失败——**由统一入口的端到端测试抓出**（2026-09-16）
- [x] 验收：三种 CLI 组合全部输出正确；W4+lookahead LOSSLESS=True

### R4 仓库清理
- [x] committed 二进制（test_w4a16_gemm）与 22 个 results/*.log 出 git，.gitignore 补全（保留 15 个结果 json）（2026-09-16）
- [x] 14 个 dbg_* 归档到 scripts/archive/debug/
- [x] m*_ 重组：4 个 bench → benchmarks/、4 个正确性门 → tests/、7 个一次性实验 → scripts/archive/
- [x] **抓出并修复重构引入的静默回归**：R2 的 sed 误伤把 loader 里 `_wrap_input_scale(mod, s)` 改成裸元组 `(mod, s)`（Python 合法但 no-op）→ load_w4_model 的 AWQ fold-back 全失效、输出乱码。oracle_alignment 恢复 True（2026-09-16）

### R5 测试与锁定
- [x] pytest 9.1.1 装入 qslab env；pytest.ini 定义 gpu/e2e 标记（2026-09-16）
- [x] tests/test_unit_cpu.py（9 个，2.1s）：pack roundtrip / 注册表语义 / 采样数学 + 拒绝采样分布无损性
- [x] tests/test_gpu_kernels.py（6 个，2.3s）：v1 kernel vs 反量化 matmul / 三 cache roundtrip 精度分级 / KV4 显存比 / KV4 奇数长度更新
- [x] tests/test_e2e_engine.py（4 个，24s）：FP16 引擎 vs HF oracle / 三种投机模式无损性
- [x] environment.yml（conda: nvcc12.4+gcc13）+ requirements.txt（pip 精确锁定 + flash-attn 来源说明）
- [x] 全套 19 passed

### R6 文档
- [x] README.md：结果表（四配置对比）、五层架构图、快速开始（CLI + Python 两种）、技术栈说明、测试表、仓库地图、路线图（M7 量化 paged attention + nano-vllm 整合）（2026-09-16）
- [x] docs/architecture.md：数据流（一次 decode step 的完整调用链）、三策略接口的注册与扩展点表、"想加什么动哪里"对照表
- [x] 最终验收：19 测试全过；关键指标无回归（W4 39.5 vs 39.8 tok/s、lookahead 1.42x vs 1.44x、AR 4.96 完全一致）

### 行为无回归基线（每阶段复测）
8B W4 e2e 39.8 tok/s | PPL 17.31 | KV4 省 3.5× | lookahead 1.44× | oracle 对齐 PASS

### 重构完成（2026-09-16）

R0-R6 全部完成。仓库从"按里程碑堆叠的研究代码"变为五层分层、策略化、有测试有文档的工程仓库。
7 个阶段 commit + 修复 2 个重构引入的真 bug（空提案 KV 错位、_wrap_input_scale 静默 no-op）。

---

## M7 量化 paged attention（2026-09-16 完成）

- [x] M7-s1 L0：`KV4PagedCache`（BLOCK_N=128 块对齐，GROUP=64 不跨块）+ Triton kernel
      （tile 内反量化，不 materialize dense fp16）；vs dense 参考 max diff 0.0
      修 3 bug：uint32 下溢（先转 int32）/ 转置 / fp16 scale 下溢（下界 1e-4）
- [x] M7-s2 接入引擎：`kv_mode="kv4.paged"` 策略注册 + attention decode 分支（GQA 映射）
      paged vs dense 端到端输出完全一致
      修 4 bug：**组内 scale 污染**（追加时要重算整组）/ GQA 头映射 / uint32 下溢 / 转置
- [x] M7-s3 8B 验收：paged PPL **18.50** vs dense 18.70（差 -0.20，噪声级）
      **@8K 峰值显存 24.4 → 18.4 GB**（O(context) 的 fp16 副本消失）
- [x] M7-s4 notes/M7-量化paged-attention.md

## M8 nano-vllm runtime 整合（进行中）

- [x] M8-s1 vendor：nano-vllm runtime（MIT）进 `qslab/runtime/`——调度/分页/块管理/
      CUDA Graph 骨架保留；TP 多进程剥离；模型定义换成我们的 primitives
- [x] M8-s2 引擎端到端打通。**两个真 bug**：
      1. **nibble 编码不对称**——store 写 offset-binary(`qi+8`)，decode 读
         two's-complement(`nib-16`)，所有值整体偏 8 个量化级
      2. **V 非连续视图**——`v` 来自 `qkv.split()`（行 stride 4096 ≠ H*D 1024），
         store kernel 按连续布局读 → 池里 V 全错；flash-attn 自己处理 stride，
         所以 prefill 无恙、只有 decode 崩
      另修：KV 显存预算用全卡口径（共享卡上算出负数）→ 扣除他进程占用
- [x] M8-s2 K/V 量化方案定版（原 per-token 方案被实测推翻，见 docs/design-m8.md 追加）：
      **K = int4 + 静态 per-channel scale（SmoothAttention 离线校准）**、
      **V = int4 + 动态 per-token group=64**。
      依据：vLLM/TRT-LLM 离线校准存 checkpoint；KIVI K per-channel / V per-token；
      QServe SmoothAttention `λ_i=max|K_i|^0.5`（且 λ_d=λ_{d+D/2} 以与 RoPE 交换）
- [x] 验收：**1.7B e2e 与 HF oracle 逐 token 一致**；KV 池 6.92 GiB（fp16 需 13.84，**2×**）；
      全套 20 测试通过
- [x] M8-s3 **CUDA Graph 实测通过**：捕获 bs=[1,2,4,8] 4 个桶，输出与 oracle 逐 token
      一致（静态 per-channel K scale 的设计目标兑现：写入纯 slot 驱动）；
      decode 吞吐 batch1/128tok **22.8 → 148.7 tok/s（6.5×）**；
      8 路连续批 64tok **286.7 → 1048.2 tok/s**
- [x] M8-s3 批处理正确性归因（单条 vs 批处理有差异，逐项排除后判定非缺陷）：
      flash-attn varlen 打包 **Δ=0**；我们 batched logits vs HF batched **top-1 8/8**；
      输出不依赖同伴内容；内核重放 **Δ=0**；**HF 自己 solo vs batched 同样漂移
      0.021~0.033**。根因是批大小改变 GEMM 的 M 维 → cuBLAS 换 tiling → 归约顺序变
- [x] M8-s4 **W4 权重接入 runtime**：`swap_w4()` 把 fp16 Linear 换成 L1 打包 W4Linear；
      运行时模型改为**独立 q/k/v 与 gate/up 投影**（不融合）——打包 checkpoint 按
      HF 模块名存权重、AWQ 输入缩放按逻辑模块存，融合会丢掉 per-shard scale；
      实测融合只值 3%（149.1→144.8 tok/s）。独立投影还从根上消除了 V 非连续的隐患
- [x] M8-s4 修两个真 bug：
      ① **lm_head 维度声明反了**（HF 存 [vocab,hidden]，而 Linear 是 [out,in]）；
         1.7B 因 tie_word_embeddings 侥幸通过，8B 加载即报 shape mismatch
      ② **w4a16_gemm 启动在默认流**（无 stream 参数），CUDA Graph 捕获在旁路流——
         该 GEMV 没被录进图、回放时跳过、输出全 0。改用 getCurrentCUDAStream()
- [x] M8-s4 8B 验收（部分）：prefill logits vs HF **max|Δ|=0.018**（top-5 相同）；
      decode **55.8 tok/s**（graph）；输出连贯（"Paris. The capital of Germany is Berlin."）
- [x] **KV4 的 PPL 代价已量化**（`benchmarks/bench_ppl_kv.py`，在 HF 上模拟方案、
      绕开 runtime）：8B **16.06 → 16.43（+0.369, +2.3%）**；1.7B 26.70 → 27.80
      （+1.098, +4.1%）。归因：K 单独 +0.143、V 单独 +0.086，合并 +0.369（超加性）
- [x] **更正**：此前"8B 静态 K scale 覆盖不足（18~66 通道超限）"是错的（归约维度搞错，
      正确测量仅 0.0~0.2%）；"逐 token 无损"也不是可达指标——4-bit 下 greedy 是混沌的，
      两个模型都只有 1/8 prompt 完全一致。**PPL 才是正确仪器**
- [x] **前缀缓存 bug 已修**（静默错答）：命中哈希复用的前缀只存在于 int4 池里，
      而我们的 prefill 只用刚算出的 fp16 K/V、不读池 → 序列静默丢掉前缀。实测同一
      prompt 第二次提交结果不同（[576,3840,...] vs [0,220,576,3840,...]）。当前
      **禁用前缀复用**（`BlockManager.ENABLE_PREFIX_CACHE = False`），并在注释里写明
      重新启用的前提：prefill 需能从池里物化 int4 缓存前缀。测试 tests/test_prefix_cache.py
- [x] 删除 8B 的 "coherence" 断言：它不是指标，阈值（"不算复读"）恰好卡在 4-bit
      greedy 的混沌分界线上，会随机变红。8B 的正确性改由 PPL 覆盖（+0.369）

### 行为无回归基线（每阶段复测）
8B W4 e2e 39.8 tok/s | PPL 17.31 | KV4 省 3.5× | lookahead 1.44× | oracle 对齐 PASS

## M9 n-gram 投机解码（2026-09-18 完成）

- [x] M9 方案：docs/design-m9.md（业界依据 vLLM V1 ngram + TGI append-only）
- [x] L0：decode kernel 通用化 M query（grid (N_Q, bs*M)，因果性=槽位下界，
      M=1 逐指令等价）；对拍/因果性/-1 防护 7 测试
- [x] L3：`runtime/ngram.py` proposer（numpy 滑窗，纯 CPU）
- [x] L4：调度 verify 分支 + `run_verify`（M=γ+1 一次前向）+ greedy 接受 +
      预留/trim 槽位（append-only 回收）+ padding 中立 + (bs,M) graph 族
- [x] **Marlin crossover 修正**：M5 微基准 crossover=8 被 e2e 推翻
      （M=1: 8B 54.1→91.1 tok/s；M=5 verify: 65→11.5ms/步）；
      **M8 遗留"8B 55.8 未查因"的答案就是 v1 GEMV**；主线 8B decode
      **97.2 tok/s（1.75×）**；Marlin/v1 数值互差 0.047（无损）
- [x] 验收：1.7B 复读 3.13×（γ=8 5.77×）；**8B W4A16KV4 复读 310.4 tok/s
      =3.19×**、自然 0.95×（诚实开销）；复读输出 128/128 逐 token 一致；
      新增 22 测试，全量回归绿
- 详见 notes/M9-投机推理.md

### M9 续：draft-model 投机迁入（2026-09-18 完成）

- [x] `runtime/draft.py`：chained draft（Qwen3-0.6B）= 第二个完整 runtime
      （自有 paged int4 池/graph/scheduler，vLLM V1 draft worker 形态）；
      proposer 接口统一 propose_batch，提案阶段上移 engine
- [x] 锁步协议：sync 截断（draft KV 前缀天然有效）+ prefill + γ 步紧循环
      （免调度往返）；修 3 bug（spec_method 只认 ngram / num_cached 漂移越界 /
      may_append 非幂等）
- [x] 结果：8B W4A16KV4+0.6B 复读 **143.5 tok/s（1.47×）**；自然文本 0.73×
      ——瓶颈实测为 draft 每步 4.3ms（0.6B fp16 前向 ~2ms + Python/同步 ~2ms），
      后续 W4 化 draft / fused argmax；n-gram 与 draft 可按负载互换
- [x] 新校准 smooth_kv4_qwen3-0.6b.pt；4 个 draft 测试；全量回归绿
      （31 非 e2e + 14 e2e + 11 spec）

### M9 续2：draft kernel 融合 + γ 帕累托（2026-09-18 完成）

- [x] W4 化 draft 实验证伪（GEMV 2.50→1.19ms 但步时不变：2241 kernel/步，
      ~1500 个未融合 elementwise 占 2.6ms——税在 kernel 数不在字节数）
- [x] **torch.compile 融合**（attention dynamo-disable + inductor 融层间
      elementwise + 录进手写 CUDA Graph）：**draft 步 4.53→2.59ms（1.75×）**
- [x] 踩坑修复：inference_mode 差异触发首提案 ~5s 重编译 → init 预热
- [x] γ 帕累托：自然 **γ=2 1.17×** / 复读 **γ=8 2.30×**；n-gram 统治复读、
      draft 统治自然文本——两种 proposer 各有主场
- 11 spec 测试全绿

### M9 续3：前缀缓存修复复活（2026-09-18 完成）

- [x] M8 静默错答 bug 修复：materialize_kv 反量化 + prepare_prefill 一次
      slot 计划下发全层（第一版每层 tolist 同步 ~14ms，教训：跨层计划不进层内）
      + attention prefill 物化拼接；`ENABLE_PREFIX_CACHE=True` 默认开
- [x] 同族修复：chunked prefill 第二块的同款错位（一次修复两个 bug）
- [x] spec 提交补 hash_blocks（投机序列也能贡献缓存条目）
- [x] 验收：8B 3800-token 前缀 **TTFT 558.3→63.0ms（8.86×）**；1.7B 2.73×；
      171-token 复现实验输出正确（错误前导 0 消失）；全量回归绿

### 收尾：文档脉络重写 + 证据链整理（2026-09-18 完成）

- [x] docs/ARCHITECTURE.md：统一入口（分层/M0-M9 主线/最终状态/两代引擎/已知限制），
      旧设计稿 stub 化归档至 docs/archive/（cc2bc45）
- [x] README 重写指向 ARCHITECTURE.md（headline 数字更新为 M9 口径）
- [x] benchmarks/ 按主题 8 分类 + 每类证据链 README + benchmarks/README.md 总索引；
      脚本 sys.path 修正，import 实测通过（5be9b7b）
- [x] TODO.md：3 项证据链缺环 + 7 项功能遗留

### M10：功能收尾（2026-09-18 完成）

- [x] **L4 门面接到新 runtime**：`api/llm.py` 的 `LLM` 改包 `LLMEngine`
      （w4/smooth_kv/spec/draft + Config 关键字透传），`generate` 返回
      text/token_ids/stats，`kv_memory_bytes`/`weight_memory_bytes` 直供显存账；
      `api/cli.py` 重写为 runtime 词表（`--device` 在 import qslab 前翻成
      CUDA_VISIBLE_DEVICES，因为 runtime 绑第一张可见卡）；CLI 两条命令实跑冒烟通过
- [x] **温度采样投机 = Leviathan 概率比接受**：`ModelRunner.rejection_verify` 按
      min(1, q/p) 接受、拒绝后从 `norm(max(0,q-p))` 重采样；proposer 接口新增
      `propose_probs`（[bs·(γ+1), V] float32 提案分布）。greedy 行仍走 argmax 最长前缀，
      所以既有 15 个 spec/draft e2e 测试一字未改照常绿
- [x] **lookahead 迁移**：`runtime/ngram.py::LookaheadProposer`（逐序列持久索引 +
      链式延伸 + 首次出现优先，对照 n-gram 的最后出现优先）；
      **动态 γ**：`Scheduler._adapt_gamma`（WINDOW=3、近期均值 ≤1.0 就把窗口折半，
      **只裁不涨**；上限就是 `config.spec_num_drafts`，因为 verify 图族按固定 M=γ+1
      捕获，只裁提案条数、不重建图族。回升分支曾被接回来复测、量出来更差，见下节）
- [x] **8B+draft e2e 固化**：`test_8b_acceptance.py::test_8b_draft_spec_acceptance`
      （一卡两完整 runtime 两 int4 池，util 0.62/0.9，断言步数/提交/spec_stats 自洽）
- [x] 新增测试 20 只（接受律 7 含 TV<0.06 无偏性 + 功效对照组、门面 6、lookahead 4、
      spec runtime 温度路径 2、8B 1）→ **非 e2e 43 + e2e 38 = 81 全绿，单卡串行 3min24s**
- ⚠ 两个"错了也不会报错"的坑（都是踩出来的）：
  ① draft 首次 prefill 吐出的 token 本身就是提案 0，其分布必须随 prefill 一起捕获
      （新增 `_prefill_capture`），否则整窗错位一格——接受率照样好看，接受的是错的分布；
  ② 提案分布必须 float32：fp16 下低概率 p(x) 下溢成 0，会把"该拒的"变成"必受"
- ⚠ 复核 TODO：原第 3 项（"32K NIAH 无独立 json 底稿"）**不成立**——
      `results/m2_niah.json` 本身就是 ctx=32768 的落盘（fp16/kv4 recall 均 1.0，
      3 深度 ×3 次），产出脚本在库；证据链缺环由 3 项减为 2 项
- 新缺环改记：温度投机的**吞吐**未测、动态 γ 阈值未调参、`w4.auto` 每层双份 int4
      （repack 后 v1 缓冲未释放）——见 ARCHITECTURE §5 与 TODO 4/5
- 说明：M10 不产性能数字（无新 bench），故未建 notes/M10；教训与结论落在本文与
      ARCHITECTURE §4 M10

### M10 续：证据链补测（2026-09-19 完成，缺环 1–6 全部闭合）

M10 本体不产性能数字，留下三处"口算/口录/从未测"。本轮把它们全部变成底稿，
并把 TODO 的 6 条缺环一次结清。

- [x] **①`w4.auto` 双份 int4 常驻**（原口算"8B ≈ 1.9 GB"）：**实测 3.335 GB**，
      口算低了近一倍，已更正。`benchmarks/02-w4-quant/bench_w4_residency.py` +
      `results/m10_w4_residency.txt`。1.7B 死重量 0.677 GB（KV 池 4.41→5.08，+15%）、
      8B 3.335 GB（**1.52→4.85，+220%**；可换出总容量约 2.1 万→6.6 万 token）。
      更深一层：**`w4.auto` 与 `w4.marlin` 逐列相同**——省不掉这一份是 Marlin 后端自身的
      性质（另存 `_B/_s` 排布），与 dispatch 策略无关；M9 把主线换成 Marlin 时就一起进来了。
      释放 `qfp/scale` 的改动**未实现**，底稿只证明空间存在。
- [x] **②8B 前缀缓存落盘**：重跑 cold 555.9ms → hit 62.3ms（**8.92×**），与 M9 口录的
      558.3/63.0 差 0.4%/1.1%，两段原始输出（含命令与 `[kvalloc]` 行）追加进
      `results/m9_prefix_cache.txt`。新记两条此前没写清的事实：bench **无 W4 开关**（8B 走
      fp16 权重），且需 `UTIL=0.9` 才放得下（`peak=16.91GiB`，KV 池只剩 3.69GiB）。
- [x] **③draft bench 入库**：`benchmarks/06-spec-draft/bench_spec_draft.py` +
      `results/m10_draft_sweep.txt`。**copy 列六格在 ±0.6% 内复现** m9_draft_fused.txt
      （最大 184.1→183.4）；natural 列抖动 ±6–9%（γ=3 98.0→103.9、γ=4 97.5→89.0）——
      这是 4-bit 贪心的混沌轨迹，不是代码差异，故 natural 只读成"γ≥3 在 ≈1.0× 噪声带里"，
      γ=3→γ=4 的 tok/step 非单调（1.97→1.91）**未解释、标为未证**。
      顺带修正 README/PROGRESS 里的"draft 统治自然文本"：M10 复测下这份收益只到 **γ=2**。
- [x] **④温度投机的吞吐税**：`bench_spec_temperature.py` + `results/m10_spec_temperature.txt`。
      **归因纠正**：税**不是**每步那份 `[bs·(γ+1), V]` float32 提案分布（lookup 类提案 one-hot，
      走不到分配，peak 5.66→5.67 GB），**而是接受率塌了**——T>0 时"逐位等于贪心前缀"不再成立：
      ngram copy 3.06→2.68×（+12.4%）、lookahead copy 3.54→1.31×（+63.0%）、
      lookahead natural acc 0.62→0.00（tok/step 退化 1.00，等于没投机）、draft copy 1.10→1.05×。
      随机提示家族上 T=0.7 的 3.3×/3.8× 用一次性探针证明是**轨迹假象**（该温度下输出周期为 1
      的重复 token，56/64 个 id 195），已从结论剔除 → **口径教训：不要用带随机字的 prompt 评投机**。
- [x] **⑤动态 γ 阈值在新 runtime 上的行为**：三种策略 × 4 次重复（8B natural，γ 上限 4，
      逐 verify 步轮询 `scheduler.proposal_gamma`）——
      固定窗口 0.91× / **只裁不涨 1.00–1.18×（mean 1.07×）** / 回升判据
      `avg >= proposal_gamma-0.1` 0.90–1.11×（mean 0.98×，4 次里 3 次跌破 1.0）。
      **回升判据被自己的复测量证伪 → 死分支删除、明确只裁不涨**（falsified 数据保留在底稿；
      注：HEAD 原写法 `avg >= self.gamma - 0.1` 阈值对着上限，折半后按定义不可达，所以旧代码
      **事实上**早就是棘轮，只是挂了一段不成立的"滞回防抖"说辞），
      `tests/test_spec_acceptance.py` 相应改成断言**单向棘轮**
      （`test_adaptive_gamma_shrinks_and_never_grows_back`）。机制（未证但自洽）：接受是
      最长前缀判据，一轮落在 2/2 对"第 3、4 条会不会被接受"零信息，回升读的是噪声；
      真正回收窗口需要**逐位置**接受率，而 `avg` 只是逐序列标量。
      仍遗留（降级为优化空间，不再算缺环）：WINDOW=3 与折半系数没扫过；窗口能否越过配置值
      结构上无从验证（verify 图族 M=γ+1 固定）。
- [x] **⑥复核后不成立的一条**：原 TODO 第 3 项（32K NIAH 无底稿）M10 已判定不成立，本轮维持。
- ⚠ **环境坑（值得单列）**：只 `export PATH=…/envs/qslab/bin:$PATH` 而不 `conda activate` 时，
      `CUDA_HOME/CC/CXX` 缺失 → torch 把 marlin 扩展当没编译过、用系统 gcc(>13) 重新 JIT，
      报 `Ninja is required…` / `cusparse.h: 没有那个文件或目录`，整条 W4 线在加载阶段崩。
      纯 fp16 脚本感觉不到，所以这个坑只在碰 W4 的 run 上暴露。已写进 `benchmarks/README.md`。
- 缺环状态：**1–6 全部闭合**。剩余是功能性收尾（128K YaRN demo）与一条未实现的优化
      （释放 v1 pack 的 3.335 GB）。
