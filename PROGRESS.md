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
- [ ] **遗留：8B 的 KV4 静态 K scale 覆盖不足**——运行时 K 有 18~66 个通道超出
      校准上限（1.7B 仅 1 个），KV 相对误差 ~0.15（1.7B ~0.12），greedy 在第 5 个
      token 附近翻转。8B 测试据此断言连贯性而非逐 token 相等
- [ ] 前缀缓存验证

### 行为无回归基线（每阶段复测）
8B W4 e2e 39.8 tok/s | PPL 17.31 | KV4 省 3.5× | lookahead 1.44× | oracle 对齐 PASS
