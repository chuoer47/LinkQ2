# qslab 架构

本文只描述**当前代码的架构**：包怎么分层、依赖往哪走、一次生成请求经过哪些文件、两种 4-bit 格式的约定、
配置面与入口

**本文写作时的代码 commit**：`53b0bd7dc2c25696b618ef1ae308992bff3f304c`（tests 分层）。
它前面两刀结构改动也在本文的覆盖范围内：`bb2cd86e213d1af9f209304ac01c332cb722c754` 删掉 `qslab/sampler.py`、
`ff9cd12` 把 runtime 拆成四个子包。本文自身在这个 commit 的下一次提交里。

**qslab 是什么**：一个单卡 LLM 推理运行时——权重存 int4、算时反量化到 fp16 做 GEMM（W4A16），
KV cache 也压成 int4（KV4），带分页块管理、前缀缓存、CUDA Graph、连续批处理和投机解码。
模型侧只支持 Qwen3。另有一条与运行时正交的 HF fp16 参照路径，用来验证量化与打包算得对不对。

---

## 1. 组件一览

| 目录 | 规模 | 职责 |
|---|---|---|
| `qslab/` | 47 个 .py + 3 个 .cu，5164 行 | 全部产品代码 |
| `adapters/` | 60 行 / 3 文件 | 与 transformers/HF 生态的接口层 |
| `tests/` | 2480 行 / 18 个 .py（17 个测试文件 + `conftest.py`），pytest 收集 101 项 | 见第 8 节 |
| `benchmarks/` | 1894 行 py，11 个专题目录 + `archive-legacy` | 每个专题一个 harness + 一份 README 底稿 |
| `results/` | 35 份 | 实测输出 |
| `scripts/` | 3 个 | 标定语料、SmoothAttention 尺度、内核编译 |
| `models/` | gitignore | fp16 权重与打包后的 W4 目录 |

## 2. 分层与依赖方向

六层，**依赖只允许向下**。这条规则不是愿望，是实测：AST 走完 `qslab/**/*.py` 的 import，
跨包边 8 条 / 引用 13 处，全部向下，向上 0 条，包级环 0 个。

```
L4  api        用户入口：门面 LLM、CLI、HTTP 服务面            → runtime
L3  runtime    引擎本体（engine / execute / model / state）    → models, quant
L2  models     W4 线性层模块 W4Linear                          → quant
L1  quant      量化算法、后端选择、打包格式、标定               → kernels, registry, reference
L0  kernels    CUDA 扩展的 JIT 加载与 Marlin 封装               →（无 qslab 依赖）
参照  reference HF fp16 oracle：只回答"量化/打包对不对"          → quant
横切  adapters  tokenizer / model_builder；registry 名字→工厂注册表
```

实测的跨包引用：`api→runtime` 3、`quant→kernels` 3、`models→quant` 2，
`runtime→models`、`runtime→quant`、`quant→registry`、`quant→reference`、`reference→quant` 各 1。

两条配套约定：

- `qslab/registry.py`（45 行）自己零 qslab 依赖，所以每层都能 import 它来注册可插拔策略。
- `adapters/` 之外的层不应 import transformers。当前代码与此不符（实测另有 3 处），见 TODO 23。

## 3. 包与模块

### L4 `qslab/api/`（564 行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `llm.py` | 130 | `LLM` 门面 + 自己的 `SamplingParams`；未知关键字直接透传给 `runtime.config.Config` |
| `cli.py` | 109 | `python -m qslab.api.cli generate --model ...`，flag 镜像门面 |
| `server.py` | 323 | aiohttp 的 OpenAI 兼容服务面：`/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`；一个 `EnginePump` 线程独占引擎，按 job 扇出流式增量 |

### L3 `qslab/runtime/`（2784 行，vendored from nano-vllm 后自研）

包内又是一个 DAG：`engine → {execute, model, state, config, sampling_params}`、
`execute → {model, state, config}`，`model` 与 `state` 是叶子。

**`engine/`（872 行）请求生命周期**

| 文件 | 行数 | 职责 |
|---|---|---|
| `llm_engine.py` | 151 | `LLMEngine`：`add_request` / `step` / `generate`，`_propose` 调提案器，`exit` 挂在 atexit |
| `scheduler.py` | 278 | `Scheduler`：`schedule` 决定本步批哪些行（prefill / decode / verify 三分支）、`preempt`、`postprocess`、`postprocess_verify`、`spec_stats`、`_adapt_gamma` |
| `draft.py` | 320 | `DraftProposer`：投机用第二个完整 runtime 跑草稿模型，`propose_batch` 批量出提案（可带概率分布） |
| `ngram.py` | 119 | `NGramProposer` / `LookaheadProposer`：纯 CPU 查表提案，无额外显存 |

**`execute/`（547 行）一步怎么落地**

| 文件 | 行数 | 职责 |
|---|---|---|
| `model_runner.py` | 534 | `ModelRunner`：`warmup_model` → `allocate_kv_cache` → `load_smoothing` → `capture_cudagraph`；每步 `prepare_prefill/decode/verify` 组张量、`run_model`/`run_verify` 回放图、`rejection_verify` 做投机接受 |
| `sampler.py` | 12 | `Sampler`：GPU 批量的温度采样（Gumbel 技巧，只有温度这一个旋钮） |

`model_runner.py` 的 `loop` / `read_shm` / `write_shm` / `call` 是 vendored 的多进程 RPC 骨架，
断言 `world_size > 1`；本仓库是单进程，四个方法没有调用者。

**`model/`（946 行）前向本身**

| 文件 | 行数 | 职责 |
|---|---|---|
| `qwen3.py` | 143 | `Qwen3ForCausalLM` / `Qwen3Model` / `Qwen3DecoderLayer` / `Qwen3Attention` / `Qwen3MLP` |
| `primitives.py` | 145 | `RMSNorm` `SiluAndMul` `Linear` `MergedLinear` `QKVLinear` `VocabEmbedding` `LMHead`（TP 已移除） |
| `attention.py` | 145 | `PagedAttention`：SmoothAttention 的 λ 缩放 → 先写池再读；prefill 走 flash-attn varlen（前缀命中时先物化池里的段），decode/verify 走 int4 paged 内核 |
| `paged_decode.py` | 247 | 两个 triton 内核 + 三个 host 封装：`store_kv_quant_kernel`（fp16→int4 入池）、`kv4_paged_decode_kernel`（读时反量化）、`paged_attention_decode`、`store_kv_quant`、`materialize_kv` |
| `rotary.py` | 127 | `RotaryEmbedding` / `get_rope` / `apply_rotary_emb`，含 YaRN 的 `_yarn_inv_freq` 与 `_yarn_attention_factor`；模块级 `_CACHE` |
| `loader.py` | 97 | safetensors → 模块的 `load_model`、`swap_w4`（把 `nn.Linear` 换成 `W4Linear`） |
| `context.py` | 37 | 前向侧线程局部上下文：`is_prefill` / `slot_mapping` / `block_tables` / `context_lens` / `cu_seqlens` / `verify_m` |

**`state/`（248 行）序列与块**

| 文件 | 行数 | 职责 |
|---|---|---|
| `sequence.py` | 86 | `Sequence` / `SequenceStatus`：一条请求的 token、缓存槽位、下一块偏移 |
| `block_manager.py` | 161 | `Block` / `BlockManager`：空闲队列、`can_allocate`/`allocate`/`can_append`/`may_append`、verify 的 `can_verify`/`reserve_verify`/`release_tail`、前缀缓存的链式 `compute_hash` 与 `hash_blocks` |

**`runtime/` 根（171 行）**：`config.py`（63，唯一的 knobs dataclass）、
`sampling_params.py`（11）、`__init__.py`（16）、
`attention_store.py`（81，⚠ 全仓零引用，处置见 TODO 18）。

### L2 `qslab/models/`（121 行）

`w4linear.py`（119）：`W4Linear` 是运行时与离线路径共用的 int4 线性层——只持 `qfp`/`scale` 两张
打包张量，`forward` 委托给后端；`swap_w4_linears` 按名字批量替换 `nn.Linear`。

### L1 `qslab/quant/`（902 行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `packfmt.py` | 88 | `qslab_w4_v1` 容器读写：`pack_w4` / `unpack_w4` / `save_qslab_w4` / 加载 |
| `w4.py` | 143 | 量化算法：`rtn_quantize_weight`、`clip_search_quantize`、`awq_find_scales`、`quantize_weight` |
| `w4_backends.py` | 187 | 后端选择：`W4V1Backend` / `W4MarlinBackend` / `W4AutoBackend`，注册名 `w4.v1` / `w4.marlin` / `w4.auto`，入口 `get_backend`；`usable()` 决定回退 |
| `backends.py` | 45 | `QUANT_BACKENDS = Registry("quant backend")` 与 `QuantBackend` 协议 |
| `calibrate.py` | 54 | `collect_activations`：AWQ 用的激活统计 |
| `quantize.py` | 105 | 离线量化主程序（`main`），产出 `qslab_w4_v1` 目录 |
| `cache/kv_cache.py` | 267 | KV4/KV8/FP16 三档缓存的 **CPU 记账参考实现**（`BaseKVCache`/`FP16KVCache`/`KV8Cache`/`KV4Cache`）。生产代码不 import 它；它守的是 `tests/kernels/test_gpu_kernels.py` 的三条往返误差界，线上 triton 内核以它对齐目标 |

### L0 `qslab/kernels/`（204 行 py + 381 行 cu）

| 文件 | 行数 | 职责 |
|---|---|---|
| `ops.py` | 42 | `_get_mod` 带 memo 的 JIT 入口 + `w4a16_gemm`；`sources=` 只列 `csrc/w4a16_gemm_api.cu` |
| `marlin_ext.py` | 35 | `get_marlin()` 加载 vendor 的 Marlin 扩展 |
| `marlin_backend.py` | 126 | `pack_v1_to_marlin`（重排 + 转容器）、`marlin_gemm`、`_get_perms` |
| `csrc/w4a16_gemm_api.cu` | 86 | **线上唯一被编译的自研内核**（uint4 访存的 W4A16 GEMV/GEMM 入口） |
| `csrc/w4a16_gemm.cu` | 111 | 从未进过 build 的旧 LUT 内核（见 TODO 25） |
| `csrc/test_w4a16_gemm.cu` | 184 | 内核自测用，`build_kernels.sh` 单编它 |

### 参照 `qslab/reference/`（162 行）

`config.py`（25）的 `ModelConfig` 是 HF config 的形状镜像；`loader.py`（133）提供
`load_reference_model`（fp16 HF 模型）、`load_model_config`、`read_safetensors_state_dict`、
`load_w4_model`、`_wrap_input_scale`、`args_model_dir`。后四个目前零调用，见 TODO 21。

## 4. 一次 `generate` 的通路

1. `qslab/api/llm.py:102` `LLM.generate` → `to_runtime()` 换成 runtime 的 `SamplingParams`；
   想要 greedy 只能给 `GREEDY = 1e-6`（`api/llm.py:22`），因为 `runtime/sampling_params.py:11` 构造时
   直接 assert 掉 `temperature == 0`——**runtime 没有 greedy 分支**，靠 softmax 饱和近似。
2. `runtime/engine/llm_engine.py:58` `add_request` 建 `Sequence`，`block_manager.allocate` 圈块。
3. `llm_engine.py:71` `step()` 循环：`scheduler.schedule()` 返回本步的 prefill 批与 decode 批，
   `sign` 约定是正数=prefill token 数、负数=decode 承诺数。
4. 开了投机则在 `llm_engine.py:95` `_propose` 里由 `ngram.py` 或 `draft.py` 出提案；
   提案数受 `spec_num_drafts`（γ）约束，自适应 γ 在 `scheduler.py:219 _adapt_gamma`。
5. `execute/model_runner.py:459 run()`：命中图则回放，否则 eager。批形状决定走
   `prepare_prefill` / `prepare_decode` / `prepare_verify`（`:188/:251/:273`）；
   verify 的每序列 M = γ+1 行，摊平后 `bs*M` 行。
6. 前向里 `model/loader.py swap_w4` 换出的 `models/w4linear.py W4Linear.forward` 向
   `quant/w4_backends.py` 要后端：`w4.auto` 先试 Marlin，不可用则回 `w4.v1`。
7. `model/attention.py:85 forward`：先 `store_kv_quant` 把本步 K/V 量化入池（顺序很重要，decode 内核
   要读到自己这一行），prefill 走 flash-attn varlen、前缀命中时先 `materialize_kv` 把池里的段接在前面，
   decode/verify 走 `paged_attention_decode`。rope 由 `rotary.py get_rope` 给，可带 YaRN。
8. 采样与接受：普通 decode 行用 `execute/sampler.py`；投机行过
   `execute/model_runner.py:347 rejection_verify`（Leviathan 概率比接受，one-hot 提案传
   `draft_probs=None` 时退化成按 `q(x)` 接受）。
9. `scheduler.postprocess` / `postprocess_verify` 提交 token、按需 `preempt`；
   `block_manager.may_append` 扩块。序列结束即从池里回收。

## 5. 两种 4-bit 的格式约定

**W4 容器 `qslab_w4_v1`**（`quant/packfmt.py:1-9`）：safetensors，每个权重三条键
`<name>.qfp` / `.scale` / `.zero`。

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `qfp` | `[O, I/8]` | uint32 | 8 个 int4 打进一个字，低位在前 |
| `scale` | `[O, I/g]` | fp16 | 分组尺度，`g` 默认 128 |
| `zero` | `[O, I/g]` | fp16 | 对称量化下恒为 0 |

**KV4 池**（`runtime/execute/model_runner.py:120-149`）：每层四个缓冲，槽位数 = 块数 × `kvcache_block_size`。

| 缓冲 | 形状 | dtype | 含义 |
|---|---|---|---|
| `kq` | `[slots, H, D/8]` | uint32 | K 的 int4，打包同 W4 的低位在前 |
| `ks` | `[H, D]` | fp16 | K 的**每层一张**常数尺度表 |
| `vq` | `[slots, H, D/8]` | uint32 | V 的 int4 |
| `vs` | `[slots, H, D/v_group]` | fp16 | V 的逐 token 分组尺度（`v_group` 默认 64） |

两条必须记住的差别：

- **nibble 约定不一致**：权重侧是二补数 `(q ^ 8) - 8`（`packfmt.py:35`、`csrc/w4a16_gemm_api.cu:13`），
  KV 侧写入是 offset-binary `nib = (qi + 8) & 0xF`（`paged_decode.py:72`）。读错不是掉精度，是每元素恒偏
  `±8·scale`。见 TODO 39。
- **池的计费口径与真实缓冲不同**：`model_runner.py:123` 按 `slot_bytes = 2 * head_dim` 折算，
  而真实缓冲是 `kq`+`vq` 各 `D/8`×4 B 加 `vs`，`ks` 是每层常数表。见 TODO 11。

## 6. 配置面与入口

`runtime/config.py` 的 `Config` 是唯一 knobs，门面未知关键字直透。按用途分组：

| 组 | 字段 |
|---|---|
| 形状与上限 | `max_model_len` `max_num_seqs` `max_num_batched_tokens` `kvcache_block_size` |
| 显存 | `gpu_memory_utilization` `num_kvcache_blocks` `enforce_eager` |
| W4 | `w4`（打包目录）`w4_backend`（`w4.v1`/`w4.marlin`/`w4.auto`） |
| KV4 | `v_group` `smooth_kv`（SmoothAttention 标定文件） |
| 长上下文 | `rope_scaling`（只用于延伸：`factor > 1`，会改写 `hf_config` 的天花板） |
| 投机 | `spec_method`（`ngram`/`lookahead`/`draft`）`spec_num_drafts`（γ）`spec_ngram_size` `spec_lookahead_span` `spec_adaptive_gamma` `draft_model` `draft_w4` `draft_w4_backend` `draft_gpu_memory_utilization` |
| 其它 | `tensor_parallel_size` `compile` `compile_mode`（接在 `model_runner.py:37-44`）`eos` `hf_config` |

三个入口：`LLM`（Python 门面）、`python -m qslab.api.cli`、`python -m qslab.api.server`（OpenAI 兼容，
`--device` 会设 `CUDA_VISIBLE_DEVICES`）。离线路径两条：`python -m qslab.quant.quantize` 产 W4 目录，
`scripts/build_calib.py` 与 `scripts/build_smooth_kv.py` 产标定文件（注意 `quantize.py` docstring 里写的
`python -m quantizer.quantize` 是错的，见 TODO 31）。

## 7. 架构前提（改动时必须守住）

- **单卡、单进程**。`tensor_parallel_size` 保留字段位但前向里没有 TP；不要假设多设备语义。
- **块大小必须 128 的整数倍**（`config.py` 的 `__post_init__` assert），因为 int4 池按 `D/8` 打包。
- **CUDA Graph 分桶**：`graph_bs = [1,2,4,8] + range(16, max_bs+1, 16)`，`max_bs = min(max_num_seqs, 512)`
  （`model_runner.py:471/479`）；行数 > 512 时 `:312` 与 `:433` 都直接落 eager。verify 图族按 `bs` 分桶、
  每桶 M = γ+1 行，所以可达性受 512 行闸门约束（TODO 41）。
- **先写池再读池**：decode 内核读的 `context_lens` 含当前 token 自己的 KV（nano-vllm 的顺序）。
- **V 必须先 `.contiguous()`**：`v` 是融合 QKV 切片的视图，行 stride 是整条 QKV 宽度，
  存储内核按 `idx*H*D` 索引（`attention.py:106` 附近）——flash-attn 自己处理 stride，所以只有入池会错。
- **前缀命中必须物化**：prefill 声明的 `cu_seqlens_k` 比传入的 k/v 长时，池里的段要先 `materialize_kv` 接上，
  否则 flash-attn 会读到错位行。
- **greedy 只有近似**：见第 4 节第 1 步。任何"关掉温度就是 argmax"的假设都要走 `api/llm.py` 的 `GREEDY`。
- **重构的零影响判据**：全量 pytest 日志逐字节 md5 与基线一致（见第 8 节）。

## 8. 测试布局

`tests/` 按包分层，与第 2 节的层一一对应；`conftest.py` 在 `tests/` 根，它负责往 `sys.path` 里塞仓库根、
提供 `has_cuda` / `has_small_model` 夹具，以及**在带 `e2e` 标记的模块之间做显存回收**
（两个 1.7B 引擎不能共存，靠这个 autouse 夹具串行）。

```
tests/api/       2 文件 / 13 项   门面 6 + 服务面 7（服务面不占 GPU）
tests/kernels/   1 文件 /  6 项   内核数值 + KV4 往返误差界
tests/quant/     1 文件 /  9 项   纯 CPU 快档：打包、注册表、W4Linear 常驻
tests/runtime/engine/ 10 文件 / 51 项
tests/runtime/model/   3 文件 / 22 项
tests/conftest.py
合计 17 文件 / 101 项
```

常用命令：`pytest tests/ -m "not e2e"` 快门档（秒级，不建引擎）；`pytest tests/` 全量（单卡串行）。
`pytest.ini` 的 `addopts = -q --strict-markers` 意味着日志里没有 "N passed" 行——判定看 `EXIT` 与逐字节 md5。
当前基线：`7cf42015eba3350a8b0768711f3bfb35`（101 个点 / 174 B 日志）。