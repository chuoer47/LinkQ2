# TODO — 待做清单（2026-09-21 整理：只留没做的事）

**整理说明**：旧台账 490 行删到只剩待办。删掉的内容是缺环 1–6 的闭合过程、M10/M11/M12 的完成记录、
四批教程的现场实测数字——结论与底稿在 `results/` 与 `benchmarks/*/README.md` 里（`docs/ARCHITECTURE.md`
已重写为只讲结构，不再收实验结论），原文用
`git show b19f4dc:TODO.md` 取回。**编号沿用旧台账**（有空洞 = 已完成），`docs/ARCHITECTURE.md` 按号引用；
23 号起是本次新分配的（括号里标出它原来在教程第几章）。

**通用口径**：单卡，跑前 `nvidia-smi` 挑空卡、绝不 kill 别人的进程、汇报写明用了哪张；
重构零影响判据 = 全量 pytest 日志逐字节 md5 与基线相同（**17 文件 / 101 项**，旧基线 md5 `cf47deb5cbc15da631ce3999c050505e`（17/104）作废；**新基线 `7cf42015eba3350a8b0768711f3bfb35`**（101 个点 / 174 B 日志，2026-09-21 在空卡 3 上跑绿，EXIT_PYTEST=0，删 `qslab/sampler.py` 带掉 3 项））；被证伪的旧论断保留一句
"测错了"；查不出原因就写"原因未查"，不编解释；`docs/study/**` 永不入库。

---

## 一、等你定，我不动手

- **16 vLLM 对照还缺两格**：① 8B W4A16↔W4A16 要先把 pack 转成 vLLM 能加载的 4-bit 格式，
  得在共享 `vllm` env 里装新依赖 ⇒ 改环境，先问；② int4 KV 在 vLLM 0.11 无对应物（它最好到 fp8），
  所以 KV4 的容量收益只能做成"同 util 下池 135,936 vs 75,808 tokens"的记账式对照。
- **18 `qslab/runtime/attention_store.py`（81 行，全仓零引用）**：它是 M8-s1 被实测推翻的那版
  per-token K 方案，真身在用 `runtime/model/paged_decode.py` 的同名 kernel。删，还是留作"被推翻的方案"教材？
- **21 `qslab/reference/loader.py` 四个零调用符号**（`load_w4_model:64`、`_wrap_input_scale:113-120`、
  `read_safetensors_state_dict:48`、`args_model_dir:123`）：删它们 = 让 `models/*.origin.txt` 彻底没有读者，
  而该文件被 gitignore 且属挂起项。要么连约定一起废弃，要么当离线对齐工具留下并补第一条测试。
- **24 采样还剩两处没归并 + 一个待决的 greedy 分支**：`qslab/sampler.py` 已删（2026-09-21 用户令；
  删前实测生产零调用，只有 `tests/quant/test_unit_cpu.py` 与 `tests/test_sampler.py` 共 4 条 import 守着，
  而那两条测试只测它自己、从不与线上实现对拍）。**删掉的连带损失**：`top_k` / `top_p` 全仓只有它实现过
  （旧 `sampler.py:20-31`），而 runtime 的 `SamplingParams` 从来没有这两个字段 ⇒ 现在整个仓库不支持
  top-k/top-p，要这个功能得重写进 `runtime/execute/sampler.py`。现存两处：`execute/sampler.py::Sampler`
  （GPU 批量温度采样）、`execute/model_runner.py:347 rejection_verify`（投机接受）。
  前置问题未变：**runtime 要不要真 greedy 分支**——现在 `sampling_params.py:11` 构造即
  `assert self.temperature > 1e-10, "greedy sampling is not permitted"`，门面靠 `api/llm.py:22 GREEDY=1e-6`
  让 softmax 饱和近似 greedy（`model_runner.py:347` 之后的 docstring 专门解释了近似平局为何单走一条路）。
- **25 死内核 `qslab/kernels/csrc/w4a16_gemm.cu`（111 行从未被编译）**：三条独立证据——`ops.py:22/30` 的
  `sources=` 只有 `w4a16_gemm_api.cu`、已编译扩展的 `build.ninja` 里 `.cu` 集合只有 api 版、
  `build_kernels.sh:28` 只编测试文件。**测错了的是** `docs/ARCHITECTURE.md` 旧版（M0 一节 + §3；该文档
  2026-09-21 已整篇重写，旧版用 `git show b19f4dc:docs/ARCHITECTURE.md` 取回）与
  `docs/STUDY-PATH.md:47`。文档已按 api 版改；文件本身删还是留作教学样本？
- **26 磁盘上还剩一份未跟踪的编译产物**：`qslab/kernels/csrc/test_w4a16_gemm`（ELF，54776 B）。旧台账写
  "跟踪在仓库里"，本轮实查 `git ls-files` 已无此项（48c7a2b 就把二进制出过 git）⇒ 只剩本地清理 + 确认
  .gitignore 覆盖到它。**测错了的是旧台账那句。**
- **39 两套 nibble 约定共存（第三套已随旧引擎消失）**：权重读二补数（`packfmt.py:35`、`api.cu:13` 的
  `(q^8)-8`），新 KV 池写 offset-binary（`paged_decode.py:72` 的 `(qi+8)&0xF`）。读错不是精度损失而是每元素
  恒偏 ±8·scale（|w| 中位数才 0.0144）。旧台账里的第三个约定（旧 Triton paged KV 栈）本轮实查已不在树里
  （只剩 `qslab/quant/cache/__pycache__/kv4_paged.cpython-311.pyc`）⇒ "旧栈谁在用、能不能删"这个待决自动消解。
- **40 `w4.v1` 内核留不留**：`CROSSOVER_M=0` 之后它在主模型上是纯 fallback（1.7B/8B 各形状 196/196 层
  Marlin 可用），访存形状已量（`grid.y=M` 各行各读一份权重，M=1→16 带宽 56.2→268.2 GB/s、耗时 38.5→129.0 µs）
  ⇒ 只作教学样本，不要再接回主路径。draft 路径仍在用它（`runtime/config.py:37 draft_w4_backend`）。
- **59 工作树里三个已跟踪文件被删**（`README.md`、`PROGRESS.md`、`docs/autonomous_journal.md`，`git status` 里的 D）：
  照删提交，还是先恢复？（`PROGRESS.md` 与 `notes/` 在本机 `.cache/` 下另有工作副本）

## 二、文档与注释

- **19 `.gitignore:2` 的 `models/` 连 `qslab/models/` 一起吞**（实测 `git check-ignore -v` 命中）：
  在该目录下新建模块时 `git add .` 会静默漏文件，`git add <已跟踪文件>` 还会非零退出中断 `&&` 链。修法：改 `/models/`。
- **20 文档在说已经不存在的东西**（本轮进度）：已改 `docs/ARCHITECTURE.md` 的五处结构段——§3 分层树（补
  `qslab/reference/` 与 runtime 四子包）、两代引擎表（旧引擎已归档）、缺环⑤、§6 文档地图、header 溯源说明。
  **还欠**：`qslab/api/llm.py` 头 docstring 仍称旧引擎"stays reachable as qslab.engine.QslabEngine"
  （b87a5f5 后不成立）；`benchmarks/03-kv4-quality/bench_ppl.py` 称"PPL 由引擎本身算出"，实际第 90 行
  用 `load_reference_model` 打的是 HF 参照模型（引擎侧是 `04-runtime-m8/bench_ppl_runtime.py`）；
  `qslab/api/cli.py` 头 docstring 同样称旧引擎"only reachable as qslab.engine.QslabEngine"（同为 b87a5f5 后不成立）；
  `docs/STUDY-PATH.md` 2 行（挂起）。
- **22 教程正文里 79 行 / 11 个文件**还写着搬家前的 `runtime/<模块>.py`（分层时故意没批量改，章里夹着逐字实测输出）。
  修法：逐处人工分辨"引用"还是"输出"，前者改路径，后者保留并加一句迁移说明。同族：`STUDY-PATH.md` 9 行、
  `attention_store.py:7` 1 行（都在挂起项里）。卡片重跑会打印新路径，与正文旧输出不一致（数值不变）。
- **23 "adapters 是唯一 import transformers 的地方"与代码不符**（本轮实测）：另有 3 处直接 import——
  `runtime/config.py:3`（AutoConfig）、`runtime/model/qwen3.py:25`（Qwen3Config）、
  `runtime/engine/llm_engine.py:5`（AutoTokenizer）。要么改文档口径，要么把这三个收进 adapters。
- **27 `scripts/build_kernels.sh:6` 的路径错**：`cd "$(dirname $0)/../kernels/csrc"` 在本仓库结构下不存在
  （真实是 `qslab/kernels/csrc`），而且即使修好也只产出测试文件的独立可执行文件，不是线上扩展（线上走 `ops.py` 的 JIT）。
- **31 `quant/quantize.py` 的 docstring 命令抄不得**：写的是 `python -m quantizer.quantize`，仓库里没有
  `quantizer` 包（真实模块 `qslab.quant.quantize`）。纯文档 bug，照抄必然失败。
- **37 `runtime/model/attention.py:34-36` 的 docstring 与 `:143` 的实际行为相反**（注释说走动态尺度，实际恒走静态表）。
  顺手把 `:52-72` 的物化前提写清。
- **48 `rotary.py:113` 注释写"36 decoder layers"**：对 1.7B（28 层）是过期文案。

## 三、能直接修的代码问题

**会静默给出错结果的**
- **17 主模型不给标定文件不报错，只静默退化**：`allocate_kv_cache` 把 K 尺度表开成 `torch.zeros(H, D)`
  （`model_runner.py:142-143`），`load_smoothing` 在 `:168` 处 `if not path: return` ⇒ K ≡ 0 ⇒ 注意力退化成
  对前缀均匀平均。实测：1.7B 贪心 24 token 从第 2 个起塌成 `is is is…`、第 0 层 `max|k_s| = 0.0`；
  引擎内清零 28 层 `k_s` ⇒ 4K NIAH 只吐一个 `7`，恢复 ⇒ 完整命中。`spec_method="draft"` 那条路有
  `assert os.path.exists(calib)`（`llm_engine.py:37`）兜底，主模型这条没有。修法：同一条 assert，
  或 `ks` 全 0 时打明确 warning。
- **11 KV 池计费按 `slot_bytes = 2*head_dim` 超订 1.94×**：真实缓冲 132 B/头/层（kq 64 + vq 64 + vs 4，
  `k_s` 是每层一张 8×128 fp16 常数表）而计费 256 B（`model_runner.py:123`）。实测 charged 9.577 vs actual
  4.938 GiB ⇒ 8B 池只装得下 **13 条** 4096-token 序列（M11 那档 128K 把 `max_num_seqs` 压到 2 就是这个原因）。
  改完并发/上下文白翻一倍，但**所有池容量与吞吐底稿要重跑**，`results/m11_yarn_niah.txt §5` 那句
  "多出的一半是 scales（未证）"随之作废。
- **33 uint4 快路径没有对齐闸门**：`api.cu:34` 要求行基址 16 B 对齐（即 `K%32==0`），但 `ops.py:35-42` 与
  `w4_backends.py:49-50` 的 `usable()` 只查 dtype 和 group 整除 ⇒ `K=104 g=8` 时 `usable()=True`，一调用抛
  `misaligned address`，而且错误粘在 CUDA context 上、后续探针全废。修法：`usable()` 加 `in_features % 32 == 0`。
- **34 `paged_decode.py:106` 表宽静默截断**：`tl.minimum(cdiv(ctx,BLOCK_N), MAX_BLOCKS)` 不报警 ⇒
  `context_len=4096` 而表宽只够 8 块时 `max|Δout| = 2.3857` 且无异常（与"只看前 1024 键"的稠密参考只差 0.0016
  ⇒ 确认是截断不是精度）。修法：host 侧 `cdiv(context_len, BLOCK_N) > MAX_BLOCKS` 就 raise。
- **35 `paged_decode.py:162` 的 `context_len=0` → NaN**：`acc/l_i` 除零，实测 `out.isnan().mean() = 1.000`。
  修法：入口 assert，或 `l_i == 0` 时返回零向量。
- **43 同进程建第二个 runtime 必然分不到 KV，且失败形态是裸 `AssertionError`**：预算 =
  `total*util − used_by_others − peak`，而 `peak` 是**整个进程**的高水位（`model_runner.py:112/125/131`）。
  实测 target util=0.35 后 draft util=0.25 ⇒ `23.53*0.25 − 1.80 − 5.97 = −1.88 GiB`。⇒
  `draft_gpu_memory_utilization` 的真义是"整机总量的分档"（必须 > (peak+others)/total ≈ 0.33），与字面相反。
  修法：异常里把这三个数和"peak 是进程级"一起打出来；字段改名或补注释。
- **45 `rotary._CACHE` 既无上限也不看设备**：一条 CPU 版 cos/sin 表能被 CUDA 引擎复用，warmup 直接
  `RuntimeError: indices should be either on cpu or on the same device`（`rotary.py:106` 用 CUDA positions
  索引 CPU 缓存；触发条件只是"先在任何默认设备上调过 `get_rope`"）。修法：缓存键加 device，或命中时校验设备。
- **46 `config.py:63` 的 `min()` 静默钳位**：请求 `max_model_len=131072`、天花板 40960 ⇒ 生效 40960，
  不打日志不报错（ch09 E3 实测）。修法：夹了就 warn 一次，把"请求值/生效值"都打出来。
- **47 `config.py:46` 把 `rope_scaling={}` 当成"没配"**（truthy 判断 ⇒ 空 dict 静默走原生路径）。
  修法：改 `is not None`，或空 dict 直接报错。
- **49 引擎不拒绝空 prompt / 被截断的 prompt**：`add_request([])` 照样走完整个 decode（ch10 卡片第一版
  因此整批作废：切片越界静默返回空列表，7 个"512 token"样本里 6 个是空的）。修法：`Sequence.__init__`
  断言 `len(token_ids) > 0`，并把"prompt + max_tokens ≤ max_model_len"放在同一处。
- **52 冻结校准文件没有长度校验**：`results/frozen/calib_c4_128x2048.pt` 的 `token_ids` 是变长 list of list，
  而协议文档写"128 条 × 2048 tokens"。修法：加载处逐条 assert `len(x) == seq_len`。

**白烧资源**
- **28 `kernels/marlin_ext.py:16-35 get_marlin()` 没有模块级缓存**，而 `marlin_backend.py:120` 每次 gemm
  都调它 ⇒ 每次走一遍 `cpp_extension.load()`（不重编，但重读源文件算哈希、比对 ninja）：单次 187–255 µs，
  在 2 MiB/6 MiB 两层、M=1/5/16 上都是 ~227–247 µs 的**平台**（对字节数不敏感 ⇒ 不是带宽）。
  后果只在 eager 下看得见：1.7B W4 整模型 `w4.auto` **9.1–9.3 tok/s**，只在测试脚本里缓存句柄就
  **24.3–24.6**（`w4.v1` 27.5–27.8）。修法照抄 `ops.py:10-12` 的 `_mod` memo。**注意：graph 与 eager 两个口径的
  Marlin 数字会同时变，改完要重跑 `results/` 里所有 eager 底稿。**
- **41 verify 图族有 26/36 张结构上不可能被 replay**：桶表由 `max_num_seqs` 决定（`model_runner.py:471/479`），
  可达性由 M 决定，而同函数里 `:312` 的 `rows > 512` 闸门排在 `:317` 选图之前 ⇒ `max_num_seqs=512`、γ=4 时
  可达 `bs ≤ 102` = 10 桶可达 / 112…512 全死重；γ=1/2/8/16 分别 16/22/29/31 张。D2 在真实负载（并发 1/2/3/5）
  下只碰过桶 [1,2,4,8]。修法分两步：**先补测试**（CPU-only 性质测试，对 γ 属于 {1,2,4,8,16} 断言
  "可达桶数 = 满足 bs*(γ+1) ≤ 512 的桶数"），**改闸门为 512 除以 M 那一条在挂起清单里，本轮不动**。
- **42 `ModelRunner.exit` 不释放第二套图族**：`:53-61` 只有 `del self.graphs, self.graph_pool`，
  `graphs_verify` 与 `graph_vars_verify` 留在原地（γ=4、max_bs=512 时光 `input_ids` 单项就是 2560 行 int64）。
  修法：同一处补 `del`，未开投机/eager 时这两个属性不存在，需先判 `hasattr`。
- **9 裁提案窗口没有成本通道**（旧论断已被 ch08 D6 证伪，但缺回归守着）：`_propose` 只做
  `cap = min(gamma, proposal_gamma)` 的事后截断（`llm_engine.py:111-115`），提案器仍按配置 γ 跑完 ⇒
  三档窗口 verify 步中位 7.35/7.34/7.33 ms（极差 0.01 ms < 本进程空 op 地板 13.3 µs），提交 47 token、
  `mean_len` 1.81 三档完全相同，只有提案数 25→23→22。修法：把 cap 传进 `engine/draft.py:292` 的循环让它早停，
  并给"同价"补测试。**收益未测**，外推上限 natural ~1.4×（21.5→16.3 ms 里"多 2 次提案前向"与"更宽 verify"
  各占多少没拆开）。落地后"回升"才重新值得测，而那需要逐位置接受率（见 13）。
- **30 `save_qslab_w4` 无条件写 `zero`**（`packfmt.py:66`）：对称量化下它恒为 0 却占磁盘 2.94%
  （1.7B 0.021 GB、8B 0.101 GB），运行时 `runtime/model/loader.py:86-87` 只读 `.qfp/.scale` ⇒ 纯磁盘账。
  修法：`symmetric=True` 不写该键，读侧对缺失键补零（要过 `tests/quant/test_unit_cpu.py` 的格式断言）。

**正确性/死码**
- **29 v1 容器的 `pack∘unpack` 不是不动点**：同一份 pack 重打，word 相同率 rtn 100.00%、rtn_clip 56.99%、
  线上 AWQ ckpt 53.78%（scale 相对差 1.15e-01 / 1.33e-01）。机理：`packfmt.py:30-31`、`w4.py:29-32` 在 fp16
  里算 amax，而 `clip_search_quantize`（`w4.py:62-64`）与 AWQ 走 `w4.py:140` 在 fp32 算、存盘才降 fp16
  ⇒ 任何"格式转换/再量化/改 group_size"的脚本都会静默改权重。修法：容器里记打包算术精度，或让 `pack_w4`
  与离线路径共用同一套 fp32 累加。
- **36 `paged_decode.py` 里两处死东西 + 一条不可达分支**：`:40 _MIN_SCALE` 定义后未被引用；`:169` 的
  `k_scale` 形参在 paged decode 路径没用（`attention.py:140-144` 只在 `dim()==2` 时传）；
  `:68-70/:138-140` 的"静态表缺失就退回动态尺度"分支永远走不到，因为 `attention.py:143` 的
  `STATIC_K = k_cache[1].dim() == 2` 恒真。修法：删死码，或把 `STATIC_K` 做成可关的开关（配 17 更合理）。
- **14 服务面断连不取消**：`head -6` 提前断管 ⇒ 服务端在 `server.py:256` 抛
  `ClientConnectionResetError: Cannot write to closing transport`。两件事：①流式写出包一层，断连只丢该 Job
  的增量、不脏日志；②给 scheduler 加"按 seq_id 摘除"的接口，让断连真的回收算力（现在会继续生成到预算用完）。
- **15 服务面只有 demo 级防护**：无鉴权、`active_requests` 只观测不设闸、无请求超时、无队列上限。
  同族边界（CPU 上直接驱动 Scheduler 复现）：单条 prompt 比整个 KV 池还长时 `can_allocate` 回 -1
  ⇒ prefill 批为空 ⇒ 掉进 `scheduler.py:119 assert scheduled_seqs`，抛裸 `AssertionError` 而不是"放不下"。
- **50 `step()` 的符号约定没有 docstring 也没有测试**（正 = prefill token 数、负 = decode 承诺数，
  `llm_engine.py:137-140`），而 `bench_batch.py:101-107` 和 ch09/ch10 卡片各自复制了这个约定。
  修法：补 docstring + 一条 CPU 性质测试。
- **51 进度条那个 tok/s 是"最后一步瞬时值"**（`llm_engine.py:133-144` 每步覆盖），容易被当平均抄进报告。
  修法：整段累加，或在 postfix 里标 `last`。

## 四、证据缺口与未查（不排期，只登记）

- **10 长上下文评分只有严格匹配口径**：`value in answer` 把"从 12 万 token 外把 6 位数字捞回来、末位抄错"
  和"完全没找到"记成同一个 miss ⇒ 128K 的 38% 是**下界**、不是检索成功率，每份底稿都要人工重读答案才敢归因。
  需要逐位/编辑距离口径（或至少分开统计"前缀对长度"）。
- **12 投机与并发抢同一份算力，没有联动策略**：1.7B 上 n-gram/关投机的比值从 bs=1 的 1.90× 单调降到
  bs=8 的 1.10×、bs=16 的 **0.90×**、bs=32 的 **0.77×** ⇒ 高并发开投机是净负收益；8B 同区间仍为正
  （bs=16 为 1.50×）⇒ 阈值存在但从未标定。`spec_num_drafts` 与 `max_num_seqs` 互不知情。
  底稿 `results/m12_batch_1.7b*.txt`。
- **13 1.7B 每序列接受率随并发掉、8B 不掉，机制未证**：`tok/step` 除以 bs 为 1.7B 2.16→1.50→1.53→1.28→1.18→1.19，
  8B 恒在 3.54。`NGramProposer.propose` 只读单条序列的 `token_ids`（`ngram.py:26`，批量入口 `:46`）⇒ **不是**串味。
  剩余候选（权重精度、token 预算 256/192、8B 是否本就吃满 γ+1）未逐一定责；要定责需要逐位置接受率。
- **44 接受规则缺两条覆盖**：① TV 随采样规模的收敛曲线（实测 0.1258@N=500 → 0.0045@N=200000，
  `tests/runtime/engine/test_spec_acceptance.py:51` 只守 N=20000 + 阈值 0.06 这一档）；② `1e-3` 闸门的**上沿**
  （现测 9.9e-4 透传 / 1.1e-3 进比值规则，测试只覆盖贪心透传那一侧）。
- **32 B4 的 CPU 税对不上单层账，原因未查**：整模型省 66.6–69.5 ms/token 除以 196 层 = 每层 340–355 µs，
  而单层量到的是 227–247 µs，多出的 ~100 µs/层没有归因（不编解释）。要查得逐层 profile。
- **38 bs=1 decode 的 attention kernel 只用 14% 的 SM**：`grid=(N_Q, rows)` 只有 16 个 program（114 个 SM），
  KV 池达成带宽 13.1/15.5/16.6 GB/s = 可达带宽 940 的 1.4/1.7/1.8%（ctx 1024/4096/16384）。
  可能的修法是 head 方向 split-K + 归约，**这条推论未实测**。
- **54 死 LUT 内核实测就是错的，但缺反证**：`w4a16_gemm.cu:52` 的 `__shared__ float lut[8][16]` 按 **warp**
  索引，而 `:57/:60/:63` 的 `cur_group/gidx/s` 是 **lane** 私有 ⇒ 写入互相覆盖（g=128 时一次迭代跨 8.0 个 group）。
  现场编译（43 s，产物只在 /tmp）后与线上 api 版对数字：三个形状的 rel = 9.883e-01 / 9.858e-01 / 9.873e-01
  （完全不一致）。机理在源码里定位了，但**未做**"把 lut 改成每-lane 就对"的反证。
- **55 128K 的墙在转写不在检索，可 YaRN 的贡献未证**：128K 档 NIAH 从 8/8 级掉到 3/8（Fisher p 约 0.012），
  漏的形态是"数字前缀全对、尾部丢"（`92142` vs `921426`）。但 32K 两列同代码同 needle、换 haystack 相位就整格移动
  （64K 两轮 8/8→7/8）⇒ 这轮不能给 YaRN 记账（没有 128K 基线：不缩放 rope 是越界外推）。
  另记结构性事实：长上下文 PPL 在这套 runtime 上**结构上不可得**（逐位置 logits 在 131072 × 151936 是 40 GB 量级）。
- **56 vs vLLM 的差距没定位**：墙钟口径 0.67–0.73×，且比例不随并发变、两条批量加速曲线重合
  （24.43× vs 23.91×）⇒ 形态上像每步固定开销而非调度缺陷，但**没做逐层 profile**。
- **57 扫描 bs=1 的投机读数（1.7B 278.1）远低于自检的出厂口径（441.3）**：除首 token 外 prompt 完全相同
  ⇒ 差异只能来自生成序列本身，**未证**。两张表因此不可互比。
- **58 T>0 时免费提案器全线劣于同配置 greedy，机制未逐条查**：ngram copy 3.06→2.68×、lookahead copy
  3.54→1.31×、lookahead natural acc 0.62→0.00（tok/step 退化成 1.00）、draft copy 1.10→1.05×。
  税不在 float32 提案分布（lookup 提案是 one-hot，走不到分配，peak 5.66→5.67 GB）。已写进
  `benchmarks/06-spec-draft/README.md` 的口径建议：温度解码 γ≤2 或直接关投机。
  一条口径教训：**不要用带随机字的 prompt 评投机**（那些 3.3×/3.8× 经探针证实是该温度下输出退化成
  周期为 1 的重复 token，任何提案器都能命中，已从结论剔除）。
- **53 教程卡片的两个读数毛病**（`docs/study/tools/`，不入库）：① 打印精度会吃掉结论——`{2*CV:.1f}%`
  把 0.04% 打成 `0.0%`（ch09 同类：`每位置 512 B` 那行的小数位）；② ch10 卡片 M6 两行自相矛盾（同一格先说
  比值 1.233"没有证明命中会变快"，后又断言"②那次走的是命中路径"）⇒ 读一次 `block_manager` 命中计数再下结论，
  或两句都删只留数字。

- **60 `tests/` 分层后残留的旧平铺路径引用**（2026-09-21 本轮实测）：本轮把 17 个平铺测试文件按包 `git mv` 进
  `tests/{api,kernels,quant,runtime/engine,runtime/model}/`（文件名与 marker 一律未改），已同步改写 18 处跟踪文件里
  的路径指针（分层 commit 内 12 处：benchmarks 四个 README 6 + 两个脚本 2、qslab 两处 docstring 2、测试 docstring 2；
  文档批 6 处：TODO 3 + ARCHITECTURE 3）。
  **还剩三类没改**：① `results/` 底稿 4 处（`m10_draft_sweep.txt:14`、`m10_spec_temperature.txt:6`、
  `m10_w4_residency.txt:73`、`m11_yarn_niah.txt:9`）——底稿是当时的实测记录，**不改**；② `docs/study/**` 96 处
  （教程不入库，正文夹着逐字输出，改前要逐处分辨引用还是输出）；③ `docs/STUDY-PATH.md`（挂起项，未动）。
