# TODO — 证据链缺环与收尾遗留（2026-09-18 整理；同日 M10 复核后更新；09-19 补测后再次更新；09-19 M11 闭合遗留 7 并新账 10/11）

## 缺环（证据链不完整处）

1. ~~**draft 投机 bench 脚本未入库**（06-spec-draft/）：
   results/m9_spec_draft.txt、m9_gamma_sweep_draft.txt、m9_draft_fused.txt 无对应脚本。
   建议：按 benchmarks/05-spec-ngram/bench_spec_ngram.py 结构补 bench_spec_draft.py
   （spec_method="draft" + GAMMA 扫描 + 融合开关对照）。~~ — **09-19 入库并复现**：
   `benchmarks/06-spec-draft/bench_spec_draft.py`（γ 扫描 + `ADAPTIVE` 开关 + 逐 verify
   步轮询 `scheduler.proposal_gamma` 打印窗口轨迹）+ 底稿 `results/m10_draft_sweep.txt`。
   **copy 列六格在 ±0.6% 内复现** m9_draft_fused.txt（最大 184.1→183.4）；natural 列
   抖动 ±6–9%（γ=3 98.0→103.9、γ=4 97.5→89.0），是 4-bit 贪心的混沌轨迹而非代码差异，
   因此 natural 只读成"γ≥3 在 ≈1.0× 噪声带里"。
2. ~~**m9_prefix_cache.txt 只有 1.7B 数字**，8B 的 558.3→63.0ms 只在 notes/PROGRESS
   里（当时跑过）~~ — **09-19 重跑落盘**：该文件追加了两段带命令的原始输出，8B
   cold 555.9ms → hit 62.3ms（**8.92×**），与 M9 口录的 558.3/63.0 差 0.4%/1.1%。
   顺带记下一条此前没写清的事实：bench 无 W4 开关，8B 走 fp16 权重，需
   `UTIL=0.9` 才放得下（`peak=16.91GiB`，KV 池只剩 3.69GiB）。
3. ~~m2_niah.json 只覆盖 @1K/@2K，32K 无独立底稿~~ — **M10 复核后不成立**：
   `results/m2_niah.json` 本身就是 ctx=32768 的落盘（fp16 与 kv4 各 3 深度 ×3 次，
   overall_recall 均 1.0），产出脚本 benchmarks/01-decode-baseline/bench_niah.py 在库。
   真正偏弱的只是探针密度（9 次/配置），不算缺环。

### M10 带来的新缺环

4. ~~**温度投机只有正确性证据，没有吞吐证据**：接受律已换成 Leviathan 概率比
   （tests/test_spec_acceptance.py 实测分布无偏：N=20000、T=0.9 时 TV<0.06），
   但概率比路径每步要多一份 `[bs·(γ+1), V]` float32 提案分布和一次 softmax，
   **加速比从未测过**。建议：06-spec-draft 下补 T>0 的 n-gram/draft 吞吐对照。~~
   — **09-19 已实测**：`benchmarks/06-spec-draft/bench_spec_temperature.py` +
   `results/m10_spec_temperature.txt`。**税不在那份 float32 分布**——lookup 类提案是
   one-hot，走不到分配（peak 5.66→5.67 GB）；税在**接受率塌**：ngram copy 3.06→2.68×
   （+12.4%）、lookahead copy 3.54→1.31×（+63.0%）、lookahead natural acc 0.62→0.00
   （tok/step 退化成 1.00，等于没投机）、draft copy 1.10→1.05×（+4.4%）。
   随机提示家族上 T>0 的 3.3×/3.8× 经探针（`/tmp` 一次性脚本，未入库）证实是**轨迹
   假象**（该温度下输出周期为 1 的重复 token），已从结论剔除。**口径教训：不要用带
   随机字的 prompt 评投机。**
5. ~~**动态 γ 的阈值未在新 runtime 上调过参**：WINDOW=3、"平均接受 ≤1.0 折半 /
   ≥γ-0.1 回升"是旧引擎 M6 直移的先验；γ 上限被 `spec_gamma` 锁死（verify 图族
   M=γ+1 固定），所以"高接受率时窗口能不能再放大"这件事当前无从验证。~~
   — **09-19 已实测并据此改判**：`results/m10_draft_sweep.txt`（8B natural，γ 上限 4，
   每策略 4 次重复）——只裁不涨 **1.00–1.18×（mean 1.07×）**、固定窗口 0.91×、把旧引擎
   的可达写法 `avg >= proposal_gamma - 0.1` 接回来则 0.90–1.11×（mean 0.98×，4 次里 3 次
   跌破 1.0）。**回升被自己的复测量证伪**：HEAD 原分支写的是 `avg >= self.gamma - 0.1`，
   阈值对着上限而 `avg` 最大只到当前窗口，折半后**按定义不可达**——旧代码事实上已是棘轮，
   只是挂着死分支和一段不成立的"滞回防抖"说辞；本次把死分支**删除而非修复**，
   `tests/test_spec_acceptance.py` 改为断言单向棘轮。机制（未证但自洽）：接受是**最长
   前缀**判据，一轮落在 2/2 对"第 3、4 条会不会被接受"零信息，回升读的是噪声；要回收窗口
   需要**逐位置**接受率，当前 `avg` 只是逐序列标量。
   两处更正/遗留：γ 上限的**运行时字段**是 `config.spec_num_drafts`（`spec_gamma` 只是 L4
   门面的参数名，旧文两者混用）；
   ~~WINDOW=3 与折半系数没扫过~~ — **09-19（M10 续）已扫完**：
   `benchmarks/06-spec-draft/bench_spec_adaptive_tune.py` + `results/m10_adaptive_tune.txt`
   （先穷举 780 条接受流证明扫参台与出厂代码判定 0 处不同，再扫）。**代码一个数字都不改**，
   三条结论：①**折半落点是唯一决定性轴**，任何"继续往下切"净亏 20%（级联 0.79、γ−1 阶梯
   0.81、一步到 1 的 0.83），因为配置 γ=4 的步价 21.5ms vs 平解码 10.2ms ⇒ 盈亏平衡要
   tok/step ≥2.11，窗口停在 2 有 2.2、切到 1 只剩 1.7；②**WINDOW 不是反应速度旋钮而是
   误触发滤波器**（γ//2 落点幂等 ⇒ 窗口只推迟那一次切割），natural 上 W=1..6 是一条平台，
   但 copy 上 W=1/2 分别因第 4/14 步误触发掉 27%/24%，阈值放宽到 1.5 掉 19%，而 W≥3
   各 10 次抽取从不触发 ⇒ 出厂的 3 恰在悬崖边沿；③**本项最大的更正**：裁窗口**不省时间**
   （`DraftProposer` 用配置 γ 循环，`proposal_gamma` 从未流进提案器；步价只有 21.5/16.3ms
   两档，与运行时窗口无关），所以上面"0.91× 救到 1.07×"那句因果解释作废——n=5 重测为
   钉上限 0.98×[0.90,1.05] vs 棘轮 1.06×[1.01,1.14]（+8%，Welch t=2.47），且切割在固定
   轨迹上只可能变差，那 8% 归属轨迹分叉、**未证**；手工把 γ 配成 2 的对照显示 natural 打平
   （1.08）但 copy 掉 13%（1.61），这才是"γ=4 + 只裁不涨"真正的理由。
   "窗口能否越过配置值"结构上仍无从验证（verify 图族按固定 M=γ+1 捕获）。
   真正的优化空间变成一条新任务：**让早停真的省钱**（提案器按 `cap` 提前结束串行解码，
   或为每个 M 各录一族 verify 图）；收益未测，外推上限 natural ~1.4×。
6. ~~**Marlin 路径双份 int4 常驻只有一句口算**（M10 修门面测试时暴露）：
   `w4.auto` 名义上按 M dispatch，实际 `CROSSOVER_M=0`，v1 pack 全程闲置，
   而 `w4.marlin` 另存一份 Marlin 排布的 `_B/_s`。口算给的"8B ≈ 1.9 GB"从未
   进测。~~ — **09-19 已实测**：`benchmarks/02-w4-quant/bench_w4_residency.py` +
   `results/m10_w4_residency.txt`，1.7B 死重量 0.677 GB（+15% KV 池）、8B
   **3.335 GB**（1.52→4.85 GB，可换出约 2.1 万 → 6.6 万 token 总容量）。口算
   低了近一倍，已更正。**注意**：`w4.auto` 与 `w4.marlin` 逐列相同——省不掉
   这一份是 Marlin 后端本身的性质，与 dispatch 策略无关。同日该改动**已实现并
   复测**（后端 `uses_v1_pack` + `W4Linear` 按需注册 buffer）：8B 常驻
   6.674→3.339 GB（0.52×→0.26× fp16），"只有走 `w4.v1` 才省得下"随之作废；
   而 "+220% / 4.85 GB" 的换算也错了——腾出的字节进的是分配预算
   （2.94→6.14 GiB），池子只兑现其中 ~51%，实测 1.52→**3.16 GB（+108%）**。

> **缺环 1–6 于 2026-09-19 全部闭合**（底稿：`results/m10_w4_residency.txt`、
> `m10_draft_sweep.txt`、`m10_spec_temperature.txt`，以及追加 8B 段的
> `results/m9_prefix_cache.txt`）。收尾遗留 7 于同日 M11 闭合（`results/m11_yarn_niah.txt`）；
> 剩下的 9/10/11 是功能与新账，不是证据链问题。

## 收尾遗留（功能，非证据链）

7. ~~**128K YaRN demo 未跑**（M4 起遗留）~~ — **M11（09-19）已跑，且前提先修了一遍**：
   runtime **没有 YaRN**——`rope_scaling` 在 `Qwen3Attention` 收到后被直接丢弃，配置写了
   不生效也不报错。本轮先实现（`rotary.py`/`config.py`/`qwen3.py`，与 transformers 4.57.6
   **逐位相同**，12 条锁在 `tests/test_rotary_yarn.py`），再按 32K/64K/128K 分档跑 NIAH：
   底稿 `results/m11_yarn_niah.txt`、脚本 `benchmarks/08-long-context/bench_niah_yarn.py`。
   读数：128K 端到端跑通，decode **13.3 → 7.1 → 3.7 tok/s**、冷 prefill **6.7 → 17.8 → 53.2 s**、
   peak 13.38 GiB；召回 3/8（其余三档合计 21/24，Fisher p≈0.012），但**漏的形态是"数字前缀
   全对、尾部丢"**（`92142` vs 921426）⇒ 塌点在转写不在检索；**YaRN 对召回的贡献未证**
   （32K 两列同代码同 needle，换 haystack 相位就能整格移动）。
8. ~~models/Qwen3-0.6B-qslab-w4-awq2~~ 已删（A 档清理 11795fd）；tests/ 4 个辅助脚本已移 scripts/archive/m-verification/。
9. **让动态 γ 的"早停"真的省钱**（M10 续扫参后新账，见缺环⑤第③条）：当前裁窗口只丢弃
   已付费的提案，控制器没有成本通道。两条路——①`DraftProposer` 按 `cap` 提前结束它的 γ 次
   串行解码（本来就只取前若干条，天然可截）；②为每个 M 各录一族 verify 图，让窗口能真的换图。
   ①便宜且不需要重录图族。**收益未测**：只有外推上限（窗口 2 的 tok/step 2.25 若配上 γ=2
   那档 16.3ms 的步价，natural 会到 ~1.4×；21.5→16.3 的 5.2ms 里"多 2 次提案前向"与
   "更宽 verify"各占多少没拆开）。落地后"回升"才重新值得测（需要逐位置接受率，见缺环⑤）。
10. **长上下文评分口径**（M11 新账，底稿 `results/m11_yarn_niah.txt` §5③）：`value in answer`
    把"从 12 万 token 外把 6 位数字捞回来、末位抄错"和"完全没找到"记成同一个 miss，
    于是每次底稿都要人工重读一遍答案才敢下结论。需要逐位/编辑距离口径（或至少分开统计
    "前缀对长度"）。本轮**没改脚本**——改了就没有可比的两轮；128K 档 38% 这个数因此是
    严格匹配下界，不是"检索成功率"。
11. **KV 池超订**（M11 读代码发现，**M12 已量化**——四份 `results/m12_batch_*.txt` 的
    `[kvschema]` 行）：`allocate_kv_cache` 按 `slot_bytes = 2 * head_dim`
    （`model_runner.py:123-124`）计费，而真实缓冲是 kq(int4, head_dim/2 B) +
    vq(int4, head_dim/2 B) + vs(fp16, 2·head_dim/v_group B) = **132 B/head/layer**
    （`:139-147`）⇒ 1.7B 计费 56.0 / 实配 28.9 KiB·token⁻¹，8B 计费 72.0 / 实配 37.1，
    **两个模型都是 1.94× 超订**（原来那句"多出的一半归属未逐字节拆开验证"就此结掉）。
    后果：8B 的 3.79 GiB 预算里只有约 2.05 GiB 真被分配，池 431 块 = 55,168 tokens，
    按 `max_model_len=4096` 只容 **13 条**满长序列（M11 那档 128K 把 `max_num_seqs`
    压到 2 就是这个原因）。修掉计费口径 ≈ 并发/上下文白翻一倍。
12. **投机与并发抢同一份算力，当前没有联动策略**（M12 新账）：1.7B 上
    n-gram / 关投机的比值从 bs=1 的 1.90× 单调降到 bs=8 的 1.10×、bs=16 的 **0.90×**、
    bs=32 的 **0.77×** ⇒ **高并发开投机是净负收益**；8B 在同区间仍为正（bs=16 为 1.50×），
    说明阈值存在但从未标定。需要一条"按当前 batch 关/降投机"的策略——现在
    `spec_num_drafts` 与 `max_num_seqs` 互不知情。底稿 `results/m12_batch_1.7b*.txt`。
13. **1.7B 每序列接受率随并发掉、8B 不掉，机制未证**（M12 新账）：
    `tok/step ÷ bs` 为 1.7B 2.16 → 1.50 → 1.53 → 1.28 → 1.18 → 1.19，8B 恒在 3.54。
    `NGramProposer.propose` 只读单条序列的 `token_ids`、无任何跨序列状态（`ngram.py:51`），
    所以**不是**提议器串味。剩余候选（权重精度 fp16/W4、token 预算 256/192、8B 是否
    本就吃满 γ+1）**未逐一定责**；要定责需要逐位置接受率（与缺环⑤同一把尺子）。
14. **服务面断连不取消**（M12 新账，底稿 `results/m12_service_smoke.txt` 已现形）：
    `head -6` 提前断管 ⇒ 服务端在 `server.py:256` 抛
    `ClientConnectionResetError: Cannot write to closing transport`。两件事要做：
    ①流式写出包一层，断连只丢该 Job 的增量、不脏服务端日志；②给 scheduler 加
    "按 seq_id 摘除"的接口，让断连真的回收算力（现在会继续生成到 token 预算用完）。
15. **服务面只有 demo 级防护**：无鉴权、`active_requests` 只观测不设闸、无请求超时、
    无队列上限 ⇒ 打满只会排队 + 抢占，不会拒绝。对外暴露之前必须补。
    另有一条同族边界（本轮在 CPU 上直接驱动 Scheduler 复现）：**单条 prompt 比整个 KV 池还长**时，
    `can_allocate` 回 -1 会让 prefill 批为空，接着掉进 decode 分支的
    `scheduler.py:119 assert scheduled_seqs`，抛的是裸 `AssertionError`，
    不是"这条请求放不下"的可读报错。
16. **vLLM 对照还缺两格**（M12）：① 8B W4A16↔W4A16——vLLM 加载不了自研
    `qslab_w4_v1` pack，要走 vLLM 侧量化（bitsandbytes / compressed-tensors），这要在
    共享的 `vllm` env 里装新依赖，**属改环境，先问再做**；② int4 KV 在 vLLM 0.11 没有
    对应物（它最好到 fp8 KV），所以 KV4 的容量收益只能做成"同 util 下池 135,936 vs
    75,808 tokens"这种记账式对照，做不了等精度吞吐对照。
17. **主模型不给标定文件不会报错，只会静默退化**（写教程 04 章时实测）：
    `allocate_kv_cache` 把 K 的尺度表开成 `torch.zeros(H, D)`（`model_runner.py:142-143`），
    只有 `load_smoothing(path)` 会往里写（`:160-180`，`:168` 处 `if not path: return`）；
    而 store 与 decode 都按 `ks.dim()==2` 走 STATIC_K 分支（`paged_decode.py:183/203`），
    **没有** per-token 的动态回退。于是 `smooth_kv=None`（`api/cli.py:37`、
    `api/server.py:291` 的默认值）时解码读出的 K 恒为 0。
    实测：`LLM("models/Qwen3-1.7B", enforce_eager=True)` 不传 smooth_kv，对
    " The capital of France is" 贪心 24 token —— 第 1 个 token 与有标定时完全相同
    （prefill 不读池），第 2 个起塌成 `is is is…`；第 0 层 `max|k_s| = 0.0`。
    `spec_method="draft"` 那条路有 `assert os.path.exists(calib)`（`llm_engine.py:37`）兜底，
    主模型这条没有。要么补同一条 assert，要么把 `k_s` 初值改成 1.0 并显式记成"未平滑"档。
18. **一个全仓零引用的文件**：`qslab/runtime/attention_store.py`（81 行，M8-s1 vendor 时带进来）。
    它是 `docs/archive/design-m8.md` 追加一节里被实测**推翻**的那版 per-token K 方案 kernel，
    真正在用的是 `runtime/model/paged_decode.py` 里的同名 kernel（`runtime/model/attention.py:26` 从后者 import）。
    `grep -rn attention_store . --exclude-dir=.git` 只剩 `docs/archive/design-m8.md:96` 一处提及。
    留着容易把人带偏（本教程初稿就误把它当成了写入路径）。删之前先确认没有外部脚本按路径引用它。

19. **`.gitignore:2` 的 `models/` 连 `qslab/models/` 一起吞了**（2026-09-20 实测）：往
    `qslab/models/` 放一个未跟踪文件 `__probe__.py`，`git check-ignore -v` 报
    `.gitignore:2:models/`；`git add qslab/models/__init__.py` 也会先吐一句
    "下列路径…被忽略"（已跟踪文件的改动仍被暂存，靠的是索引豁免，所以一直没暴露）。
    后果：在 `qslab/models/` 下新建模块时 `git add .` 会静默漏文件。修法：改成 `/models/`。
20. **文档在指向已经删掉的路径**（刀 1+2+4 + 旧文档清理之后的新账，2026-09-20 实测计数）：
    live 文本里还有 24 行引用 `notes/M*.md`、`docs/00–04-*.md`、`docs/design-m*.md`、
    `docs/archive/*.md`（按文件：README.md 2、PROGRESS.md 6、TODO.md 3、docs/ARCHITECTURE.md 11、
    docs/STUDY-PATH.md 2），另外本 TODO 里 `bench_throughput.py:48-67` 那条脚本也已随旧引擎进了
    存档分支。修法：改写成 `git show b87a5f5:<path>` 的取回形式，或者整句删掉；
    untracked 的 docs/study/* 教程另算（不入库，单独改）。

21. **`qslab/reference/loader.py` 里四个零调用符号，但不能直接删**（搬 reference/ 包时实测，2026-09-20）：
    `load_w4_model`（:64）、`_wrap_input_scale`（:113-120）、`read_safetensors_state_dict`（:48）、
    `args_model_dir`（:123）在 `qslab/` `tests/` `scripts/` `benchmarks/` 里**没有代码调用者**
    （`qslab/models/w4linear.py:81` 只是 docstring 提了一句）。但 `args_model_dir:129` 是全仓
    **唯一**读 `models/*.origin.txt` 的地方，而 `.origin.txt` 既被 gitignore、也没有任何代码写它
    （挂起项：不许碰）。所以删这四个符号 = 让 `.origin.txt` 彻底没有读者。
    修法二选一：确认后连同 `.origin.txt` 约定一起废弃（archive/legacy-engine 上另有一份代码），
    或者把它当「packed ckpt 装回 fp16 参照模型」的离线对齐工具留着并补第一条测试。

22. **runtime 分层后，教程正文里的旧路径没跟着改（故意没改）**（2026-09-20 实测计数）：
    `docs/study/*.md` 里还有 **79 行 / 11 个文件** 写着搬家前的 `runtime/<模块>.py`
    （01 章 8、README 5、02 章 1、03 章 1、04 章 3、05 章 4、06 章 7、07 章 5、08 章 16、
    09 章 26、10 章 3）。代码与 tracked 文档里的路径本次已批量改对（96 处），教程正文
    不动的原因是章里夹着「逐字实测输出」块，机器替换会连证据一起改。修法：逐处人工分辨
    「引用」还是「输出」，前者改路径，后者保留并加一句「该文件已于 2026-09-20 移至
    runtime/<子包>/」。第一条已定位：`docs/study/01-一次请求的一生.md:328`
    的 `from qslab.runtime.llm_engine import LLMEngine`。
    同类：`docs/STUDY-PATH.md`（挂起项，本次一字未动）里 9 行、
    `qslab/runtime/attention_store.py:7` 的 docstring 里 1 行（挂起项，md5 自证未动）。

## M10 已完成（2026-09-18，功能遗留全部落地）

- [x] **L4 门面接新 runtime**：`api/llm.py` 的 `LLM` 改为包 `LLMEngine`，
      `api/cli.py` 重写为 runtime 词表（--w4/--smooth-kv/--spec/--draft/--stats-only）；
      6 个门面测试（tests/test_api_facade.py）+ CLI 两条实跑冒烟（greedy ngram、T=0.8 lookahead）
- [x] **温度采样投机**：`ModelRunner.rejection_verify` 概率比接受 + 拒绝后
      `norm(max(0,q-p))` 重采样；draft proposer 下发提案分布（float32，防 fp16 下溢把
      拒绝变成必受）；greedy 路径逐行不变（15 个既有 spec/draft e2e 测试原样绿）
- [x] **lookahead 迁移**：`runtime/engine/ngram.py::LookaheadProposer`（持久逐序列索引、
      链式延伸、首次出现优先），`--spec lookahead` 可达
- [x] **动态 γ**：`Scheduler._adapt_gamma`——只裁剪提案条数、不动 verify 图的 M，
      故无需新图族；仅对按条付费的 draft 生效（ngram/lookahead 提案免费）
- [x] **8B+draft e2e 固化**：`tests/test_8b_acceptance.py::test_8b_draft_spec_acceptance`
      （0.62/0.9 显存分配，两 int4 池共卡；实跑 33.41s 通过）
- [x] 旧 `qslab/engine/spec/`（lookahead/dynamic）**逻辑已迁移**，旧包保留为冻结参照

## M10 补测（2026-09-19，缺环 1/4/5/6 与 2 的 8B 段全部落盘）

- [x] **缺环①**：`bench_spec_draft.py` 入库，copy 列 ±0.6% 复现 M9 融合表；顺带确认
      natural 列不可复现是 4-bit 贪心混沌，不是代码差异（底稿已写明）
- [x] **缺环④**：`bench_spec_temperature.py` + 底稿，把"无损"这件事的吞吐价签贴上；
      并纠正了一个此前的口头归因（税不在 float32 提案分布，在接受率）
- [x] **缺环⑤**：三种窗口策略 × 4 次重复对照，**回升分支证伪并回退**，
      `tests/test_spec_acceptance.py` 改断言单向棘轮；
      GPU e2e `tests/test_draft_spec.py::test_adaptive_gamma_shrinks_when_proposals_keep_failing`
      重跑通过
- [x] **缺环⑥ / ②**：常驻字节 + 8B 前缀缓存（见上文两条）
- [ ] **新认账的产品限制（非缺环）**：T>0 时免费提案器（ngram/lookahead）全线劣于同配置
      greedy，只有 draft 勉强守在 1.0× 附近 → 温度解码建议 γ≤2 或直接关投机。已写入
      `benchmarks/06-spec-draft/README.md`，README/docs 同步
- [x] **释放 v1 pack 死重量**（原"未实现的优化"）：后端新增 `uses_v1_pack`，
      Marlin repack 后丢 `qfp/scale`，`W4Linear` 只在需要时注册这两个 buffer，
      `W4Linear.to()` 的设备推导改成不依赖"一定有 buffer"。实测 8B 常驻
      6.674→3.339 GB、KV 池 1.52→3.16 GB（+108%，不是估的 +220%）；1.7B graph
      decode 同运行内 `w4.auto`≡`w4.marlin`（≤0.6%），省显存没付速度。PPL 未复测
      （`_B/_s` 逐字节不变）。全量测试 84/84（新增 3 条 CPU 常驻契约测试）。
      底稿同一文件新增复测 + 点测两节

## M10 续（2026-09-19，动态 γ 两个先验扫参）

- [x] **缺环⑤的剩余半项**：`bench_spec_adaptive_tune.py` 入库 + `results/m10_adaptive_tune.txt`
      （GPU 约 31 分钟，只用 1、3 号空卡）。三组轴扫完：**折半落点**是唯一决定性轴
      （级联/阶梯/一步到 1 一律 0.79–0.83×，机制=盈亏平衡 tok/step ≥2.11）；
      **WINDOW** 在 natural 上是平台、在 copy 上是悬崖（W=1/2 误触发掉 27%/24%，
      W≥3 各 10 次抽取从不触发）⇒ 它的身份是"一次不可逆动作的误触发滤波器"；
      **阈值**保持 1.0。**决定：`_adapt_gamma` 一个数字都不改**，扫参换来的是优化空间⑨。
- [x] **推翻并撤销 M10 为这条控制器写的因果解释**：裁窗口不省时间（提案器循环用配置 γ，
      `proposal_gamma` 从未流进它；步价只有 21.5/16.3ms 两档，实测）。n=5 重测
      钉上限 0.98× vs 棘轮 1.06×，原"0.91 救到 1.07"降级为弱信号 + 轨迹分叉（未证）。
      补上原设计缺的对照：手工 γ=2 → natural 打平（1.08×）但 copy 掉 13%。
- [x] 顺带量化混沌带（natural 投机重复 ±8%、平解码与 copy 投机可复现）⇒ 立了条口径：
      natural 上 <5% 的格间差不读成结论。`tests/test_spec_acceptance.py` 加一条 CPU 锁
      （窗口是误触发滤波器：单轮 1/4 不触发、连续三轮 ≤1.0 才触发）。全量测试 85/85。

## M11（2026-09-19，128K YaRN：先实现，再分档跑）

- [x] **实现 runtime 的 YaRN**：`rotary.py` 加 `_yarn_inv_freq`/`_yarn_attention_factor`
      （从 transformers 4.57.6 转写：`find_correction_dim`、truncate→floor/ceil、clamp、
      `linear_ramp_factor` 的 `high += 0.001` 退化处理、按 `1-ramp` 混合、
      `attention_factor = 0.1·ln(f)+1.0` 同乘 cos 与 sin），`qwen3.py` 停止丢弃 `rope_scaling`，
      `config.py` 新增 `rope_scaling` 字段并在 `__post_init__` 里把
      `hf_config.max_position_embeddings` 改写成 `original × factor` **再**过原有 clamp
      （天花板只有一处定义）。`lru_cache(1)` → 按 rope 设置建键的 dict（36 层共享）。
- [x] **正确性锁**：`tests/test_rotary_yarn.py` 12 条 —— 与 HF 的 `inv_freq` 差 ≤1e-9、
      cos/sin 缓存 `rtol=0, atol=0`（**逐位相同**，位置 0…131071）、attention factor 严格相等；
      另两条防静默：native 路径的缓存与实现前**逐位相同**且 `attention_scaling == 1.0`
      （这次重写不会改动其他里程碑的数字）、非 yarn 的 `rope_type` 抛 `NotImplementedError`
      而不是被忽略（一次没生效的长度外推会跑完、数字好看、什么都不能证明）。
      全量测试 **97/97**。
- [x] **分档 NIAH 落盘**：`benchmarks/08-long-context/bench_niah_yarn.py` +
      `results/m11_yarn_niah.txt`（8B W4A16+KV4，util 0.8，GPU3）。速度曲线是本轮最硬的
      产品数字（见上第 7 条）；召回读数只允许说"128K 更差 + 差在转写"，
      YaRN 的贡献**未证**（无 128K 基线：不缩放是越界外推而非基线）。
- [x] **两条 bench 口径教训**（已写进 `benchmarks/08-long-context/README.md`）：
      ① 同列 trials 的 haystack 必须**相位错开**，否则前缀缓存会把 prefill 从 6.7s 打到 1.0s
      —— 第一轮整张 prefill 表因此作废并重跑；② 答案片段必须宽到看得见数字（≥60 字符）+
      单独抽 `got=`，否则 24 字符套话让所有 miss 长得一样、无法归因。
- [ ] 新账 10（评分口径）与 11（池计费）见上。

## M12（2026-09-19，并发轴首次定价 + 官方 vLLM 对照 + 最小服务面）

- [x] **补上从未有过的 bs>1 读数**：`benchmarks/09-batch-throughput/bench_batch.py`。
      仪器先自证：`SELF=1` 用**出厂 prompt + 出厂 token 数**复测 bs=1，偏差 >5% 就
      `exit(1)`、bs>1 一格都不报——四格 Δ 0.4% / 3.2% / 0.7% / 0.1% 全过。扫描用的
      prompt 每条前置一个不同随机 token id 打掉前缀缓存（`compute_hash` 链式哈希，
      首块不同即整条不命中）。读数：1.7B 关投机 bs1→32 **24.43×**（3572.8 tok/s 解码
      窗口）、8B bs1→16 **12.99×**；**投机在高并发翻负**（1.7B bs16 0.90×、bs32 0.77×）
      ⇒ 新账 12/13。另记一条口径边界：扫描的 bs=1 投机读数（1.7B 278.1）远低于自检的
      出厂口径（441.3），除首 token 外 prompt 完全相同 ⇒ 差异只能来自生成序列本身，
      **机制未证**，所以两张表不可互比。
- [x] **TODO 11 从"读代码发现"变成实测**：`[kvschema]` 行直接印出计费 vs 实配，两个
      模型都是 **1.94× 超订**；顺带量出 8B 池 = 431 块 = 55,168 tokens = **13 条**
      4096-token 序列，这就是当前的并发天花板。
- [x] **官方 vLLM 同 prompt 对照**：`benchmarks/10-vllm-compare/bench_vllm.py` +
      `results/m12_vllm_1.7b.txt`（独立 env：vllm 0.11.0 / torch 2.8.0+cu128）。墙钟口径
      下本引擎 = vLLM 的 **0.67–0.73×**，且比例不随并发变；两条批量加速曲线重合
      （24.43× vs 23.91×）⇒ 形态上是每步固定开销而非调度缺陷（**未做逐层 profile**）。
      量化侧的收益是记账式的：同 util 下 KV 池 **1.79×** 容量，按真实字节 **3.87×**。
      两条环境坑固化进脚本：`VLLM_USE_FLASHINFER_SAMPLER=0`（该 env 无 nvcc，flashinfer
      的 JIT 采样器在 engine 初始化即炸）；父进程失败会留孤儿 `VLLM::EngineCore` 占 8 GiB，
      下一次启动以 `Free memory on device (7.11/23.53 GiB)` 失败。
- [x] **最小服务面**：`qslab/api/server.py`（aiohttp，optional extra `.[serve]`，核心包
      不引入 web 依赖）+ `tests/test_api_server.py` 7 条 + `benchmarks/11-service-surface/`。
      设计要点：**一个泵线程独占引擎**（不按请求开线程），批仍然在 scheduler 里成；为此
      `add_request` 改为返回 `Sequence`（本轮唯一引擎侧改动），泵每步 diff
      `completion_token_ids` 才能逐 token 流式；`LLM.engine` 属性公开给服务面。
      读数：8 并发 384 tokens / 0.444 s = **865 tok/s**，8 条同批完成；对照进程内 bs=8
      扫描（1015 tok/s）差 14.8%——**这 14.8% 不能整笔记在 HTTP 上**（prompt 长度、curl
      进程启动、util 0.9 vs 0.5、有无 smooth calib 四个混杂都在里面），所以口径只到
      "同数量级、批确实成了"。
- [x] **测试先抓出一个真 bug**：`body.get("max_tokens") or DEFAULT` 把 `max_tokens: 0`
      静默吃成 128 ⇒ 改成显式判 `None`。这条值得单独记：**校验测试抓到的是人眼漏过的
      静默默认值**，不是"多写几条测试"。
- [x] 采集底稿的两条流程教训（已写进 `benchmarks/11-service-surface/README.md`）：
      ① `pkill -f qslab.api.server` 会匹配到发起命令自己的命令行，把 ssh 会话一起打死
      （第一版底稿因此只剩空头，重做过一次）；收尾必须按记录的 PID 精确 kill。
      ② 用 `sleep` 撑长的远程命令会掐断 MCP 传输 ⇒ 改成"后台起 + 短命令轮询"。
- [ ] 新账 12–16（投机-并发联动、接受率机制、断连取消、服务面防护、vLLM 空格子）见上。

## 教程第三批新账（06 章实测，2026-09-20）

来源：`docs/study/tools/ch06-pack-repack.py`（卡片 A，CPU）与
`docs/study/tools/ch06-kernels-real.py`（卡片 B，空卡 0）。正文见 `docs/study/06-从HF权重到int4.md`。
以下都是**可修问题**，与教程批次分开攒 commit。

- [ ] **`kernels/marlin_ext.py:16-35 get_marlin()` 没有模块级缓存**，而
      `kernels/marlin_backend.py:120` 每次 gemm 都调它 ⇒ 每次调用都走一次
      `torch.utils.cpp_extension.load()`（不重编，但重读源文件算哈希、比对 ninja）。
      实测：单次 187–255 µs（两次跑分别 197/255），在 2 MiB 与 6 MiB 两层、M=1/5/16 上
      都是 ~227–247 µs 的**平台**（对字节数不敏感 ⇒ 不是带宽）。
      后果只在 eager 下看得见：1.7B W4 整模型 `w4.auto` **9.1–9.3 tok/s**，
      只在测试脚本里把扩展句柄缓存掉就 **24.3–24.6**，`w4.v1` **27.5–27.8**。
      修法照抄 `kernels/ops.py:10-12` 的 `_mod` memo。**注意这会让 graph 与 eager
      两个口径的 Marlin 数字同时变化，改完要重跑 `results/` 里所有 eager 底稿。**
- [ ] **v1 容器的 `pack∘unpack` 不是不动点**：同一份 pack 用 `unpack_w4`→`pack_w4`
      重打，word 相同率 rtn 100.00%、rtn_clip 56.99%、线上 AWQ ckpt 53.78%
      （scale 相对差 1.15e-01 / 1.33e-01）。机理是 `packfmt.py:30-31`、
      `w4.py:29-32` 在 fp16 里算 amax，而 `clip_search_quantize`（`w4.py:62-64`）与
      AWQ 走 `w4.py:140` 在 fp32 里算、存盘才降 fp16。任何"格式转换/再量化/迁移
      group_size"的脚本都会静默改权重。修法：容器里记打包算术精度，或让
      `pack_w4` 与离线路径共用同一套 fp32 累加。
- [ ] **`save_qslab_w4` 无条件写 `zero`**（`packfmt.py:66`），对称量化下它恒为 0，
      却占磁盘 2.94%（1.7B 0.021 GB、8B 0.101 GB）。运行时 `runtime/model/loader.py:86-87`
      只读 `.qfp/.scale`，所以这笔只在磁盘账上。修法：`symmetric=True` 时不写该键，
      读侧对缺失键补零（要过 `tests/test_unit_cpu.py` 的格式断言）。
- [ ] **`quant/quantize.py` 的 docstring 命令抄不得**：写的是
      `python -m quantizer.quantize`，仓库里没有 `quantizer` 包（真实模块是
      `qslab.quant.quantize`）。纯文档 bug，但照抄必然失败。
- [ ] **B4 的 CPU 税对不上单层账，原因未查**：整模型省 66.6–69.5 ms/token ÷ 196 层
      = 每层 340–355 µs，而卡片 B2 单层量到的是 227–247 µs，多出的 ~100 µs/层
      没有归因（**不编解释**）。要查得逐层 profile。
- [ ] **`CROSSOVER_M=0`（`quant/w4_backends.py:131`）之后 `w4.v1` 在主模型上是纯
      fallback**：196/196 层 Marlin 可用（1.7B 与 8B 各形状都过 `usable`，卡片 A5）。
      v1 内核仍被 draft 路径用着（`runtime/config.py:37 draft_w4_backend="w4.v1"`）。
      待决：07 章量完 v1 GEMV 的访存形状后，再决定它是留作教学样本还是删。

---

## 第七章（kernel 层）现场实测新增

来源：`docs/study/tools/ch07-kernels.py`（卡片 C0–C8，空卡 0，run 2）。
正文见 `docs/study/07-kernel层.md`。以下都是**可修问题**，与教程批次分开攒 commit。

- [ ] **`csrc/w4a16_gemm.cu`(111) 是从没被编译的死代码，而两处文档称它现役**。
      三条独立证据（卡片 C1）：`kernels/ops.py:22/30` 的 `sources=` 只有
      `w4a16_gemm_api.cu`；已编译扩展的 `build.ninja` 里出现的 `.cu` 集合是
      `['w4a16_gemm_api.cu']`；`scripts/build_kernels.sh:28` 只编
      `test_w4a16_gemm.cu`。**测错了的是** `docs/ARCHITECTURE.md:177`、`:221` 和
      `docs/STUDY-PATH.md:47`。修法：文档改指向 `api.cu`，再决定这个文件删还是留作教学样本。
- [ ] **那份死 LUT 内核实测数值就是错的**：`w4a16_gemm.cu:52` 的
      `__shared__ float lut[8][16]` 按 **warp** 索引，而 `:57/:60/:63` 的
      `cur_group/gidx/s` 是 **lane** 私有的→写入互相覆盖。g=128 时一个 warp 的一次
      迭代跨 8.0 个 group。现场编译（43 s，产物只在 /tmp/ch07_build）后与线上 api 版对数字：
      三个形状的 rel 分别是 9.883e-01 / 9.858e-01 / 9.873e-01（即完全不一致）。
      机理定位在源码里，但**未做**“把 lut 改成每-lane 就对”的反证。
- [ ] **`uint4` 快路径没有对齐闸门**：`w4a16_gemm_api.cu:34` 要求行基址 16 B 对齐（即
      `K%32==0`），但 `ops.py:35-42` 和 `w4_backends.py:49-50` 的 `usable()` 只查 dtype
      与 group 整除。实测（C8）：`K=104 g=8` 时 `usable()=True`，行基址 52 B，一调用就抛
      `CUDA error: misaligned address`，而且**错误会粘在 CUDA context 上**，后续探针全部失败。
      修法：`usable()` 里加 `in_features % 32 == 0`，非对齐形状走标量分支或直接拒绝。
- [ ] **`scripts/build_kernels.sh:6` 的路径错**：`cd "$(dirname $0)/../kernels/csrc"` 在本仓库的
      结构下不存在（真实路径是 `qslab/kernels/csrc`），而且即使修好也只产出测试文件的
      独立可执行文件，不是线上扩展（线上走 `ops.py` 的 `torch.utils.cpp_extension.load()`）。
- [ ] **仓库里跟踪着一份编译产物**：`qslab/kernels/csrc/test_w4a16_gemm`（ELF，54776 B，
      sha256 88b28fafebd7）。修法：删 + 进 .gitignore（**删除前要先出清单等确认**）。
- [ ] **`paged_decode.py:106` 的表宽静默截断**：`tl.minimum(cdiv(ctx,BLOCK_N), MAX_BLOCKS)`
      不报警。实测（C5）：`context_len=4096` 而表宽只够 8 块时 `max|Δout|=2.3857`，无异常；
      与“只看前 1024 个键”的稠密参考比只差 0.0016 → 确认就是截断，不是精度损失。
      修法：host 侧 `cdiv(context_len, BLOCK_N) > MAX_BLOCKS` 时 raise。
- [ ] **`paged_decode.py:162` 的 `context_len=0` → NaN**：`acc/l_i` 除零，C6 实测
      `out.isnan().mean()=1.000`。修法：入口 assert，或 `l_i == 0` 时返回零向量。
- [ ] **`paged_decode.py` 里两处死东西**：`:40 _MIN_SCALE` 定义后未被引用；`:169` 的
      `k_scale` 形参在 paged decode 路径里没用（`attention.py:140-144` 只在 `dim()==2` 时传）。
- [ ] **`grid=(N_Q, rows)` 在 bs=1 decode 下只有 16 个 program**（114 个 SM 的 14%），
      实测（C4）KV 池达成带宽 13.1/15.5/16.6 GB/s = 可达带宽(940) 的 1.4/1.7/1.8%，
      ctx=1024/4096/16384。可能的修法是 head 方向 split-K + 归约，但**这条推论未在第七章实测**。
- [ ] **两套 nibble 约定共存**（C3）：权重读二补数（`packfmt.py:35` 截 4 bit、
      `api.cu:13` 的 `(q^8)-8`），新 KV 池写 offset-binary（`paged_decode.py:72` 的
      `(qi+8)&0xF`），而旧 Triton 栈 `kv4_paged_attention.py` 又是二补数。读错不是精度损失：
      换约定读同一份权重，每元素恒偏 ±8·scale（|w| 中位数才 0.0144）。待决：旧栈谁在用、能不能删。
- [ ] **回应第六章最后一条待决**：v1 GEMV 的访存形状已量（C2）：`grid.y=M` 各行各读一份
      权重，M=1→16 时带宽从 56.2 到 268.2 GB/s（N=2048 K=2048），耗时 38.5→129.0 µs。
      结论：`CROSSOVER_M=0` 之后 v1 确实只能做 fallback / 教学样本，不要再接回主路径。

## 第八章（投机解码）现场实测新增

- [ ] **verify 图族有 26/36 张结构上不可能被 replay**：桶表由 `max_num_seqs` 决定
      （`runtime/execute/model_runner.py:471/479`），可达性由 M 决定——同一个函数里
      `:312` 的 `rows > 512` 闸门排在 `:317` 的选图之前。实测（ch08 D1，
      `max_num_seqs=512`、γ=4、M=5）：可达 `bs ≤ 102` ⇒ 10 桶可达 / 26 桶死重
      （112…512）；γ=1/2/8/16 分别 16/22/29/31 张。D2 在真实负载（并发 1/2/3/5 条）
      下只碰过桶 [1, 2, 4, 8]。修法（把闸门换成 `512 // M`）**在挂起清单里，本轮不动**；
      先要的是让这件事可测（下一条）。
- [ ] **图桶选择与可达性没有任何测试**：`tests/` 里没有一条检查 `graphs_verify`
      的键集是否可能被 replay。修法：CPU-only 的性质测试，对 γ∈{1,2,4,8,16}
      断言"可达桶数 = |{bs ∈ graph_bs : bs*(γ+1) ≤ 512}|"。
- [ ] **`ModelRunner.exit` 不释放第二套图族**：`model_runner.py:53-61` 只有
      `del self.graphs, self.graph_pool`，`graphs_verify` 与 `graph_vars_verify`
      （γ=4、max_bs=512 时 `input_ids` 单项就是 2560 行 int64）留在原地。
      修法：同一处补 `del`，未开投机/eager 时这两个属性不存在，需先判 `hasattr`。
- [ ] **同进程建第二个 runtime 会分不到 KV，失败时只有裸 `AssertionError`**：
      `allocate_kv_cache` 的预算 = `total*util − used_by_others − peak`，而 `peak`
      是**整个进程**的高水位（`model_runner.py:112/125`）。实测（ch08 D7）：
      target util=0.35 之后 draft util=0.25 ⇒ `23.53*0.25 − 1.80 − 5.97 = −1.88 GiB`
      → `:131` assert。⇒ `draft_gpu_memory_utilization` 的真义是"整机总量的分档"
      （必须 > (peak+others)/total ≈ 0.33），与字段名给人的"分给 draft 多少"相反。
      修法：assert 换成把这三个数和"peak 是进程级"一起打出来的异常；字段改名或补注释。
- [ ] **裁提案窗口没有成本通道（旧论断已被 D6 证伪，但缺回归守住）**：`_propose`
      只做 `cap = min(gamma, proposal_gamma)` 的事后截断（`llm_engine.py:111-115`），
      提案器仍按配置的 γ 跑完。实测（ch08 D6）：三档窗口 verify 步中位
      7.35/7.34/7.33 ms（极差 0.01 ms < 本进程空 op 地板 13.3 µs），提交 47 token、
      `mean_len` 1.81 三档完全相同，只有提案数 25→23→22。修法：把 cap 传进提案器
      让它早停（draft 侧就是 `runtime/engine/draft.py:292` 的 γ 次循环），并给"同价"补测试。
- [ ] **接受规则缺两条覆盖**：(1) TV 随采样规模的收敛曲线（实测 0.1258@N=500 →
      0.0045@N=200000，`tests/test_spec_acceptance.py:51` 只守 N=20000 + 阈值 0.06
      这一档）；(2) `1e-3` 闸门的**上沿**（现测 9.9e-4 透传 / 1.1e-3 进比值规则，
      测试只覆盖了贪心透传那一侧）。


## 第九章（长上下文）现场实测新增（2026-09-19，ch09 卡片）

- [ ] **不给标定文件会静默毁掉 decode 的 K**：`Config.smooth_kv` 默认 `None` ⇒
      `load_smoothing` 在 `model_runner.py:168-169` 直接 return ⇒ 每层 `ks` 保持 `torch.zeros`
      （`:142`）⇒ store 端除以 0 后 nibble 全饱和、decode 端乘 0 ⇒ **K ≡ 0 ⇒ 注意力退化为对前缀
      均匀平均**。全程无告警、无断言。实测：`docs/study/tools/ch09-run.txt` E8（内核往返
      `max|K̂|=0.0000`、K 相对误差 1.0000、饱和比例 1.000）与 E9（引擎内清零 28 层 `k_s` ⇒
      4K NIAH 只吐一个 `7`；恢复 ⇒ 完整命中）。修法：`allocate_kv_cache` 之后检查 `ks` 是否全 0，
      是则打一条明确 warning（或要求显式 `allow_uncalibrated_kv4=True`）。
- [ ] **`_MIN_SCALE` 是死常量、动态 per-token 尺度分支不可达**：`paged_decode.py:40` 定义了
      `_MIN_SCALE`，`:68-70/:138-140` 有"静态表缺失时退回动态尺度"的分支，但 `attention.py:143`
      的 `STATIC_K = k_cache[1].dim() == 2` 恒为真（池一定会建静态表），所以那条分支永远走不到。
      修法：要么删死码，要么把 `STATIC_K` 变成可关的开关（配合上一条更合理）。
- [ ] **`attention.py:34-36` 的 docstring 与 `:143` 的实际行为相反**（注释说走动态尺度）。
      修法：改注释；顺手把 `:52-72` 的物化前提写清（09 章 §2.9 已抄）。
- [ ] **`rotary._CACHE` 既无上限也不看设备**：一条 CPU 版 cos/sin 表能被 CUDA 引擎复用，
      warmup 直接 `RuntimeError: indices should be either on cpu or on the same device...`
      （逐字在 `docs/study/tools/ch09-crash-first-try.txt`；`rotary.py:106` 用 CUDA 位置的
      positions 索引 CPU 的 `cos_sin_cache`）。触发条件只是"先在任何默认设备上调过 `get_rope`"，
      而 `model_runner.py:29` 构造模型时会把默认设备设成 cuda。修法：缓存键里加 device，
      或命中时校验 `cos_sin_cache.device` 不一致就重建。
- [ ] **`config.py:63` 的 `min()` 静默钳位**：请求 `max_model_len=131072`、天花板 40960 ⇒
      生效 40960，不打日志不报错（ch09 E3 实测）。修法：夹了就 warn 一次，把"请求值/生效值"都打出来。
- [ ] **`config.py:46` 把 `rope_scaling={}` 当成"没配"**：truthy 判断 ⇒ 空 dict 静默走原生路径。
      修法：改成 `is not None` 判定，或空 dict 直接报错。
- [ ] **`rotary.py:113` 的注释写"36 decoder layers"**：对 1.7B（28 层）是过期文案。
- [ ] **KV 池计费口径多计 1.939×**：`model_runner.py:123` 的 `slot_bytes = 2*head_dim = 256 B/头/slot`，
      真实是 132 B（kq 64 + vq 64 + vs 4），`k_s` 是每层一张 8×128 fp16 常数表。实测 ch09 E5：
      charged 9.577 GiB vs actual 4.938 GiB；按真实占用本可买到 3.88× fp16 池，现在只报 2.00×。
      修法：按真实字节计费（并让 `results/m11_yarn_niah.txt §5` 那句"多出的一半是 scales（未证）"
      作废），或在启动日志里明写"这是保守上界，实际能装更多"。

## 第十章（怎么测、怎么读证据）现场实测新增（2026-09-19，ch10 卡片）

- [ ] **引擎不拒绝空 prompt / 被截断的 prompt**：`add_request([])` 照样走完整个 decode
      （ch10 卡片第一版就是这么废的：`ids_of` 切片越界静默返回空列表，7 个"512 token"样本里 6 个是空的，
      于是 M1/M3/M4/M6 全部作废）。修法：`add_request` 或 `Sequence.__init__` 断言
      `len(token_ids) > 0`，并把"prompt + max_tokens ≤ max_model_len"的同一条断言放在那里
      （08/09 章卡片也各自踩过后半条）。
- [ ] **`bench_throughput.py:48-67` 与它自己的 docstring 三处不一致**：签名 `-> float` 但 `:67`
      返回二元组；docstring 说"returns tokens/s"，实际还返回"最后 100 步的秒数和"；
      协议 `docs/03:35` 要"均值"，代码给的是 `1/median`。修法：改 docstring + 明确返回结构
      或加 `--agg median|mean`；并给这个脚本补第一条测试（现在没有任何测试引用它）。
- [ ] **`step()` 的符号约定没有 docstring 也没有测试**（正=prefill token 数、负=decode 承诺数，
      `llm_engine.py:137-140`）：`bench_batch.py:101-107` 和 ch09/ch10 卡片各自复制了这个约定。
      修法：`LLMEngine.step` 补 docstring + 一条 CPU 性质测试。
- [ ] **进度条那个 tok/s 是"最后一步瞬时值"**（`llm_engine.py:133-144` 每步覆盖），容易被当平均
      抄进报告。修法：改成整段累加，或在 postfix 里标 `last`。
- [ ] **ch10 卡片 M6 两行自相矛盾**：同一格里先说比值 1.233"没有证明命中会变快"，后面又断言
      "②那次走的是命中路径"。修法：卡片读一次 `block_manager` 的命中计数再下结论，
      或者两句都删掉只留数字（对应正文 §8 未查第 1 条）。
- [ ] **卡片打印精度会吃掉结论**：`{2*CV:.1f}%` 把 0.04% 打成 `0.0%`。修法：门槛按量级选精度，
      或同时打印原始 CV。同类问题 ch09 也有一处（`每位置 512 B` 那行的小数位）。
- [ ] **冻结校准文件没有长度校验**：`results/frozen/calib_c4_128x2048.pt` 是
      dict{`dataset`,`n`=128,`seq_len`=2048,`token_ids`=len 128 的 list of list（变长）}，
      而 `docs/03:15` 写的是"128 条 × 2048 tokens"。修法：加载处逐条 assert `len(x) == seq_len`。
