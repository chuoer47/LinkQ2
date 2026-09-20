# 结构梳理自主推进台账（2026-09-20，goal 模式）

目标：A 方案建 `qslab/reference/`（收 `models/loader.py` + `ModelConfig`，runtime 侧不改名）；
然后给 `qslab/runtime/` 做分层（原本 18 个平铺 .py）。

## 阶段 0｜动手前的台账（全部实测）

- 起点 HEAD：`bdd93dc`（前面依次是 `1c944a7` 结构梳理刀 1+2+4、`b87a5f5` 旧引擎剥离）。
- 全量测试基线：`pytest tests/` 收集 **17 个测试文件 / 104 项**，日志逐字节 md5
  `cf47deb5cbc15da631ce3999c050505e`（= `/tmp/strip/base.log`）。零影响判据用它比对。
- `qslab/runtime/` 动手前：18 个平铺模块，行数 11〜534；`qslab/` 包根剩 4 个 .py。
- 挂起项本轮全部避开：`docs/STUDY-PATH.md`、`qslab/runtime/attention_store.py`、
  `models/*.origin.txt`、`512//M` verify 桶、vLLM 依赖。

## 阶段 1｜commit `3a70346`：qslab/reference/ 落地

搬（`git mv`，函数体逐行未改）`qslab/models/loader.py` → `qslab/reference/loader.py`；
把 `ModelConfig` 从 `qslab/config.py` 拆到 `qslab/reference/config.py`，然后删掉 `qslab/config.py`。
6 处 import 跟进：`qslab/quant/quantize.py`、`scripts/build_smooth_kv.py`、
`benchmarks/03-kv4-quality/bench_ppl.py`、`bench_ppl_kv.py`、`tests/test_e2e_engine.py`、
untracked 的 `docs/study/tools/ch10-measure.py`。`qslab/models/` 至此只剩 `w4linear.py`。

决定不混进本次搬迁的：`reference/loader.py` 里 `load_w4_model` / `_wrap_input_scale` /
`read_safetensors_state_dict` / `args_model_dir` 四个零调用符号。新实测到的理由是
`args_model_dir`(:129) 是全仓唯一读 `models/*.origin.txt` 的代码，而那些文件被 gitignore
且没有任何代码写它 —— 删它等于让一个手写约定彻底没有读者。记 TODO 21 等定夺。

## 阶段 2｜runtime 分层

分层不是按想象画的，是按 AST 抽出的 import 边定 rank：

| 层 | 模块（搬前） | 依据（实测边，只列关键的） |
|---|---|---|
| `engine/` | llm_engine, scheduler, ngram, draft | llm_engine→draft/scheduler；scheduler→block_manager+ngram；draft→model_runner+Scheduler(:100 真依赖) |
| `execute/` | model_runner, sampler | model_runner→config/context/loader/qwen3/sampler/sequence |
| `model/` | qwen3, attention, paged_decode, primitives, rotary, context, loader | qwen3→attention+primitives+rotary；attention→context+paged_decode；loader→primitives |
| `state/` | sequence, block_manager | block_manager→sequence→sampling_params |
| 根（保持路径不变） | config, sampling_params | 被 16 / 30 处 import，且是跨层契约 |
| 根（挂起） | attention_store | 全仓零引用，TODO 18 未定，故不给它分层 |

`ngram` 与 `draft` 并进 `engine/` 而不是单开 `speculate/`，理由是实测到 `draft.py:100` 真构造
`Scheduler`、`scheduler.py` 又 import `ngram` —— 单开会造出包级环。

执行：15 个 `git mv` + 4 个新 `__init__.py` + 96 处路径改写（46 个文件），脚本
`/tmp/mv_rt2.py`（先 dry-run 再看清单）。批量改写只动 `qslab.**.py` / `tests` / `scripts` /
`benchmarks` / `adapters` / `docs/ARCHITECTURE.md` / `README.md` / `PROGRESS.md` /
`TODO.md`；`results/**`（原始底稿）、`.cache/`、`docs/study/*.md`、`docs/STUDY-PATH.md`、
`attention_store.py` 全部排除在外。

分层后的包级 import 边（`/tmp/layerchk.py` 输出逐字）：

```
OK   qslab.runtime.engine     -> qslab.runtime            (3 -> 0)
OK   qslab.runtime.engine     -> qslab.runtime.execute    (3 -> 2)
OK   qslab.runtime.engine     -> qslab.runtime.model      (3 -> 1)
OK   qslab.runtime.engine     -> qslab.runtime.state      (3 -> 1)
OK   qslab.runtime.execute    -> qslab.runtime            (2 -> 0)
OK   qslab.runtime.execute    -> qslab.runtime.model      (2 -> 1)
OK   qslab.runtime.execute    -> qslab.runtime.state      (2 -> 1)
OK   qslab.runtime.state      -> qslab.runtime            (1 -> 0)
向上边数: 0
包级环: []
模块数分布: root 3 / engine 4 / execute 2 / model 7 / state 2
```

runtime 对下的唯一出口仍是 `model/loader.py` → `qslab.models.w4linear` + `qslab.quant.packfmt`；
`model/` 不再有任何人 import 它之外的 runtime 模块。

## 阶段 3｜复验（2026-09-20，全部实查，卡 3）

- 全量 pytest：104 项、`EXIT_PYTEST=0`，日志 174 B 与基线**逐字节相同**
  （md5 `cf47deb5cbc15da631ce3999c050505e`）。收集仍 17 个测试文件。
- 入口冲烟：`python -m qslab.api.cli generate --help`、`python -m qslab.quant.quantize --help`、
  `python scripts/build_smooth_kv.py --help` 全部正常退出（后两个直接依赖搬过的
  `qslab.reference.loader`）。
- 端到端：拿教程卡片 01-A 那条命令原样跑（W4 + KV4 + ngram，`--device cuda:3`），
  输出与卡片里记录的实测值逐字相同（只因为 stdout/stderr 而行的顺序不同）：

```
[w4] swapped 196 linears from models/Qwen3-1.7B-qslab-w4-awq2 (backend=w4.auto)
[smooth_kv] loaded results/smooth_kv4_qwen3-1.7b.pt: lambda/kscale for 28 layers
[kvalloc] total=23.53GiB util=0.9 used_by_others=0.49 peak=2.10 current=1.29 budget=18.58GiB block_bytes=7168.0KiB
 Paris. The capital of Germany is Berlin. The capital of Japan is Tokyo. The capital of Korea is Seoul. The capital of Mexico is Mexico City. The
stats: {"steps": 28, "proposals": 16, "accepted": 3, "committed": 31, "acceptance_rate": 0.1875, "mean_len": 1.1071428571428572}
```

- 另一次不带 `--smooth-kv` 的 fp16 贪心冲烟输出塔成 `is is is…`：这是 TODO 17
  已经实测登记过的“缺标定文件就静默退化”，不是本轮重弄坏的；卡片命令（带标定）才是正常路径。
- 两个重构 commit 合计 54 文件 / +113 / −88：除 `runtime/__init__.py` 的层序 docstring
  （+14/−3）外，每个文件改动都不超过 12 行，全是指向搬迁后模块的路径。

## 推翻过的判断（本轮改口的）

| 原来的说法 | 实测后的结论 |
|---|---|
| 把包根“旧时代三件套”算作 config / sampler / **registry** | registry 出列：`qslab/quant/backends.py:21` 在用，无撞名，位置正确 |
| 刀 3 = 三个模块改名 | 拆成三件事：改名（3a）、摘零引用符号（3b）、sampler 双实现（3c）。只有 3a 是改名 |
| `qslab/sampler.py` “被测试锁住所以得留” | 它生产零调用，runtime 另有 `Sampler` + `rejection_verify`，三套实现各自带测试；而且 runtime 根本没 greedy 分支（`sampling_params.py:10` 直接 assert，门面用 `api/llm.py:22 GREEDY=1e-6` 饱和近似）。这不是改名能解决的 |
| `models/loader.py` 六个符号都是参照 oracle | `load_w4_model` / `_wrap_input_scale` / `read_safetensors_state_dict` / `args_model_dir` 零调用，属旧引擎残留；但因 `.origin.txt` 约定没拆（TODO 21） |
| commit `bdd93dc` 写“内容只存在历史里” | 对 `notes/` 11 篇和 8 个脚本不准：它们被搬到了 `.cache/`（逐字节相同，md5 已比），只有 `docs/` 那 26 篇真的只剩历史 |
| ngram/draft 单开 `speculate/` 包 | 实测 `draft.py:100` 真造 `Scheduler`，单开造出包级环 → 归入 `engine/` |

## 提示词 vs 实查

| 提示词里的说法 | 实查结果 |
|---|---|
| “runtime 现在全是散落文件” | 18 个平铺 .py（不含 `__init__.py`），行数 11〜534；对外入口只有 `llm_engine` / `sampling_params` / `config` 三个 |
| “runtime 侧保持原名” | `runtime/config.py`、`runtime/sampling_params.py` 路径未动；包根 `qslab/config.py` 直接消失 |
| “把 models/loader.py + ModelConfig 收进 reference/” | 收了，共 3 个文件（`__init__.py` / `loader.py` / `config.py`）；`qslab/models/` 只剩 `w4linear.py` |
| “做好分层”（未指定层数） | 按实测 DAG 定 4 子包 + 根上 3 件；向上边 0、包级环 0 |

## 挂起项自证（本轮一个没碰）

- `qslab/runtime/attention_store.py`：md5 `2d5dfcac4497ab3c65772826053f2b93` 搬迁前后一致，
  仍在 runtime 根上（零引用所以不进任何层）；它 :7 的 docstring 里 `runtime/attention.py`
  已成旧路径，没改。
- `docs/STUDY-PATH.md`：一字未动，里面 9 行 runtime 旧路径（其中 :34 还写着
  `新 runtime/loader.py`）。
- `models/*.origin.txt`：没碰；它是 `reference/loader.py:129 args_model_dir` 在读的唯一东西。
- `512//M` verify 桶、vLLM 依赖安装：未动。`docs/study/**`：照旧 untracked，一个字节没入库。

## 下一步（不属本 goal，已规置到 TODO）

1. TODO 21：`.origin.txt` 约定跟不跟 `load_w4_model` 一起废弃。
2. TODO 22：79 行教程正文旧路径，人工分辨“引用”还是“逐字输出”。
3. TODO 19：`.gitignore:2 models/` 改成 `/models/`（否则 `qslab/models/` 新增文件会被静默漏接）。
4. 刀 3c：sampler 三套实现的归并（先得定 runtime 要不要真 greedy 分支）。
5. 刀 5：文档收口（旧引擎/旧文档带出的 24 行悬空引用 + `ARCHITECTURE.md §3` 重画 +
   study README 的 HEAD 与 97→104 勘正）。本 goal 只顺手改了因搬迁而错的 runtime 路径。
