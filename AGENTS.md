# 跑测口径（单卡 4090）

## 环境在哪
- 开发与执行都在 4090 服务器；本地工作树是可写的镜像，两边尽量保持一致。
- 模型权重只在服务器的 `models/`（已被 git 忽略）。本地不碰 GPU，也不跑需要权重的测试。
- 同步方向以服务器为准：改动先在本地写好，再上传服务器复跑验证。

## 跑 GPU 任务前
1. `nvidia-smi` 看一遍，挑一张显存和 util 都空着的卡。
2. 用 `CUDA_VISIBLE_DEVICES=N` 把任务钉在那张卡上，一个任务一张卡，不与他人共用。
3. **绝不 kill 不属于我的进程**。显存不够就换卡或降 `gpu_memory_utilization`，不动别人的显存。
4. 汇报结果时写明：用了哪张卡、跑前该卡是否空闲。占用不干净的数据要标注。

## 测试分层
- `pytest.ini`：`testpaths = tests`，`--strict-markers`，标记只有 `gpu` 和 `e2e`。
- 快测（无权重、可在本地跑）：`pytest -q -m "not gpu and not e2e"`
- 全量（服务器，含权重）：`pytest -q`
- 只跑显存相关：`pytest -q -m gpu`；只跑端到端：`pytest -q -m e2e`
- e2e 模块之间由 `tests/conftest.py` 的 autouse fixture 在 teardown 时 `empty_cache()`：
  两个 1.7B 引擎加各自 KV 池无法共存，一个模块必须拆干净了下一个才能建引擎。
  因此不要给 e2e 开并行插件，也不要在一个进程里同时留两个引擎。

## 环境激活
必须用激活的 `qslab` 环境：
`source ~/miniconda3/etc/profile.d/conda.sh && conda activate qslab`。
直接喊 `~/miniconda3/envs/qslab/bin/python` 会缺 `CONDA_PREFIX` 和 ninja 路径，w4.v1 的
C++ 扩展加载失败，测出来的是假红，不是代码问题。

## 改动零影响判据（重构类改动）
- 判据：全量 pytest 通过，且**进度行日志逐字节 md5** 与改前基线一致（新增测试文件除外，
  要单列说明）。
- 基线必须在独占空卡上跑：跑前 `nvidia-smi` 确认选中的卡 0 进程，跑完再查一次。
  别人中途占卡会让 KV 池分配失败，表现为一整片 e2e 红，与代码无关。
- 长跑一律 `setsid nohup ... > /tmp/x.log 2>&1 &` 起后台，再用短命令 tail 轮询。

## 提交
- 一批工作一个 commit（代码 + 测试 + 文档一起），每次提交前都要先问。
- push（含 fork）单独批准，一次授权只对当次有效。

## 别写进代码的东西
- 注释和 docstring 里不放：里程碑编号、实测数字、`results/*.txt` 之类的底稿引用、"上次测得 X" 的叙事。
  这些在 coding 过程中一定失真。校验由 `tests/meta/test_comment_hygiene.py` 兜住。
- 跑测数据落 `results/`（目录已保留，产物不入库），结论进汇报，不进源码。
