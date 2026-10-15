# 量化 Linear 算子正确性与性能基准

本基准测量真实 packed W4 权重经 CUDA 算子计算后，与同一份逻辑量化权重的 FP32 参考结果之间的误差，以及一次算子调用的耗时。实验使用合成矩阵，不加载语言模型。

## 实验设计

输入为 `X[M,K]`，权重为 `W[N,K]`，输出为 `Y[M,N]`。`M` 是输入行数，`K` 是输入维度，`N` 是输出维度。默认覆盖三个后端：

| 后端 | 权重来源 | 格式 | Group size |
|---|---|---|---:|
| `qslab_w4a16` | 仓库 AWQ | 对称 W4，项目 packed 格式 | 128 |
| `marlin` | 仓库 AWQ | 对称 W4，重排为 Marlin 格式 | 128 |
| `nunchaku_awq` | 仓库 Nunchaku AWQ | 带 zero point 的 affine W4，Nunchaku 格式 | 64 |

默认 seed 为 2026，生成 FP16 权重 `randn(N,K) × 0.02`、FP16 输入 `randn(M,K) × 0.1` 和合成的 activation absolute-mean 统计。这里的 AWQ 统计来自合成数据，不是 C4 校准。

每个 case 的逻辑量化权重分别用于后端打包与 FP32 参考计算。参考按对称或 affine 规则反量化；若有 AWQ activation scale，两条路径均缩放输入。比较的是算子实现误差，而非量化相对原始浮点权重的损失。

随机数按后端与 case 的执行顺序生成，全量运行不保证不同后端的随机权重逐元素相同；每个 case 的 kernel 与自己的 reference 使用相同逻辑权重。切换后端或 case 子集时，随机样本也可能变化。

矩阵定义在 [cases.py](cases.py)：

| Case | M | N | K |
|---|---:|---:|---:|
| `decode_small` | 1 | 512 | 1024 |
| `decode_batch` | 4 | 512 | 1024 |
| `prefill_batch16` | 16 | 512 | 1024 |
| `prefill_medium` | 32 | 1024 | 2048 |

名称用于标识矩阵规模，不表示执行了完整 decode 或 prefill。Nunchaku 使用 AWQ GEMV，超过 8 行时拆成多次支持的调用并拼接输出；整个适配器调用作为一次算子计时。

## 正确性与计时口径

记录最大绝对误差、RMS、相对 L2 与 cosine similarity。以下两个条件同时满足才通过：

```text
relative_l2 = ||Y_kernel - Y_reference||₂ / max(||Y_reference||₂, 1e-12)
relative_l2 < 5e-3 且 cosine_similarity > 0.9999
```

准备时间 `prepare_ms` 和首次调用时间 `first_call_ms` 单独记录，首次运行可能包含 CUDA 扩展编译或加载。稳态默认预热 20 次、测量 100 次：每次测量前同步设备，然后用主机时钟包围算子调用与结束同步，记录中位数和 P90。

`latency_median_ms` 包含 Python 调用、适配器张量操作、CUDA 执行及同步开销，不是纯 CUDA kernel 时间。`rows_per_second` 是矩阵行吞吐，不是模型生成 token/s。

运行前后记录 GPU 显存与利用率。运行前所选 GPU 占用超过 256 MiB，或利用率超过 10%，设置 `performance_may_be_contended=true`；这类性能数据按共享 GPU 测量解读。快照标记不能保证整个测量期间没有外部干扰。

## 如何运行

在服务器仓库根目录激活已有 `qslab` conda 环境。已验证环境为 Python 3.11、PyTorch `2.5.1+cu124`、CUDA 12.4、Nunchaku `0.3.0+torch2.5`，其包导入需要 `diffusers` 与 Pillow。项目 W4A16、Marlin 扩展需要 CUDA/C++ 编译工具链。已有环境的 wheel 对应 [Nunchaku v0.3.0 上游发布](https://github.com/nunchux-ai/nunchaku/releases/tag/v0.3.0)。

```bash
cd /home/amax01/lingchen/LinkQ2
conda activate qslab

# 本服务器已验证的 conda CUDA / GCC 设置
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4

# 三个后端 × 四种矩阵，选择空闲 GPU
python -u -m benchmarks.quant_kernel.run --device cuda:3

# 独立输出目录，保存完整日志
mkdir -p results/quant_kernel/my_run
python -u -m benchmarks.quant_kernel.run \
  --device cuda:3 --backend all --case all \
  --warmup 20 --repetitions 100 --seed 2026 \
  --output results/quant_kernel/my_run \
  2>&1 | tee results/quant_kernel/my_run/run.log

# 只检查 Marlin 的一个矩阵
python -u -m benchmarks.quant_kernel.run \
  --device cuda:3 --backend marlin --case prefill_medium \
  --output results/quant_kernel/marlin_check
```

上述编号按未预设 `CUDA_VISIBLE_DEVICES` 编写。GPU 快照按 `nvidia-smi` 的物理编号匹配所选编号；外层限制可见 GPU 时需额外核对映射与占用标记。编译设置用于复现本服务器环境，其他环境应使用自身工具链。正式计时应避开同卡上的模型实验。

| 参数 | 默认值 / 用途 |
|---|---|
| `--backend` | `all`；或 `qslab_w4a16`、`marlin`、`nunchaku_awq` |
| `--case` | `all`；或上表的 case 名称 |
| `--device` | `cuda:0` |
| `--warmup` | 20，稳态计时前的预热次数 |
| `--repetitions` | 100，稳态测量次数，应为正整数 |
| `--seed` | 2026 |
| `--output` | `results/quant_kernel/`，写入 `summary.json` |

汇总包含环境、GPU 快照、各 case 的误差、性能、准备与首次调用时间。不支持的形状记录为 `skip`；正确性失败记录为 `fail`，保存汇总后进程以非零状态退出。同一输出目录再次运行会覆盖 summary，建议每次使用独立目录。

## 已完成实验：2026-10-13

运行编号 `2026-10-13_144133`。采用质量实验全部结束后，在物理 GPU 3 上重新测量的 `clean` 结果。硬件 RTX 4090 D，PyTorch `2.5.1+cu124`，CUDA 12.4，seed 2026，预热 20 次，测量 100 次，运行前占用标记为 `false`。

下表是中位调用延迟，单位 **微秒（µs）**，由 JSON 的毫秒值乘以 1000 得到：

| Case（M,N,K） | qslab W4A16 | Marlin | Nunchaku AWQ |
|---|---:|---:|---:|
| `decode_small`（1,512,1024） | 24.935 | 241.589 | 714.743 |
| `decode_batch`（4,512,1024） | 31.386 | 241.813 | 714.830 |
| `prefill_batch16`（16,512,1024） | 43.191 | 241.880 | 695.285 |
| `prefill_medium`（32,1024,2048） | 143.706 | 254.791 | 731.361 |

共 **12/12 case 通过**。各后端在四个 case 中最大的相对 L2 误差：

| 后端 | 最大 relative L2 |
|---|---:|
| qslab W4A16 | 0.0003025 |
| Marlin | 0.0006634 |
| Nunchaku AWQ | 0.0005730 |

在这四种小矩阵与当前适配器的同步主机计时口径下，qslab W4A16 延迟最低。不能据此推导纯 kernel 排名或真实模型长输入吞吐；后端的分组、量化格式与适配器开销存在差异。

### 覆盖范围与后续发现

历史结果仅覆盖 `M ≤ 32`。后续 real-quant 整模型实验发现，当前 sm_89 / CUDA 12.4 环境中，原生 Marlin 的部分大行数调用（包括 `M=48`、`M=2048`）出现额外数值误差与重复运行差异。此前 12 个 case 通过，不能说明长输入路径正确。

曾尝试在 [Marlin 包装层](../../qslab/kernels/marlin/ops.py) 中按 32 行分块，此处理现已撤销，调用恢复为原始版本，vendor CUDA 源码保持不变。内部根因仍未确定，数值异常按原始路径记录。[补充 GPU 回归检查](../../tests/kernels/test_marlin_prefill.py) 覆盖 32、48、65、2048 行。

上表是实施临时分块方案之前保存的历史测量。恢复原始调用后也未重跑这些历史 case。模型维度与长输入验证见 [real_quant](../real_quant/README.md)。

## 原始记录

基准 revision 为 `120853ea1a309bfae180a6d39b74b27776bfbfa5`，执行时的修正保存在 `run-info/source.diff`。

- 本地归档：[kernel/summary.json](../../benchmark-results/2026-10-13_144133/kernel/summary.json)、[隔离计时启动脚本](../../benchmark-results/2026-10-13_144133/run-info/run_clean_kernel.py)、[结果报告](../../benchmark-results/2026-10-13_144133/report.md)。归档位于本次工作区，其他 checkout 需另行获取。
- 服务器最终结果：`/home/amax01/lingchen/LinkQ2/results/quant_kernel/2026-10-13_144133/clean/`，包含 `summary.json` 与 `run.log`。
- 服务器启动记录：`/home/amax01/lingchen/LinkQ2/.cache/benchmark_runs/2026-10-13_144133/`，包含 `run_clean_kernel.py`、命令、状态、返回码与源码差异。

## 补测 FP16 / Fake Linear：2026-10-14

只补测缺失的 dense 对照，原有量化 kernel 实验没有重跑。完整报告见 [补测报告](../../benchmark-results/2026-10-14_092426-linear-supplement/report.md)。

自研 W4A16 的同形状对照如下，延迟为中位数，单位 µs。Fake 将 AWQ 输入缩放提前折入重建的 FP16 权重；预热 20 次、正式测量 100 次，计时复用原 `measure()`。

| (M,N,K) | FP16 | Fake | 自研 W4A16（历史） | 自研相对 Fake 的耗时倍数 |
|---|---:|---:|---:|---:|
| (1,512,1024) | 14.842 | 14.600 | 24.935 | 1.71× |
| (4,512,1024) | 16.559 | 16.486 | 31.386 | 1.90× |
| (16,512,1024) | 16.597 | 16.571 | 43.191 | 2.61× |
| (32,1024,2048) | 19.371 | 19.486 | 143.706 | 7.37× |

这四种小矩阵的历史自研算子调用比新增 FP16/Fake 对照更慢。历史量化数据与新增 dense 数据来自不同时段，因此比值属于跨运行对照；原始输入张量未归档，本轮按原 seed 与后端/case 顺序重建，并保存重建张量哈希。其他两个后端的逐项对照与 P90 见补测报告及 JSON。

补测脚本为 `supplement_linear.py`，不导入或调用量化后端适配器：

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n qslab python -m benchmarks.quant_kernel.supplement_linear \
  --historical-summary results/quant_kernel/2026-10-13_144133/clean/summary.json \
  --output results/quant_kernel/my-linear-supplement
```
