# Fake-quant 模型质量基准

本基准测量 Qwen3-8B 的目标权重量化再还原后，模型困惑度（PPL）相对 FP16 增加多少。PPL 越低越好。

## 实验设计

使用 Hugging Face Qwen3-8B，FP16 权重与计算，attention 为 `eager`，评测设置 `use_cache=False`。处理每个 decoder 层的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`；本次各量化配置实际处理 252 个 Linear。Embedding、LM head 和其他模块保持原模型参数。

Fake-quant 先生成逻辑量化值、scale 和必要的 zero point，再反量化并写回原 Linear 的 FP16 权重；AWQ 的 activation scale 合并回权重。前向仍由普通浮点 Linear 完成。本实验用于判断量化算法的模型质量，不输出部署用 packed 权重，也不调用量化 CUDA 算子。

| 默认配置 | Group size | 校准数据 |
|---|---:|---|
| `baseline`：原始 FP16 | — | 无 |
| `rtn` | 128 | 无 |
| `rtn_clip` | 128 | 无 |
| `awq` | 128 | C4，128 条 × 2048 token |
| `nunchaku_awq` | 64 | 同一份 C4 校准 token |

C4 从 `allenai/c4` 的 `en/train` 流式读取，按数据顺序 tokenize、拼接并切成 2048-token 序列，取前 128 条，不做随机抽样。评测数据为 `Salesforce/wikitext` 的 `wikitext-2-raw-v1/test`：过滤空文本后用双换行拼接，使用模型 tokenizer，`add_special_tokens=False`。

PPL 使用 2048-token 滑动窗口、stride 512。首个窗口计算全部可预测 token，后续窗口只累计新出现的预测目标，重叠上下文不重复计分。最终按 token 数汇总：

```text
mean_nll = 已计分 token 的负对数似然之和 / 已计分 token 数
PPL = exp(mean_nll)
相对 FP16 变化 (%) = (PPL_quant / PPL_baseline - 1) × 100
```

每个配置在独立子进程中加载模型、量化、评测；默认入口按顺序运行，进程退出后释放 GPU 分配。

## 如何运行

在服务器仓库根目录、已有的 `qslab` conda 环境中运行。模型默认在 `models/Qwen3-8B`；首次准备数据需要能读取上述 Hugging Face 数据集，后续复用本地缓存。

```bash
cd /home/amax01/lingchen/LinkQ2
conda activate qslab

# FP16 + 四种量化算法，顺序使用指定 GPU
python -u -m benchmarks.fake_quant_quality.run --device cuda:1

# 指定模型和独立输出目录，并保存控制台日志
mkdir -p results/fake_quant_quality/my_run
python -u -m benchmarks.fake_quant_quality.run \
  --device cuda:1 --model models/Qwen3-8B \
  --output results/fake_quant_quality/my_run \
  2>&1 | tee results/fake_quant_quality/my_run/run.log

# 只比较 FP16 和 RTN
python -u -m benchmarks.fake_quant_quality.run \
  --device cuda:1 --algorithms baseline rtn \
  --output results/fake_quant_quality/rtn_check
```

示例按服务器未预设 `CUDA_VISIBLE_DEVICES` 编写；若外层已限制可见 GPU，需要先确认编号映射。

| 参数 | 用途 |
|---|---|
| `--device` | 默认 `cuda:0`；入口将选中 GPU 映射为子进程的 `cuda:0` |
| `--model` | 模型目录，默认 `models/Qwen3-8B` |
| `--output` | 默认 `results/fake_quant_quality/qwen3-8b/` |
| `--algorithms` | 可选 `baseline rtn rtn_clip awq nunchaku_awq` |
| `--force` | 覆盖所选算法已有的指标 JSON；默认跳过已有结果 |
| `--force-data` | 重新生成校准与评测 token 缓存 |

窗口、stride、group size 和校准数量在 [config.py](config.py) 中定义，目前没有对应 CLI 参数。入口会准备两份数据缓存，即使只运行 RTN；worker 仅执行 AWQ 类算法时加载校准 token。改变模型、tokenizer 或数据设置后，应重新生成缓存，并使用新的输出目录，避免混合旧结果。

## 输出与复现

- `datasets/fake_quant_quality/qwen3-8b/`：`calibration.pt`、`wikitext_test.pt`、`manifest.json`；manifest 记录 token 数和 token SHA-256。
- 输出目录下的 `<algorithm>.json`：PPL、累计 NLL、计分 token 数、实际量化层数、group size、评测设置。
- `summary.json`：所选配置结果，以及包含 baseline 时的 PPL 差值。CLI 不单独保存日志，上面的 `tee` 示例可保存完整控制台输出。

## 已完成实验：2026-10-13

运行编号 `2026-10-13_144133`，模型 Qwen3-8B，硬件 RTX 4090 D，服务器 `qslab` 环境，PyTorch `2.5.1+cu124`。基准 revision 为 `120853ea1a309bfae180a6d39b74b27776bfbfa5`，执行时的两处代码修正保存在 `run-info/source.diff`。

评测 token 流长 298,938，每个配置实际计分 **298,937** 个 token；首个 token 不作为预测目标。每个量化配置处理 252 个 Linear。

| 配置 | Group size | PPL ↓ | 相对 FP16 增加 |
|---|---:|---:|---:|
| FP16 baseline | — | 8.582099 | 0.000% |
| RTN | 128 | 9.094096 | 5.966% |
| RTN Clip | 128 | 9.149667 | 6.613% |
| AWQ | 128 | 9.618920 | 12.081% |
| Nunchaku AWQ | 64 | 8.717640 | 1.579% |

这次 Nunchaku AWQ 的 PPL 最接近 FP16；AWQ 的 PPL 增幅最大，RTN Clip 也没有优于 RTN。结论限定于仓库当前算法实现、模型与数据设置。Nunchaku AWQ 的 group size 为 64，其余为 128，差异包含算法与分组粒度的影响；结果不能用来判断部署显存或推理速度。

本次 FP16、AWQ 使用物理 GPU 1，RTN、Nunchaku AWQ 使用 GPU 2，RTN Clip 使用 GPU 3；同一 GPU 上的任务顺序执行。多卡调度由保存的 `run_all.py` 完成，普通 CLI 仍是单卡顺序运行。

原始记录：

- 本地归档：[quality/summary.json](../../benchmark-results/2026-10-13_144133/quality/summary.json)、[运行记录与脚本](../../benchmark-results/2026-10-13_144133/run-info/)、[结果报告](../../benchmark-results/2026-10-13_144133/report.md)。归档位于本次工作区，其他 checkout 需另行获取。
- 服务器结果：`/home/amax01/lingchen/LinkQ2/results/fake_quant_quality/2026-10-13_144133/`。
- 服务器启动记录：`/home/amax01/lingchen/LinkQ2/.cache/benchmark_runs/2026-10-13_144133/`，包含 `run_all.py`、命令、日志、返回码、revision 与数据 manifest。

整模型 packed 权重与量化算子接入见 [real_quant](../real_quant/README.md)。
