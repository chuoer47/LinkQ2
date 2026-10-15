# Real-quant 小样本 benchmark

本实验衡量真实量化接入完整模型后，相对 fake-quant 和 FP16 的实际性能收益、
实际显存收益，以及 real/fake 的 PPL 差异。采用固定小样本，默认不会跑全量测试集。

## 范围与约束

- 模型为本地 Qwen3-8B，使用 Hugging Face eager attention，batch size 为 1。
- 每层 q/k/v/o/gate/up/down 七类投影共 252 个 Linear，替换成现有真实量化模块，
  常驻 packed INT4 权重，调用原始 CUDA 算子。
- embedding、norm、LM head 与激活保留 FP16。完整 forward 包括 LM head。
- 不接入项目推理引擎，不使用 CUDA Graph、投机解码或量化 KV cache。
- 第三方算子及调用参数保持原样，不为通过验证修改算子或增加新的分块路径。
  启动时检查 Marlin、自研 W4A16 的代码相对仓库 HEAD 没有差异。
- Nunchaku 使用项目已有的适配路径，每次扩展调用至多 8 行；不是本实验新增的处理。

原来的全量运行 `2026-10-13_183137` 已停止。该运行的 Marlin 使用过临时 32 行分块封装，
此封装已撤销，旧结果仅作历史记录，不能复用为本实验原始算子的结果。

## 对照矩阵

| 量化算法 | 真实算子 | 分组与表示 |
|---|---|---|
| RTN | qslab W4A16、Marlin | symmetric INT4，group 128 |
| RTN Clip | qslab W4A16、Marlin | symmetric INT4，group 128 |
| AWQ | qslab W4A16、Marlin | symmetric INT4，group 128 |
| Nunchaku AWQ | Nunchaku AWQ | affine INT4，group 64 |

共 7 组 real，加上 FP16 基线与 4 组配对 fake，共测量 12 个模型配置。
每个算法仅生成一份逻辑量化产物：fake 使用它重建的 FP16 权重，real 使用它导出的
packed 权重。同一算法的 qslab 与 Marlin 共用相同量化值、scale 和 AWQ 输入缩放。
不同算法/group 的横向差异需结合配置解释。

## PPL：固定 8 段 × 512 token

使用已有 WikiText-2 test token 数据，按长度 512 划分为不重叠候选块；固定种子
`20261013` 抽取 8 块，再按原位置排序。`samples.json` 保存片段起点、完整 token、
样本哈希。所有模型使用完全相同的 4096 个输入 token。

每段独立执行一次无 cache 的完整 forward，预测该段第 2 至第 512 个 token。
共评分 `8 × 511 = 4088` 个目标，不补 padding、不跨段预测、不用重叠窗口。
交叉熵在 FP32 计算，总体 `PPL = exp(所有片段 NLL 之和 / 有效目标数)`，不平均片段 PPL。

保存每段及总体 PPL，报告 real/fake 的绝对差、相对差，另保存 FP16 基线。
这是固定小样本上的配对差异，不代表全量 WikiText-2 质量。
缩小评测集不改变现有 AWQ 校准配置；校准仍使用已有 C4 数据。

## 性能：prefill 与 decode

性能使用 `samples.json` 中第一个片段，batch 1。各配置均在 `--devices` 的第一张 GPU
顺序测量，使用新的独立进程。其余指定 GPU 仅用于准备阶段。
选中的 GPU 在 worker 启动前已有明显占用时，任务报错，避免共享同卡导致不公平计时。

### Prefill

默认长度 128、512，`use_cache=False`。每种长度先记录首次测量 forward，再预热 2 次，
清理临时对象和空闲 allocator 缓存，正式重复 5 次。
记录完整 forward 延迟中位数、nearest-rank P90、每次原始延迟、输入 token/s。
第一次测量发生在正确性/PPL 检查之后，因此不是冷启动时间。

### Decode

每轮先以相同的 128-token 上下文执行 `use_cache=True` 的 prefill，获得新的 HF 动态 KV cache；
随后把数据中的后续 32 个 token 逐步喂入，每步只输入一个 token。
采用固定真实 token，而不是各自 argmax，保证模型之间的输入与 cache 长度一致。
每轮结束检查 cache 长度为 160，并记录其 FP16 K/V 张量存储量。

先执行一次首轮，再预热 2 个完整轮次，正式重复 5 轮。
每轮取 32 步延迟平均值，再对 5 轮报告中位数、P90 和 token/s；
同时保存逐步延迟和 pooled step 分位数。上下文 prefill 的延迟单独保存，不计入 decode。
模型调用包括 LM head、HF 调度和动态 cache 更新；不包括采样器、网络服务或文本输出。

计时均使用 forward 前后 CUDA 同步的主机墙钟。输入 token 已在 GPU 上，CPU→GPU
数据传输、模型加载、量化、格式转换、编译和正确性检查均不计入稳定运行延迟。
5 次重复的 P90 只适合粗略观察；本实验不宣称统计意义上的精确尾延迟。

每个 real 配置按同样场景计算：

- 相对 fake 加速比 = fake 延迟 / real 延迟。
- 相对 FP16 加速比 = FP16 延迟 / real 延迟。

加速比大于 1 表示更快，小于 1 表示更慢。

## 显存口径

所有数据以字节保存在 JSON，报告中 GiB 使用 `2^30` 字节。

| 指标 | 定义 |
|---|---|
| 模型存储 | GPU 独立 storage 去重统计：未量化参数、packed 权重、scale/zero/input-scale、workspace |
| 常驻显存 | 加载与 probe 完成、临时对象清理后、不带 KV cache 的 PyTorch allocated |
| 加载峰值 | 模型加载和替换阶段的 PyTorch allocated 峰值 |
| PPL 峰值 | 8 段评分中的 allocated/reserved 峰值，含 logits 与交叉熵临时张量 |
| Prefill 峰值 | 每种长度正式测量的 allocated/reserved 峰值，含完整 logits |
| Decode 峰值 | 正式 5 轮的峰值，含上下文 prefill、逐步 forward 与 KV cache |
| KV cache | 每轮上下文结束及 decode 结束时 K/V 的独立 storage、dtype 和长度 |
| 硬件占用 | CUDA 初始化、加载、probe 后的 nvidia-smi 快照；包含 context 与第三方私有分配 |

PyTorch allocated 表示活跃分配，reserved 包含 allocator 缓存，不等同于实际模型存储。
硬件快照不能当成瞬时峰值；外部扩展私有分配也可能不计入 PyTorch 峰值。
原始记录并列保留这些口径，不将它们混为同一个数字。
常驻与各运行场景的峰值分别计算相对 fake、FP16 的节省比例：`1 - real / reference`。
KV cache 使用相同的普通 FP16 实现，避免把 cache 精度变化算作权重量化收益。

## 正确性与异常

加载时检查 packed 形状与全部目标层覆盖。短模型 probe 用 hook 验证全部 252 个量化层执行。
首层七类投影在 `M=1,32,128,512` 与同份逻辑权重的 FP32 reference 对照，记录 relative L2、
cosine 与绝对误差。算子检查沿用 relative L2 `< 5e-3`、cosine `> 0.9999` 的标准。
另保存短输入和 32 步 decode 的 real/fake logits 对照。

PPL 相对差默认超过 1% 时标记警示；模型 logits 相对 L2 超过 2% 也标记警示。
这些是诊断阈值，不替代原始数据。数值失败不会触发替换算子或隐藏该配置：
仍完成小样本测量，将失败检查与观测延迟写入报告。
异常场景的速度比只能称为观测值，不能当作正确路径的有效性能收益。
运行错误保存 traceback；准备或配对基线失败时中止，real 配置失败时其他配置继续执行。
最终报告明确区分执行完成与数值通过。
此前在此机器发现过 Marlin 长输入数值异常，恢复原始调用后必须重新观察，根因未确定。

## 运行与产物

在服务器仓库根目录，使用现有 `qslab` conda 环境：

```bash
conda run --no-capture-output -n qslab python -m benchmarks.real_quant.run \
  --devices 1 2 3 --output results/real_quant/my-small-run
```

默认就是上述小样本协议。可通过 `--sample-count`、`--sample-length`、`--sample-seed`、
`--lengths`、`--decode-context`、`--decode-steps`、`--warmup`、`--repetitions` 调整，
全部参数写入记录。参数必须保证性能输入位于一个保存的片段内。
`--algorithms`、`--backends` 可选择子矩阵，`--force` 强制重测输出目录中的任务。

逻辑与 packed 权重缓存在 `.cache/real_quant/artifacts/<artifact_identity>/<algorithm>`。
复用前核对模型、数据、量化源码、环境与产物文件 SHA256；本次只变评测协议时可复用权重。
历史完成记录中的量化时间另行保留，不冒充本轮实际重新量化耗时。

结果目录包含：

- `README` 对应的协议已在源码中保存；`source.zip` 与 `source.diff` 为本轮代码快照。
- `identity.json`：模型/数据/源码/环境及评测协议；`invocation.json`：实际参数和 GPU。
- `samples.json`：可复现的完整小样本 token。
- `jobs/`、`logs/`：每个任务的命令、PID、耗时、退出码和日志。
- `prepare/`、`reference/`、`real/`：逐配置原始结果；reference 另保留配对 logits。
- `summary.json`、中文 `report.md`：PPL、性能、显存、对照收益与异常汇总。
- `status.json`：实验运行、完成、失败或停止状态。

每个模型一次加载后完成全部测量，避免额外的重复加载阶段。
默认实验只测小样本；不支持旧版默认全量 PPL 流程。
