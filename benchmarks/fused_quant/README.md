# 成熟融合 W4A16 算子基准

使用原版 GemLite 0.4.8，将已有 RTN、RTNClip、AWQ、NunchakuAWQ artifact 的逻辑整数重打包。**不重新量化，不修改第三方 kernel，不接推理引擎。** 使用现有 PyTorch 2.5.1，在项目 venv 内安装 Triton 3.3.0；不代表最新 GemLite 版本的性能上限。0.4.7 在原 Triton 3.1.0 中导入失败，预检日志予以保留。

## 范围与指标

- FP16：原始模型、四种算法各自 fake 和 GemLite，共 9 个配置。
- BF16：原始模型、RTN fake、RTN GemLite，共 3 个配置。
- PPL：复用前次成功实验的同一份 8×512 token manifest，4088 个计分目标；报告相对原始模型和配对 fake 的差异。
- prefill：B=1、S=128/512，完整 HF eager forward，包括 lm_head。
- decode：普通 HF 动态 KV cache，128 token 上下文后 32 步固定输入；记录缓存类型、精度与容量，排除上下文 prefill 时间。
- 延迟：预热 2 次、重复 5 次，CUDA 同步主机墙钟；提供原始样本、中位数、P90 和相对同精度原始模型加速比。
- 显存：权重独立 storage 总量、执行后清理的常驻 allocated、加载/PPL/prefill/decode 的峰值 allocated 和 reserved。
- 单算子：RTN 的第一层四类实际投影形状，M=1/32/128/512；同输入比较原始浮点 Linear、fake Linear 和 GemLite。分别记录同步 host 延迟和 CUDA graph/event 的 GPU 延迟；图回放测量属于热缓存条件，不能视为全模型收益。
- 数值门槛：第一层 7 个投影 ×4 个 M，与 FP32 逻辑权重参考比较；FP16 relative L2<0.005，BF16<0.01，cosine>0.9999。全模型 probe 核实 252 个替换层执行，并保存相对 fake 的 logits 和 decode 差异。

主矩阵向 GemLite 请求 `acc_dtype=FP32`，BF16/FP16 表示输入与输出精度。GEMM/SplitK 按请求设置；M=1 的 GEMV_REVSPLITK 由上游覆盖为输入 dtype，并使用输出 dtype 的原子归约，不能称作全路径 FP32 累加。scale 保留 artifact 原始 FP16 类型。`fma_mode=False` 保留 `(q-zero)*scale` 运算，避免提前舍入 `zero*scale`；对称 int4 通过 `q+8, zero=8` 等价变换为 unsigned int4。AWQ 的 real 路径在 forward 除输入 scale；fake 路径将其离线折入浮点权重，存在舍入顺序差异。

补充实验比较两种接入方式：`module` 使用上游 GemLiteLinear 和 torch.library 调度；`direct` 由本项目适配器直接调用同一上游 kernel.forward，并预先保存不变的张量和元数据。重打包、选核规则、算子数学运算保持相同，不修改 GemLite 包文件。每个 direct 投影额外与 module 输出配对校验。对 RTN 另测一组 FP16 累加，以区分累加精度和接入方式的影响。

自动调优采用上游默认 fast，清除内置其他型号预设，在当前 GPU 运行前验证并调优；每种输入/累加精度单独缓存配置。编译和调优不计入稳态延迟，加载和质量阶段耗时单独记录。单卡顺序运行，每个配置独立进程，不重跑旧的自研、Marlin、Nunchaku kernel 实验。

## 环境与运行

以下命令在服务器项目根目录执行。环境位于项目内，继承原 conda 的只读包，只额外安装 GemLite 和 Triton，禁止 pip 自动拉取或升级其他依赖。PyTorch 2.5.1 的包元数据固定 Triton 3.1.0，本环境对 GemLite 直接调用采用 Triton 3.3.0；不使用 torch.compile/Inductor，不能视为通用训练或编译环境。

```bash
/home/amax01/miniconda3/envs/qslab/bin/python -m venv --system-site-packages .venvs/gemlite-0.4.8
.venvs/gemlite-0.4.8/bin/python -m pip install --no-deps --no-build-isolation .cache/gemlite/triton-3.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl .cache/gemlite/gemlite-0.4.8.tar.gz
.venvs/gemlite-0.4.8/bin/python -u -m benchmarks.fused_quant.run \
  --gpu 1 --output results/fused_quant/$(date +%Y-%m-%d_%H%M%S)-gemlite
```

原始矩阵完成后，复用该目录的浮点/fake 基线，仅补充 direct 接入及 FP16 累加：

```bash
.venvs/gemlite-0.4.8/bin/python -u -m benchmarks.fused_quant.supplement \
  --output results/fused_quant/上一条命令实际生成的目录
```

补充实验单独保存 `supplement_manifest.json` 和 `source_direct.zip`，已有 case 禁止覆盖。

FP16 累加的 down_proj 在 M=128/512 时 relative L2 约 0.00576，超过预设 0.005 门槛。标准流程在这里终止并保留失败记录。若要观察其 PPL 和延迟，可显式运行独立的诊断 case，结果标记为不参与有效加速结论：

```bash
CUDA_VISIBLE_DEVICES=1 .venvs/gemlite-0.4.8/bin/python -u -m benchmarks.fused_quant.worker \
  --output results/fused_quant/同一次实验目录 --dtype fp16 --algorithm rtn \
  --mode gemlite --dispatch direct --accumulation fp16 \
  --diagnostic-after-rejection --micro
```

官方包下载地址、版本及 SHA256 记录在 `.cache/gemlite/packages048.json` 并复制进每次实验的 manifest。本实验不使用未经验证的新 artifact，启动时检查 logical manifest 和 tensor 文件与旧实验缓存的完整 SHA256 一致。

结果保存在新目录，已有目录禁止覆盖。`manifest.json` 记录范围、来源和状态，`jobs.json` 保存 PID、命令、起止时间；`cases/` 为原始数据，`summary.json/csv` 和中文 `README.md` 为汇总，`logs/` 为完整日志，`source.zip` 为源码快照。

前台按 Ctrl+C，或向父进程发送 SIGTERM，会终止该实验自己的子进程组。不要使用按 Python 名称批量杀进程的命令。
