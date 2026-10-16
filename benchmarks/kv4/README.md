# KV4 小规模整模 benchmark

固定 Qwen3-8B FP16 权重，对比 FP16 KV 与项目 INT4 KV 的精度、速度和存储。
两者使用相同的离线 SmoothAttention 参数，模型来自 `qslab/runtime/model`。

## 一键运行

在仓库根目录、项目 `.conda` 环境下运行：

```bash
conda activate "$PWD/.conda"
python -m benchmarks.kv4.run --gpu 1
```

需要与 torch 匹配的 FlashAttention。本轮环境为 torch 2.7.1+cu126、FlashAttention 2.8.3，均安装在项目环境。
默认原模型为 `models/Qwen3-8B`。若 `datasets/kv4/qwen3-8b.pt` 不存在，先用现有 C4 训练 token 的 8×512 重新校准。
可通过 `--model`、`--calibration`、`--calibration-tokens`、`--samples`、`--v-group`、`--out` 修改配置。

## 精度口径

- WikiText-2 固定 8 段×512 token，4088 个预测目标，与校准集分离。
- 每段第一个 token 使用 prefill，后续按 32 query 的 causal verify 路径执行，实际读取 INT4 KV；没有用完整浮点 prefill 冒充 KV4 PPL。
- 每行只访问自己的历史及当前 token，量化后的前层输出会继续传入后层。
- FP16 对照使用官方 `flash_attn_with_kvcache`，先验证分块执行与完整 FP16 prefill 的 logits 一致性。
- 比较整模 PPL、首段前 32 个位置的 logits 相对 L2/top-1、全部 attention 层执行覆盖。

## 速度和显存口径

- batch=1，FP16 权重、无 CUDA graph、无 torch.compile。
- prefill 长度 128/512；decode 从 128/512 上下文各逐 token 执行 32 步。
- 预热 1 轮，测量 3 轮；报告同步主机墙钟中位数，含完整模型、KV 写入/读取及最后位置 lm_head。
- 上下文元数据准备、调度器和采样不在计时内。
- 两种 KV 使用相同 1024 slot 容量、256 token 分页，以满足官方 FlashAttention paged KV 接口要求。
- FP16 基线使用官方融合 KV 更新/attention API；KV4 使用项目真实写入和 attention 算子。prefill 分别走官方 cache API 与项目的写入+varlen 路径。
- 单独报告 KV 张量字节、PyTorch allocated/reserved/峰值和释放记录，不将预分配池大小当作整个模型显存。

默认不接入 W4 权重，不测投机接受率、多请求吞吐或完整语料 PPL。
结果包含中文报告、JSON、日志及固定 token，位于 `results/kv4/<时间>/`。

## 修复回归测试

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest \
  tests/runtime/model/test_kv4_configuration.py \
  tests/runtime/model/test_kv4_paged.py \
  tests/runtime/model/test_verify_kernel.py -q
```

覆盖必需校准文件、有效静态 scale、非默认 V 分组的真实读写、分页与因果边界。
