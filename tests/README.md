# 正确性测试

测试围绕项目的四个模块，使用固定随机种子、随机张量和临时生成的小 Qwen3 checkpoint。只验证关键行为和数值误差，不要求代码覆盖率。

| 目录 | 验证内容 |
|---|---|
| `compression/` | 离线算法、校准要求、完整 checkpoint、原分片与权重保留、删除源目录后独立加载 |
| `kernels/` | W4 打包执行对照 FP32 反量化矩阵乘；KV4 编码、分页 decode/verify、因果遮罩、跨页与前缀物化对照 PyTorch |
| `runtime/` | 请求状态、调度预算、KV 页分配/共享/释放、校准布局、混合 prefill/decode、Graph 重放、模型数值对照与关闭释放 |
| `speculation/` | n-gram/lookahead、接受概率与重采样分布、连续接受前缀、提交与回滚、普通草稿及 DFlash 缓存和特征同步 |

## 运行

在 Linux/CUDA 的仓库根目录使用项目 `.conda`：

```bash
# 不执行 GPU 测试
.conda/bin/python -m pytest tests -m "not gpu"

# 指定空闲物理 GPU，运行全部测试
CUDA_VISIBLE_DEVICES=1 .conda/bin/python -m pytest tests

# 按模块执行，例如只测 Kernel
CUDA_VISIBLE_DEVICES=1 .conda/bin/python -m pytest tests/kernels
```

`gpu` 标记只表示运行条件；没有 CUDA 时这些测试跳过。Runtime 和投机执行测试会在临时目录创建小模型、W4 包和 KV4 参数，结束后由 pytest 清理，不读取现有 1.7B/8B 权重。

DFlash 的小模型测试需要官方模型 Python 实现，默认从 `models/Qwen3-8B-DFlash-b16` 读取，也可以设置 `DFLASH_CHECKPOINT`。只复用其代码，随机初始化小草稿权重；目录不存在时明确跳过。其余测试无需下载模型。

## 测试与 benchmark 的边界

- 测试回答：算法/张量是否正确、误差是否在指定容差内、请求和缓存状态是否正确、资源是否归还。数值 oracle 来自公式、FP32 PyTorch/HF 参考或相同小模型的非投机执行，不固定自然语言生成文本。
- [benchmark](../benchmarks/README.md)回答：真实模型的 PPL/生成质量、耗时、吞吐、显存峰值、接受长度与场景收益。它保存测量原始数据和报告，不以某个速度或接受率作为 pytest 的通过门槛。
- Kernel 误差与性能分开：正确性对照使用相同逻辑量化值，避免把 int4 本身的误差与解包/计算错误混在一起；小张量 roundtrip 另行验证量化误差范围。整模质量仍由 benchmark 测量。

不维护模拟 HTTP 测试、固定文本/token 序列验收、注释/行数规范检查或依赖完整模型的重复验收。接口和性能数据可通过对应 benchmark 或按需冒烟检查。

小模型投机输出与基线对齐只验证缓存/接受执行链路，不代表真实 W4KV4 模型已经通过严格逐 token 无损验收。
