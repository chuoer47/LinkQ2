# 离线量化算法

每个算法独立实现 `WeightQuantizer.quantize()`，输出逻辑量化权重与可选输入缩放；后端负责布局打包。

```text
algorithms/
  base.py          WeightQuantizer 抽象类
  rtn.py           RTN：分组对称量化
  rtn_clip.py      RTNClip：搜索分组裁剪范围
  awq.py           AWQ：搜索激活感知输入缩放，再调用 RTNClip
  awq_affine.py    AWQAffine：搜索输入缩放与非对称量化
  __init__.py      类导出与算法名称映射
```

统一接口：

```python
quantize(weight, *, group_size, activation_absmean=None)
# 返回 (QuantizedTensor, input_scale 或 None)
```

- `weight`：形状 `[N,K]` 的浮点权重。
- `group_size`：沿 K 维分组的大小。
- `activation_absmean`：校准得到的输入通道绝对值均值 `[K]`；RTN 与 RTNClip 不使用。
- `requires_calibration`：RTN/RTNClip 为 False，AWQ/AWQAffine 为 True，pipeline 据此决定是否收集激活。
- `default_group_size`：供 CLI 选择默认值，AWQAffine 为 64，其他算法为 128；直接调用 `quantize()` 时显式传入分组大小。

算法参数在实例化时指定：

```python
from qslab.compression.algorithms import RTN, RTNClip, AWQ, AWQAffine

quantizer = AWQ(n_grid=20, max_shrink=1.0)
weight, input_scale = quantizer.quantize(
    fp_weight, group_size=128, activation_absmean=activation_mean)
```

RTN 可配置 `clip_ratio`；RTNClip 可配置 `n_grid`；AWQ 与 AWQAffine 可配置 `n_grid`、`max_shrink`。
当前 AWQ 是项目内的激活感知缩放与量化实现。

新增算法时增加一个实现文件，继承基类，在 `qslab.quantization.QuantizationMethod` 定义算法名称，并加入 `QUANTIZERS` 字典。在线策略独立加入 `RUNTIME_STRATEGIES`。pipeline、CLI 和 checkpoint 使用同一个枚举，名称包含算法与 W4A16 精度。

完整 checkpoint 通过 `qslab.compression.save_checkpoint(artifact, model_path, output)` 保存，沿用源模型权重名与分片。CLI 示例：

```bash
python -m qslab.compression --model models/Qwen3-8B \
  --algorithm awq_w4a16 --calibration datasets/w4a16/calibration.pt \
  --out models/Qwen3-8B-W4-awq
```

小规模测试：

```bash
python -m pytest tests/quant/test_quantization.py -m 'not gpu' -q
```
