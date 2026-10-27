# 投机推理的性能与质量

## 接受长度与每轮成本

设普通目标 decode 每 token 耗时为 $T_0$，一轮投机的草稿、验证和其他耗时为 $T_d,T_v,T_o$，平均提交 token 数为 $E[C]$，则稳态加速比近似为：

$$
S\approx\frac{E[C]T_0}{T_d+T_v+T_o}.
$$

候选接受率高，不表示整次请求一定更快。自回归草稿需要逐 token 生成，还可能补算尚未写入 KV 的 token；DFlash 按块提出候选，成本应直接测量。

固定窗口为 $k$、不考虑 EOS 截断时，每轮提交数约为 $1+ka$，其中 $a$ 为候选接受率。窗口变化或提前停止时，直接使用实际提交数。

## 如何读结果

[N-gram](../../benchmarks/speculation/ngram/RESULTS.md)、[自回归草稿](../../benchmarks/speculation/draft/RESULTS.md)和 [DFlash](../../benchmarks/speculation/dflash/RESULTS.md)使用相同的问题和非 thinking 模板，分别与不启用投机的目标模型比较。

同时查看总耗时、实际输出 token 数和 TPOT。若两个配置生成长度不同，只比较总秒数会把少生成的工作量误当成加速。接受率反映草稿与目标的匹配，不是回答正确率；这里的六题也不足以代表完整数学、代码或聊天评测。

显存要包含目标、草稿、两侧 KV、目标特征和 Graph。小草稿节省权重，并不保证额外常驻显存很小，也不保证其开销能被接受长度抵消。

## 运行与数值检查

```bash
.conda/bin/python -m benchmarks.speculation.dflash.run --gpu 1 --limit 6
CUDA_VISIBLE_DEVICES=1 .conda/bin/python -m pytest tests/speculation
```

测试检查接受概率、因果范围、连续接受、拒绝回滚、有效 KV 长度以及草稿同步。小模型对齐不等于实际 W4KV4 模型在不同查询行数下逐 token 完全相同；量化和归约顺序都可能改变接近的 logits。性能结果因此需要与计算误差分开解释。

不同权重与缓存组合的性能见 [DFlash 配置对照](../../benchmarks/speculation/dflash/CONFIGURATIONS.md)。
