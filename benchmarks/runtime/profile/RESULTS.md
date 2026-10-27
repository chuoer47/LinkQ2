# Nsight 性能分析结果

[单层 Attention](LAYER.md)比较 FP16 KV 与 KV4 的 decode、verify 和分块 prefill；[完整模型](ENGINE.md)比较相同 W4 模型下的请求执行。

表中是采集区间内各内核的累计 GPU 时间。查看调用间隙与具体 NVTX 区间时，使用 results/runtime/profile 下对应的 .nsys-rep。
