# Nsight 内核耗时摘要

| KV | 内核 | 调用次数 | 总 GPU 时间 ms |
| --- | --- | --- | --- |
| fp16 | flash_fwd_splitkv_kernel | 9 | 0.124 |
| fp16 | elementwise_kernel | 12 | 0.024 |
| fp16 | vectorized_elementwise_kernel | 18 | 0.02 |
| fp16 | flash_fwd_splitkv_combine_kernel | 6 | 0.019 |
| fp16 | store_kvcache_kernel | 9 | 0.011 |
| kv4 | kv4_grouped_verify_kernel | 3 | 0.059 |
| kv4 | flash_fwd_kernel | 3 | 0.059 |
| kv4 | kv4_paged_decode_kernel | 3 | 0.041 |
| kv4 | elementwise_kernel | 12 | 0.024 |
| kv4 | CatArrayBatchedCopy_contig | 6 | 0.017 |
| kv4 | store_kv_quant_kernel | 9 | 0.015 |
| kv4 | merge_kv4_splits | 6 | 0.009 |
| kv4 | materialize_kv_kernel | 3 | 0.008 |
| kv4 | vectorized_elementwise_kernel | 6 | 0.007 |

同一内核的模板配置合并统计，每种 KV 列出总耗时前十项。target=layer，上下文 512 token。总时间包含采集区间内多次调用，不能当作单次延迟；按 NVTX 区间查看细节请打开同目录的 .nsys-rep。
