# Nsight 内核耗时摘要

| KV | 内核 | 调用次数 | 总 GPU 时间 ms |
| --- | --- | --- | --- |
| fp16 | gemm_splitK_A16fWnO16f_int32packing_kernel | 2016 | 100.918 |
| fp16 | kernel | 16 | 20.302 |
| fp16 | vectorized_elementwise_kernel | 10255 | 12.819 |
| fp16 | elementwise_kernel | 5216 | 9.88 |
| fp16 | reduce_kernel | 1192 | 5.413 |
| fp16 | unrolled_elementwise_kernel | 2329 | 4.791 |
| fp16 | flash_fwd_splitkv_kernel | 252 | 3.464 |
| fp16 | CatArrayBatchedCopy_aligned16_contig | 576 | 1.01 |
| fp16 | index_elementwise_kernel | 290 | 0.919 |
| fp16 | cunn_SoftMaxForward | 16 | 0.831 |
| kv4 | gemm_splitK_A16fWnO16f_int32packing_kernel | 2016 | 100.283 |
| kv4 | kernel | 16 | 20.291 |
| kv4 | vectorized_elementwise_kernel | 10255 | 12.762 |
| kv4 | elementwise_kernel | 5216 | 9.837 |
| kv4 | reduce_kernel | 1192 | 5.384 |
| kv4 | unrolled_elementwise_kernel | 2329 | 4.761 |
| kv4 | flash_fwd_kernel | 288 | 3.231 |
| kv4 | CatArrayBatchedCopy_contig | 504 | 1.114 |
| kv4 | CatArrayBatchedCopy_aligned16_contig | 576 | 1.009 |
| kv4 | index_elementwise_kernel | 290 | 0.914 |

同一内核的模板配置合并统计，每种 KV 列出总耗时前十项。target=engine，上下文 512 token。总时间包含采集区间内多次调用，不能当作单次延迟；按 NVTX 区间查看细节请打开同目录的 .nsys-rep。
