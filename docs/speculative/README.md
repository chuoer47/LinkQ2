# 投机推理专题

本专题分开说明纯算法、模型接入与实际收益：

1. [候选与接受算法](01-proposal-and-acceptance.md)：k+1 验证、greedy 前缀接受、概率比与残差采样、四类 proposer。
2. [模型与状态同步](02-models-and-state.md)：自回归草稿、DFlash、特征前缀、两侧 KV/Graph 和使用示例。
3. [收益与验证](03-performance-and-validation.md)：打平条件、接受长度、已有正负案例、复现及严格一致性边界。

请求调度与资源管理见 [Runtime](../runtime/README.md)，计算路径见 [Kernel](../kernels/README.md)。DFlash 当前范围为 greedy、非 thinking；量化路径的性能收益不等于已经验收严格无损生成。
