# Runtime 专题

本专题从单 GPU 引擎的加载讲到请求结束与资源释放：

1. [架构、加载与生命周期](01-architecture.md)：各层职责、初始化、配置、请求准入及关闭。
2. [调度与执行](02-scheduling-execution.md)：Sequence、分页 KV、抢占、分块 prefill、执行路径选择与投机提交。
3. [CUDA Graph 设计](04-cuda-graph.md)：捕获范围、固定 buffer、图族与 batch 档位、填充隔离、图外执行、显存及输出生命周期。
4. [使用与验证](03-usage-validation.md)：Python、CLI、HTTP、参数组合、运行命令及指标说明。

量化产物见[量化专题](../quantization/README.md)，底层计算见 [Kernel](../kernels/README.md)，候选与接受见[投机推理](../speculative/README.md)。当前主体是项目内 Qwen3 单 GPU Runtime；实验覆盖范围与通用服务能力应分别判断。
