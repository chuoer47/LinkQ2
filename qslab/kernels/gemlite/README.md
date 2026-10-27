# GemLite W4A16 算子源码

来源：GemLite 0.4.8，https://pypi.org/project/gemlite/0.4.8/ 。源包 SHA256：`a4513bacf7b718e13b86185f9b9f194afe1dc14dc1b4f01c7511dc823955f035`。许可见 LICENSE。

下列文件直接提取，内容与源包一致：`bitpack.py`、`dtypes.py`、`triton_kernels/utils.py`，以及三个 `*_int32packing.py` kernel。它们负责打包、M=1 GEMV、M=2–64 SplitK GEMM、M>64 GEMM。

本项目的 `config.py` 只保留这三条路径的默认 fast 配置；`core.py` 提供调优缓存；`ops.py` 选择并调用原始 kernel。在线模块位于 `qslab/runtime/backends/gemlite.py`。运行时不依赖 pip gemlite、HQQ 或上游高层模型封装。

模型加载时进行一次 W4 打包；forward 内按块反量化，不生成完整浮点权重。GEMM 使用 FP32 累加；M=1 GEMV 的归约精度由上游实现决定。
