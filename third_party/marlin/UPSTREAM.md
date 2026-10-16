# Marlin 源码

来源：[IST-DASLab/marlin](https://github.com/IST-DASLab/marlin/tree/1f25790bdd49fba53106164a24666dade68d7c90)。
固定提交：`1f25790bdd49fba53106164a24666dade68d7c90`，官方包版本 `0.1.1`。

仅提取官方安装脚本、Python 算子接口、C++ binding、CUDA kernel、测试和 LICENSE。
这六个文件保留原始内容，不包含 GPTQ 算法、数据工具或其他项目代码。

通过原 `setup.py` 构建并安装到项目 `.conda`，调用原 `marlin.Layer.pack()` 和 `Layer.forward()`。
项目后端只负责准备临时 FP16 fake 权重、传入原 scale，以及执行项目 AWQ 产物定义的输入缩放。
详细构建和验证入口见 [benchmark README](../../benchmarks/marlin/README.md)。
