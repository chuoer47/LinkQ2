"""Construction of online quantized Linear implementations."""


def make_quant_linear(backend, values, scale, **kwargs):
    if backend != "gemlite":
        raise ValueError(f"unknown quantization backend: {backend}")
    from .gemlite import GemLiteLinear
    return GemLiteLinear.from_quantized(values, scale, **kwargs)
