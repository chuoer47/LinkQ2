"""Construction of online quantized Linear implementations."""


def make_quant_linear(backend, values, scale, **kwargs):
    if backend == 'gemlite':
        from .gemlite import GemLiteLinear
        implementation = GemLiteLinear
    elif backend == 'marlin':
        from .marlin import MarlinLinear
        implementation = MarlinLinear
    else:
        raise ValueError(f"unknown quantization backend: {backend}")
    return implementation.from_quantized(values, scale, **kwargs)
