"""W4A16 Linear backed by the project-local GemLite Triton kernels."""
import torch

from qslab.runtime.backends.base import QuantLinear


class GemLiteLinear(QuantLinear):
    @classmethod
    @torch.no_grad()
    def from_quantized(cls, values, scale, *, group_size, symmetric,
                       zero_point=None, input_scale=None, device, dtype):
        from qslab.kernels.gemlite.bitpack import pack_weights_over_cols_triton
        from qslab.kernels.gemlite.dtypes import DType
        if dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("GemLite W4A16 requires FP16 or BF16 activations")
        n, k = values.shape
        if group_size < 32 or group_size & (group_size - 1) or k % group_size or n % 32:
            raise ValueError(f"unsupported W4A16 shape/group: N={n}, K={k}, group={group_size}")
        layer = cls()
        layer.in_features, layer.out_features = k, n
        layer.group_size, layer.compute_dtype = group_size, dtype
        codes = values.to(device=device, dtype=torch.int16)
        codes = (codes + 8 if symmetric else codes).to(torch.uint8)
        packed, _ = pack_weights_over_cols_triton(codes, 4, 32, transpose=True)
        layer.register_buffer("packed", packed.contiguous())
        layer.register_buffer("scales", scale.to(device=device).t().contiguous())
        zeros = torch.tensor(8, dtype=torch.int32, device=device) if symmetric else zero_point.to(device).t().contiguous()
        layer.register_buffer("zeros", zeros)
        layer.register_buffer("input_scale", None if input_scale is None else input_scale.to(device=device, dtype=dtype))
        kind = DType.FP16 if dtype == torch.float16 else DType.BF16
        # Same metadata as upstream pack(fma_mode=False), with FP32 GEMM accumulation.
        # Upstream GEMV chooses its own reduction precision.
        layer._meta = (4, group_size, 15, 8, kind.value, kind.value,
                       DType.FP32.value, DType.FP16.value, 0, 3, True, 104)
        return layer

    def forward(self, x):
        from qslab.kernels.gemlite.ops import linear
        if x.dtype != self.compute_dtype:
            raise TypeError(f"expected {self.compute_dtype}, got {x.dtype}")
        if self.input_scale is not None:
            x = x / self.input_scale
        return linear(x, self.packed, self.scales, self.zeros, self._meta)
