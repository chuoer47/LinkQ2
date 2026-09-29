"""Qslab runtime model primitives — vendored from nano-vllm (MIT), single-GPU."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x, residual=None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class Linear(nn.Module):
    """Plain linear with the weight_loader protocol nano-vllm's loader uses."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedLinear(nn.Module):
    """Two logical linears packed into one weight matrix (gate/up, QKV)."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        super().__init__()
        self.output_sizes = output_sizes
        self.input_size = input_size
        self.output_size = sum(output_sizes)
        self.weight = nn.Parameter(torch.empty(self.output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        self.bias = None
        self._offsets = [0]
        for s in output_sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        offset = self._offsets[loaded_shard_id]
        size = self.output_sizes[loaded_shard_id]
        param.data[offset:offset + size].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class QKVLinear(MergedLinear):
    """q/k/v packed into one matrix; shard ids are 0/1/2 -> q/k/v."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, bias: bool = False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        super().__init__(hidden_size, [self.q_size, self.kv_size, self.kv_size],
                         bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        offset = {0: 0, 1: self.q_size, 2: self.q_size + self.kv_size}[loaded_shard_id]
        size = {0: self.q_size, 1: self.kv_size, 2: self.kv_size}[loaded_shard_id]
        param.data[offset:offset + size].copy_(loaded_weight)


class VocabEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class LMHead(VocabEmbedding):
    """Output projection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
