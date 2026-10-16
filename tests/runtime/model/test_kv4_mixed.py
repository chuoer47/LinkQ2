"""Mixed varlen batch preserves KV4 decode's quantized-current-token semantics."""
from types import SimpleNamespace

import pytest
import torch

from qslab.runtime.execute.model_runner import ModelRunner
from qslab.runtime.model.attention import PagedAttention
from qslab.runtime.model.context import reset_context
from qslab.runtime.state.sequence import Sequence
from qslab.kernels.kv4 import store_kv_quant, materialize_kv


@pytest.mark.gpu
@torch.inference_mode()
def test_decode_and_prefill_use_their_own_kv_semantics():
    torch.manual_seed(17)
    Sequence.block_size = 256
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.config = SimpleNamespace(kv_cache_dtype='kv4')
    decode, prefill = Sequence([1] * 6), Sequence([2] * 3)
    decode.num_cached_tokens, decode.num_scheduled_tokens = 5, 1
    decode.block_table, decode.is_prefill = [1], False
    prefill.block_table, prefill.num_scheduled_tokens = [2], 3
    attn = PagedAttention(2, 64, 64**-0.5, 1, block_n=256)
    attn.k_cache = (torch.zeros(1024, 1, 8, device='cuda', dtype=torch.uint32),
                    torch.full((1, 64), 0.2, device='cuda', dtype=torch.float16))
    attn.v_cache = (torch.zeros_like(attn.k_cache[0]),
                    torch.zeros(1024, 1, 1, device='cuda', dtype=torch.float16))
    prefix_k = torch.randn(5, 1, 64, device='cuda', dtype=torch.float16) * 0.5
    prefix_v = torch.randn_like(prefix_k)
    slots = torch.arange(256, 261, device='cuda', dtype=torch.int32)
    store_kv_quant(prefix_k, prefix_v, attn.k_cache, attn.v_cache, slots)
    q = torch.randn(4, 2, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(4, 1, 64, device='cuda', dtype=torch.float16) * 0.5
    v = torch.randn_like(k)
    runner.prepare_prefill([decode, prefill])
    try:
        actual = attn(q, k, v)
        dk, dv = materialize_kv(attn.k_cache, attn.v_cache,
            torch.arange(256, 262, device='cuda', dtype=torch.int32))

        def reference(query, keys, values):
            keys, values = keys.float().expand(-1, 2, -1), values.float().expand(-1, 2, -1)
            scores = torch.einsum('hd,thd->ht', query.float(), keys) * 64**-0.5
            return torch.einsum('ht,thd->hd', scores.softmax(-1), values).half()

        expected = [reference(q[0], dk, dv)]
        for index in range(3):
            expected.append(reference(q[index + 1], k[1:index + 2], v[1:index + 2]))
        torch.testing.assert_close(actual, torch.stack(expected), atol=0.002, rtol=0.002)
    finally:
        reset_context()
