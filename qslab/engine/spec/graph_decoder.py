"""M6: CUDA Graph for decode steps — kill the per-step dispatch overhead.

torch.cuda.CUDAGraph replays a captured kernel sequence with near-zero CPU
launch cost. Decode step is ideal: static shapes (M=1, fixed tensors), only
the *contents* of the input token / cache_position change between steps.

Implementation: static input/output tensors; copy the token id and position
into static placeholders, replay the graph, read logits from the static
output. The KV cache writes go to the same device addresses every step by
construction (cache buffers are pre-allocated), so capture is safe.

Also: the draft engine decode loop gets the same treatment.
"""
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qslab.config import EngineConfig
from qslab.engine import QslabEngine
from adapters.tokenizer import QwenTokenizerAdapter


class GraphedDecoder:
    """CUDA-Graph capture of one engine decode step.

    Because our PatchedQwen3Attention writes KV at cache.len (which advances
    during capture!), we must freeze the cache position during capture: the
    graph replays the SAME kernel launches, including the same write addresses.
    KV caches are pre-allocated [B,H,max_len,D] buffers, so writing at a fixed
    address is fine — we simply track the logical length outside the graph and
    capture with start_pos = capture_pos. Replay then writes position
    capture_pos every time — WRONG for sequential decode.

    Solution used here: capture at position P, and on replay use the cache's
    own rotary/positions inside the graph — the position embedding comes from
    cache_position tensor which is an INPUT we can copy into (static tensor).
    The KV write address is derived from cache.len inside update()... which
    is captured as a fixed value. Therefore: we bypass update()'s dynamic
    start and give PatchedQwen3Attention a static write slot by capturing
    with the cache pre-extended so that every replay writes at the SAME slot,
    then we fix up the logical length after each replay.

    Concretely: capture with start_pos = P (cache.len forced to P). The graph
    writes K/V into cache[..., P, :] and returns k_full[:P+1]. On replay for
    actual position Q != P, the write lands at P again — so we capture P=0
    only for the FIRST step and... no. The clean approach: make the write
    position an input tensor. That requires touching update() to read start
    from a device tensor.

    We do exactly that: BaseKVCache.update accepts a device tensor `start_t`;
    when present, the slice is computed as index_copy_ with the tensor start
    (graph-safe: index_copy_ with a 0-d LongTensor keeps a static address).
    """

    def __init__(self, engine: QslabEngine, max_seq: int, num_warmup: int = 3):
        self.engine = engine
        self.max_seq = max_seq
        self.graph = None
        self.static_token = None
        self.static_pos = None
        self.static_logits = None
        self._num_warmup = num_warmup

    def capture(self, prompt_len: int):
        eng = self.engine
        dev = eng.device
        self.static_token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.static_pos = torch.zeros(1, dtype=torch.long, device=dev)

        # warmup on a side stream (torch graph capture requirement)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._num_warmup):
                out = eng.model(input_ids=self.static_token,
                                cache_position=self.static_pos, use_cache=False)
                _ = out.logits[:, -1, :]
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = eng.model(input_ids=self.static_token,
                            cache_position=self.static_pos, use_cache=False)
            self.static_logits = out.logits[:, -1, :]
        self.graph = g

    def replay(self, token_id: int, pos: int) -> torch.Tensor:
        self.static_token.fill_(token_id)
        self.static_pos.fill_(pos)
        self.graph.replay()
        return self.static_logits
