from collections import deque
import xxhash
import numpy as np

from qslab.runtime.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    #: Prefix reuse is ON. A hit hands `prepare_prefill` a shorter chunk plus
    #: a block_table; PagedAttention._materialize_prefix dequantizes the
    #: cached int4 prefix back to fp16 and feeds flash-attn the full-length
    #: K/V, repairing the M8 silent-wrong-answer bug (cu_seqlens_k claimed a
    #: length only the pool held, so flash-attn consumed misaligned suffix
    #: rows). A hit now differs from a cold run only by the int4 quantization
    #: error — the already accepted KV4 noise (notes/M9).
    ENABLE_PREFIX_CACHE = True

    def can_allocate(self, seq: Sequence) -> int:
        if not self.ENABLE_PREFIX_CACHE:
            # still refuse when the pool cannot take the whole sequence
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        need = len(seq) % self.block_size == 1 and len(seq.block_table) < seq.num_blocks
        return len(self.free_block_ids) >= need

    def may_append(self, seq: Sequence):
        # idempotent: verify steps (and draft syncs) trim the table to
        # num_blocks, which may already cover the block this step's query
        # lands in — allocate only when it is genuinely missing
        if len(seq) % self.block_size == 1 and len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._allocate_block())

    # --- speculative verify (design-m9 §3): reserve + trim ---
    # The verify forward stores the KV of all gamma+1 rows BEFORE attention
    # runs, so the block table must cover position L+gamma-1 up front. After
    # acceptance the table is trimmed back to the canonical length; rejected
    # slots live in blocks that go back to the free list with their garbage —
    # nobody reads past a row's own context length.

    def _verify_blocks(self, seq: Sequence, gamma: int) -> int:
        return (len(seq) + gamma - 1) // self.block_size + 1

    def can_verify(self, seq: Sequence, gamma: int) -> bool:
        need = self._verify_blocks(seq, gamma) - len(seq.block_table)
        return len(self.free_block_ids) >= max(0, need)

    def reserve_verify(self, seq: Sequence, gamma: int):
        required = self._verify_blocks(seq, gamma)
        while len(seq.block_table) < required:
            seq.block_table.append(self._allocate_block())

    def release_tail(self, seq: Sequence):
        """Free the last block of the table (used by post-verify trim)."""
        block_id = seq.block_table.pop()
        self.blocks[block_id].ref_count -= 1
        self._deallocate_block(block_id)

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
