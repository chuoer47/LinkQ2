"""Page ownership, complete-prefix reuse and speculative reservation."""
import pytest

from qslab.runtime.state.block_manager import BlockManager
from qslab.runtime.state.sequence import Sequence


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(Sequence, 'block_size', 4)
    return BlockManager(5, 4)


def test_completed_prefix_reuses_full_pages_but_not_partial_tail(pool):
    first = Sequence(list(range(9)))
    pool.allocate(first, 0)
    first.num_scheduled_tokens = len(first)
    pool.hash_blocks(first)
    cached = first.block_table[:2]
    pool.deallocate(first)
    second = Sequence(list(range(8)) + [99])
    hits, needed = pool.can_allocate(second)
    assert (hits, needed) == (2, 3)
    pool.allocate(second, hits)
    assert second.block_table[:2] == cached
    assert second.num_cached_tokens == 8
    assert len(set(second.block_table)) == 3
    pool.deallocate(second)
    assert not pool.used_block_ids and len(pool.free_block_ids) == 5


def test_prefix_identity_includes_the_preceding_page(pool):
    first = Sequence(list(range(9)))
    pool.allocate(first, 0)
    first.num_scheduled_tokens = len(first)
    pool.hash_blocks(first)
    pool.deallocate(first)
    # An identical second page is not reusable under a different first page.
    other = Sequence([90, 91, 92, 93] + list(range(4, 9)))
    assert pool.can_allocate(other) == (0, 3)


def test_append_and_verify_reservations_do_not_leak_pages(pool):
    seq = Sequence([1, 2, 3, 4])
    pool.allocate(seq, 0)
    seq.append_token(5)
    assert pool.can_append(seq)
    pool.may_append(seq)
    allocated = list(seq.block_table)
    pool.may_append(seq)
    assert seq.block_table == allocated
    assert pool.can_verify(seq, 4)
    pool.reserve_verify(seq, 4)
    assert len(seq.block_table) == 3
    while len(seq.block_table) > seq.num_blocks:
        pool.release_tail(seq)
    pool.deallocate(seq)
    assert not pool.used_block_ids and len(pool.free_block_ids) == 5
    assert all(block.ref_count == 0 for block in pool.blocks)
