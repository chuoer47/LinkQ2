"""NGramProposer: pure-CPU lookup semantics (design-m9 §1, L3)."""
from qslab.runtime.ngram import NGramProposer


def test_repeated_pattern_proposes_followers():
    p = NGramProposer(3, 4)
    # key [1,2,3] matches at 0 and 4; the most recent (4) is the better
    # predictor, so the proposal is what followed THAT occurrence
    assert p.propose([1, 2, 3, 9, 1, 2, 3, 1, 2, 3]) == [1, 2, 3]


def test_no_match_returns_empty():
    p = NGramProposer(3, 4)
    assert p.propose([1, 2, 3, 4, 5, 6, 7, 8]) == []


def test_too_short_sequences():
    p = NGramProposer(3, 4)
    assert p.propose([1, 2, 3]) == []
    assert p.propose([1, 2]) == []
    assert p.propose([]) == []


def test_self_match_excluded():
    # the key is the tail itself; a self-match would propose nothing new
    p = NGramProposer(3, 4)
    assert p.propose([5, 5, 5, 1, 2, 3]) == []


def test_followers_run_to_end_of_history():
    p = NGramProposer(3, 4)
    assert p.propose([1, 2, 3, 9, 1, 1, 2, 3]) == [9, 1, 1, 2]


def test_gamma_truncation():
    # occurrence at 3 has only 3 followers left in history
    p = NGramProposer(3, 4)
    assert p.propose([9, 9, 9, 1, 2, 3, 1, 2, 3]) == [1, 2, 3]


def test_most_recent_occurrence_wins():
    q = NGramProposer(1, 2)
    # '7' at 0 -> '1', at 2 -> '2'; the last occurrence is at 2
    assert q.propose([7, 1, 7, 2, 7]) == [2, 7]


# ---------------- LookaheadProposer (migrated from the frozen M6 mode) -------

def test_lookahead_first_occurrence_wins():
    from qslab.runtime.ngram import LookaheadProposer
    p = LookaheadProposer(3, 4)
    # same history, opposite bet from the plain lookup above: the FIRST
    # occurrence of (1,2,3) is the one this proposer caches and uses
    assert p.propose([1, 2, 3, 10, 1, 2, 3, 11, 1, 2, 3], seq_id=1) == [10, 1, 2, 3]
    assert NGramProposer(3, 4).propose([1, 2, 3, 10, 1, 2, 3, 11, 1, 2, 3]) == [11, 1, 2, 3]


def test_lookahead_chain_extension():
    from qslab.runtime.ngram import LookaheadProposer
    # the matched span at 0 has only three followers before the text moves on,
    # so a plain lookup stops at 3 of gamma=6; chaining keeps walking the window
    # through (7,8,9) -> (8,9,1) -> (9,1,2) and bridges the gap
    p = LookaheadProposer(3, 6, span=4)
    assert p.propose([1, 2, 3, 7, 8, 9, 1, 2, 3], seq_id=1) == [7, 8, 9, 1, 2, 3]


def test_lookahead_index_is_per_sequence():
    """Batch composition must not leak into a proposal.

    A single shared index would let sequence 2's tail match sequence 1's text
    and propose from it — output depending on who else is in the batch.
    """
    from qslab.runtime.ngram import LookaheadProposer
    p = LookaheadProposer(3, 4)
    assert p.propose([1, 2, 3, 10, 1, 2, 3], seq_id=1) == [10, 1, 2, 3]
    assert p.propose([1, 2, 3, 77, 1, 2, 3], seq_id=2) == [77, 1, 2, 3]


def test_lookahead_drop_seqs_clears_the_index():
    from qslab.runtime.ngram import LookaheadProposer
    p = LookaheadProposer(3, 4)
    assert p.propose([1, 2, 3, 77, 1, 2, 3], seq_id=2) == [77, 1, 2, 3]
    p.drop_seqs([2])
    # same id, new sequence: the released tail must not come back as a proposal
    assert p.propose([5, 5, 5, 1, 2, 3], seq_id=2) == []
