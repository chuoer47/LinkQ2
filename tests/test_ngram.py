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
