"""M10: the acceptance rule behind temperature-aware speculation.

The engine's prefix walk (scheduler.postprocess_verify) is only correct if a
rejected row can never return its own draft token, which is what
``ModelRunner.rejection_verify`` provides. These tests pin the rule itself —
Leviathan's ``min(1, q/p)`` accept plus the ``norm(max(0, q - p))``
resample — rather than any engine output, because the losslessness claim is a
statement about this function's output distribution:

  - drawing proposals from p and running the rule reproduces q exactly
    (total variation against the target's own q, with a control assertion so a
    biased rule cannot pass by both being wrong in the same direction);
  - a deterministic proposal is the p = one-hot case of the same rule, which is
    where the n-gram/lookahead proposers live;
  - greedy and non-speculative rows are returned untouched, which is what keeps
    every M9 greedy test measuring the thing it always measured.

The adaptive window policy (migrated from the frozen DynamicMode) is a pure
function of the recent acceptance history, so it is exercised through a
duck-typed scheduler slice instead of a loaded model.
"""
from collections import deque

import pytest
import torch

from qslab.runtime.model_runner import ModelRunner
from qslab.runtime.scheduler import Scheduler

V = 64
N = 20000
T = 0.9


def _q_from(logits):
    return torch.softmax(logits / T, dim=-1)


def _tv(p, q):
    return 0.5 * (p - q).abs().sum().item()


def _hist(tokens, vocab=V):
    return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()


def _run(logits, temps, drafts, probs, sampled):
    return ModelRunner.rejection_verify(logits, temps, drafts, probs, sampled)


def test_ratio_acceptance_reproduces_the_target_distribution():
    """The output histogram equals the target's q, whatever p proposed."""
    torch.manual_seed(0)
    row_logits = torch.randn(V)
    logits = row_logits.unsqueeze(0).expand(N, V).contiguous()
    draft_logits = torch.randn(V) * 2.0          # a proposer with a real opinion
    p = torch.softmax(draft_logits / T, dim=-1)
    drafts = torch.multinomial(p.unsqueeze(0).expand(N, V).contiguous(), 1).squeeze(1)
    draft_probs = p.unsqueeze(0).expand(N, V).contiguous()
    temps = torch.full((N,), T)
    out = _run(logits, temps, drafts, draft_probs, torch.zeros(N, dtype=torch.long))

    q = _q_from(row_logits)
    assert _tv(_hist(out), q) < 0.06, "speculated output drifted off the target q"
    # power check: a rule that simply kept every proposal would score the TV
    # between p and q here, and that gap has to be big enough to notice
    assert _tv(p, q) > 0.3, "test setup: p too close to q to detect a bias"


def test_one_hot_proposal_is_the_same_rule():
    """draft_probs=None means p = delta_x: accept with probability q(x).

    That is the n-gram/lookahead case, and the theorem says it is still
    exactly q — the resample must renormalize q with x's own mass removed.
    """
    torch.manual_seed(1)
    row_logits = torch.randn(V)
    logits = row_logits.unsqueeze(0).expand(N, V).contiguous()
    q = _q_from(row_logits)
    # proposals from a third distribution: the rule must not care who proposed
    drafts = torch.multinomial(torch.softmax(torch.randn(V), 0)
                               .unsqueeze(0).expand(N, V).contiguous(), 1).squeeze(1)
    out = _run(logits, torch.full((N,), T), drafts, None,
               torch.zeros(N, dtype=torch.long))
    assert _tv(_hist(out), q) < 0.06


def test_rejected_row_never_returns_its_own_draft():
    """The prefix walk's only requirement on the rule above it."""
    torch.manual_seed(2)
    row_logits = torch.full((V,), -20.0)
    row_logits[3] = 20.0                      # q(7) ~ 0: token 7 must be rejected
    logits = row_logits.unsqueeze(0).expand(512, V).contiguous()
    drafts = torch.full((512,), 7, dtype=torch.long)
    out = _run(logits, torch.full((512,), T), drafts, None,
               torch.zeros(512, dtype=torch.long))
    assert not bool((out == 7).any()), "a rejected row came back as its draft"
    assert _tv(_hist(out, V), _q_from(row_logits)) < 0.02


def test_greedy_and_non_speculative_rows_pass_through():
    """Greedy keeps the plain sample (argmax-equality as always), and so does
    every row that carries no proposal — the bonus row and the padded tail."""
    torch.manual_seed(3)
    logits = torch.randn(4, V)
    sampled = torch.tensor([5, 6, 7, 8])
    temps = torch.tensor([1e-6, 1e-6, 0.8, 0.8])
    drafts = torch.tensor([0, 1, -1, 2])
    probs = torch.softmax(torch.randn(4, V), dim=-1)
    out = _run(logits, temps, drafts, probs, sampled)
    assert out[:3].tolist() == [5, 6, 7], "greedy/padded rows were modified"


@pytest.mark.gpu
def test_draft_rows_align_with_the_verify_window():
    """Row m of the flattened batch is the decision for drafts[m] — the same
    indexing postprocess_verify walks, so a shift here would accept tokens the
    target never produced."""
    from qslab.runtime.sequence import Sequence
    from qslab.runtime.sampling_params import SamplingParams

    def s(drafts):
        seq = Sequence([1, 2, 3], SamplingParams(temperature=1e-6, max_tokens=8))
        seq.spec_drafts = drafts
        return seq

    M = 5
    flat = ModelRunner._flat_drafts([s([11, 12]), s([])], M).tolist()
    assert flat == [11, 12, -1, -1, -1, -1, -1, -1, -1, -1]


class _SchedSlice:
    """The four attributes _adapt_gamma reads/writes, no model on disk."""
    gamma = 4
    _adapt_gamma = Scheduler._adapt_gamma

    def __init__(self):
        self._recent = deque(maxlen=4)
        self.proposal_gamma = self.gamma


def _drive(hist):
    s = _SchedSlice()
    seen = []
    for a in hist:
        s._recent.append(a)
        s._adapt_gamma()
        seen.append(s.proposal_gamma)
    return seen


def test_adaptive_gamma_shrinks_and_never_grows_back():
    assert _drive([0.0, 0.0, 0.0])[-1] == 2, "a window that lands nothing must halve"
    # Regrowth is deliberately absent, so these drives assert the ratchet:
    # acceptance is prefix-based, and a round that lands 2/2 after a halve
    # carries no information about a 3rd or 4th draft. Measured over 4 repeats
    # on 8B natural (M10, results/m10_draft_sweep.txt): holding the halved
    # window 1.00-1.18x, a regrow rule 0.90-1.11x, pinned ceiling no
    # adaptation 0.91x. The means (1.07x vs 0.98x) are read as trajectory
    # divergence, not as a saving — a repeat of one cell lands anywhere in
    # 0.90-1.05 (results/m10_adaptive_tune.txt), and a window cut saves no
    # time anyway because the proposer loop still runs at configured gamma.
    assert _drive([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])[-1] == 2, "halving is one-way"
    assert _drive([0.0, 0.0, 0.0, 4.0, 4.0, 4.0])[-1] == 2
    # the ceiling is the configured gamma: M is baked into the verify graph
    # family, so adaptation may only spend fewer draft forwards
    assert _drive([4.0] * 6)[-1] == 4


def test_adaptive_window_is_a_false_positive_filter():
    """WINDOW is not a reaction-speed knob — γ//2 is an idempotent target, so a
    wider window only delays the one irreversible cut. Measured on copy
    (results/m10_adaptive_tune.txt): a 1-round window fired on a single
    lone 1-of-4 round and cost 27% of throughput, 2 rounds cost 24%; the
    shipped 3-round window ignored that same round and never fired in 30 draws."""
    assert _drive([4.0, 4.0, 1.0])[-1] == 4, "one bad round among three is not a trend"
    assert _drive([1.0, 1.0, 1.0])[-1] == 2, "three rounds at break-even must cut"


def test_adaptive_gamma_never_reaches_zero():
    """gamma=1 is still a speculation round; 0 would divide by zero upstream."""
    assert _drive([0.0] * 6)[-1] >= 1
    s = _SchedSlice()
    s.gamma = 1
    s.proposal_gamma = 1
    s._recent.append(0.0)
    s._adapt_gamma()
    assert s.proposal_gamma == 1
