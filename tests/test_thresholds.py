"""
The cost model behind the allow / review / deny thresholds (training/train.py).

Small hand-built score arrays where the best policy is obvious, so a mistake in the cost
arithmetic or in the threshold search shows up as a wrong answer, not a slightly worse number.
Needs the training dependencies (lightgbm, catboost); skipped in CI, which installs only
the API + test group.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from app.data.data_models import DecisionResponse

pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
from training.train import (  # noqa: E402
    pick_threshold,
    pick_thresholds,
    policy_cost,
    policy_outcome,
)

FD, RC = 10.0, 5.0   # false-decline cost, review cost


def _scores():
    """Three score groups of 100 orders each, every order worth $40:
         0.9 -> 80% fraud   (a decline pays: 20 x $10 = $200 vs reviewing 100 x $5 = $500)
         0.6 -> 25% fraud   (review pays: $500 vs $750 of wrong declines vs $1,000 lost)
         0.1 ->  2% fraud   (leave alone: $80 lost vs $500 to review)"""
    p = np.repeat([0.9, 0.6, 0.1], 100)
    y = np.concatenate([np.r_[np.ones(80), np.zeros(20)],
                        np.r_[np.ones(25), np.zeros(75)],
                        np.r_[np.ones(2), np.zeros(98)]]).astype(int)
    return y, p, np.full(300, 40.0)


def test_cost_arithmetic():
    y, p, v = _scores()
    # deny the 0.9 group, review the 0.6 group, allow the rest
    assert policy_cost(y, p, v, 0.6, 0.9, FD, RC) == 20 * FD + 100 * RC + 2 * 40
    # review catches only 80%: a fifth of the reviewed fraud dollars are lost as well
    assert policy_cost(y, p, v, 0.6, 0.9, FD, RC, catch_rate=0.8) == pytest.approx(
        20 * FD + 100 * RC + 2 * 40 + 0.2 * 25 * 40)
    # doing nothing loses every fraud
    assert policy_cost(y, p, v, 1.01, 1.01, FD, RC) == 107 * 40


def test_picks_deny_for_the_top_group_and_review_for_the_middle():
    y, p, v = _scores()
    review_t, deny_t = pick_thresholds(y, p, v, FD, RC)
    # The grid interpolates between scores, so any value that selects the same orders is
    # an equally good answer: check which groups fall in which band.
    assert 0.1 < review_t <= 0.6 < deny_t <= 0.9
    o = policy_outcome(y, p, v, review_t, deny_t, FD, RC)
    assert (o["denied"], o["reviewed"]) == (100, 100)
    assert (o["fraud_denied"], o["fraud_reviewed"], o["fraud_allowed"]) == (80, 25, 2)
    assert (o["false_declines"], o["legit_reviewed"]) == (20, 75)
    assert o["review_precision"] == 0.25 and o["deny_precision"] == 0.8


def test_review_never_starts_above_deny():
    y, p, v = _scores()
    for rc in (0.5, 5.0, 50.0):
        for cr in (1.0, 0.5, 0.0):
            review_t, deny_t = pick_thresholds(y, p, v, FD, rc, cr)
            assert review_t <= deny_t


def test_expensive_review_falls_back_to_the_single_deny_threshold():
    # When a review costs more than a wrong decline it is never worth using, and the
    # two-threshold search must give the same answer as the old one-threshold search.
    y, p, v = _scores()
    review_t, deny_t = pick_thresholds(y, p, v, FD, review_cost=50.0)
    assert review_t == deny_t
    single = pick_threshold(y, p, v, FD)
    assert ((p >= deny_t) == (p >= single)).all()
    assert policy_outcome(y, p, v, review_t, deny_t, FD, 50.0)["reviewed"] == 0


def test_useless_review_is_not_used():
    # A review that stops no fraud only adds cost.
    y, p, v = _scores()
    review_t, deny_t = pick_thresholds(y, p, v, FD, RC, catch_rate=0.0)
    assert policy_outcome(y, p, v, review_t, deny_t, FD, RC, 0.0)["reviewed"] == 0


def test_cheap_review_replaces_declines():
    # At $1 per review nothing is worth declining: even the 80% group costs less to
    # review ($100) than to decline ($200 of wrong declines).
    y, p, v = _scores()
    review_t, deny_t = pick_thresholds(y, p, v, FD, review_cost=1.0)
    o = policy_outcome(y, p, v, review_t, deny_t, FD, 1.0)
    assert o["denied"] == 0 and o["reviewed"] == 200


def test_search_matches_brute_force_on_random_scores():
    rng = np.random.default_rng(0)
    p = rng.random(400).round(2)
    y = (rng.random(400) < p * 0.5).astype(int)
    v = rng.integers(10, 100, 400).astype(float)
    grid = np.append(np.unique(np.quantile(p, np.linspace(0, 1, 401))), 1.01)
    for rc, cr in ((5.0, 1.0), (3.0, 0.8), (8.0, 0.6)):
        best = min(policy_cost(y, p, v, r, d, FD, rc, cr) for d in grid for r in grid if r <= d)
        review_t, deny_t = pick_thresholds(y, p, v, FD, rc, cr)
        assert policy_cost(y, p, v, review_t, deny_t, FD, rc, cr) == pytest.approx(best)


def test_response_model_accepts_only_the_three_decisions():
    base = dict(transaction_id="00000000-0000-0000-0000-000000000001",
                response_time="2025-03-14T12:00:00Z")
    for decision in ("allow", "review", "deny"):
        assert DecisionResponse(fraud_decision=decision, **base).fraud_decision == decision
    with pytest.raises(ValidationError):
        DecisionResponse(fraud_decision="maybe", **base)
