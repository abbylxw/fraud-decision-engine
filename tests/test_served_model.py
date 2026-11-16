"""
The model file in app/decisions/ must work the way the API uses it.

Every other test replaces the model with a mock or a fixed score, so a model.joblib that
loads but cannot predict would pass the whole suite. That happened once (2026-11-15: a
monotone constraint saved in a form XGBoost rejects after set_params). These tests use the
real file and the real scoring function.
"""
import json
import pathlib

import pytest

from app.data.data_models import Features
from app.decisions import models
from app.decisions.models import REVIEW_THRESHOLD, THRESHOLD, score

DECISIONS = pathlib.Path(models.__file__).parent


def _features(sample_transaction, **overrides) -> Features:
    return Features(**sample_transaction.model_dump(by_alias=False)).model_copy(update=overrides)


def test_served_model_scores_a_request(sample_transaction):
    s = score(_features(sample_transaction))
    assert 0.0 <= s <= 1.0


def test_model_files_agree_with_each_other():
    meta = json.loads((DECISIONS / "model_meta.json").read_text())
    features = json.loads((DECISIONS / "model_features.json").read_text())
    assert meta["features"] == features
    assert models._model.n_features_in_ == len(features)
    assert REVIEW_THRESHOLD <= THRESHOLD


def test_more_other_accounts_never_lowers_the_score(sample_transaction):
    meta = json.loads((DECISIONS / "model_meta.json").read_text())
    if "did_other_uid_all" not in meta["features"]:
        pytest.skip("served model does not use did_other_uid_all")
    scores = [score(_features(sample_transaction, did_other_uid_all=n)) for n in (0, 1, 2, 5, 20, 100)]
    assert scores == sorted(scores), scores
    assert scores[-1] > scores[0]


def test_monitoring_reference_matches_the_model():
    meta = json.loads((DECISIONS / "model_meta.json").read_text())
    ref = meta.get("monitoring_reference")
    if ref is None:
        pytest.skip("served model was shipped before drift monitoring existed")
    for name, buckets in {"score": ref["score"], **ref["features"]}.items():
        assert len(buckets["shares"]) == len(buckets["edges"]) + 1, name
        assert buckets["edges"] == sorted(buckets["edges"]), name
        assert abs(sum(buckets["shares"]) - 1) < 1e-3, name
    assert set(ref["features"]) <= set(meta["features"])


def test_no_audited_out_feature_is_a_model_input():
    """docs/feature_audit.md: these count the buyer's own purchases or raw traffic volume.
    They are still computed and stored, but must not come back as model inputs."""
    features = json.loads((DECISIONS / "model_features.json").read_text())
    banned = [f for f in features
              if f.startswith(("did_txn_cnt_", "country_txn_count_")) or f == "did_txn_cnt_all"]
    assert not banned


def test_a_customers_own_history_does_not_raise_their_score(sample_transaction):
    """A repeat customer on their own device: more of their OWN earlier purchases (device
    transaction counts, purchase counts, a ratio of 1 to their own average) must not move
    the score. Before the audit the same inputs made a loyal customer look risky."""
    new = _features(sample_transaction)
    loyal = _features(sample_transaction, did_txn_cnt_30d=6, did_txn_cnt_7d=2, did_txn_cnt_all=40,
                      purchase_cnt_30d=6, purchase_amt_30d=600.0, purchase_cnt_90d=15,
                      amount_vs_user_avg_30d=1.0, time_since_last_txn_uid=86400.0,
                      country_txn_count_30d=50_000)
    assert score(loyal) == pytest.approx(score(new), abs=1e-6)
