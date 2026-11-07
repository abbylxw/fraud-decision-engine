"""
Tests for training/build_features.spend_spike_features(): today's spend vs the median
daily spend on the user's active days (training-only feature).
"""
import pandas as pd
import pytest

from training.build_features import spend_spike_features

TODAY = pd.Timestamp("2020-03-31 12:00")


def _frame(rows):
    """rows = [(user_id, timestamp, amount), ...] in time order."""
    return pd.DataFrame(rows, columns=["user_id", "purchase_time", "purchase_value"])


def _history(user="u1"):
    # Active days 20, 18, 16, 12 and 10 days ago: median daily spend = $40.
    return [(user, TODAY - pd.Timedelta(days=d), amt)
            for d, amt in [(20, 20.0), (18, 35.0), (16, 40.0), (12, 50.0), (10, 60.0)]]


def test_worked_example():
    # $60 earlier today + $150 now = $210 today; 210 / 40 = 5.25.
    df = _frame(_history() + [("u1", TODAY - pd.Timedelta(hours=2), 60.0),
                              ("u1", TODAY, 150.0)])
    f = spend_spike_features(df).iloc[-1]
    assert f["spend_ratio_day_vs_median_30d"] == pytest.approx(5.25)
    assert f["spend_spike_3x_30d"] == 1
    assert f["spend_spike_5x_30d"] == 1
    assert f["spend_ratio_day_vs_median_90d"] == pytest.approx(5.25)


def test_window_ignores_older_days():
    # Only 2 of the 5 active days fall inside the last 14 days -> no baseline.
    df = _frame(_history() + [("u1", TODAY, 150.0)])
    f = spend_spike_features(df).iloc[-1]
    assert f["spend_ratio_day_vs_median_14d"] == -1
    assert f["spend_spike_3x_14d"] == 0
    assert f["spend_ratio_day_vs_median_30d"] == pytest.approx(150 / 40)
    assert f["spend_spike_3x_30d"] == 1
    assert f["spend_spike_5x_30d"] == 0


def test_today_counts_current_and_earlier_txns_but_not_in_baseline():
    df = _frame(_history() + [("u1", TODAY - pd.Timedelta(hours=1), 10.0),
                              ("u1", TODAY, 10.0)])
    f = spend_spike_features(df)
    # First txn today: $10 / 40. Second: earlier $10 + $10 = $20 / 40 (baseline unchanged).
    assert f["spend_ratio_day_vs_median_30d"].iloc[-2] == pytest.approx(0.25)
    assert f["spend_ratio_day_vs_median_30d"].iloc[-1] == pytest.approx(0.5)


def test_too_few_active_days_means_no_baseline():
    df = _frame(_history()[:4] + [("u1", TODAY, 1000.0)])
    f = spend_spike_features(df).iloc[-1]
    for w in (14, 30, 90):
        assert f[f"spend_ratio_day_vs_median_{w}d"] == -1
        assert f[f"spend_spike_3x_{w}d"] == 0
        assert f[f"spend_spike_5x_{w}d"] == 0


def test_utc_day_boundary_starts_a_new_day():
    midnight = pd.Timestamp("2020-03-31 00:00")
    df = _frame([("u1", midnight - pd.Timedelta(days=d), 40.0) for d in range(6, 1, -1)]
                + [("u1", midnight - pd.Timedelta(minutes=1), 400.0),   # 23:59, previous day
                   ("u1", midnight + pd.Timedelta(minutes=1), 40.0)])   # 00:01, new day
    f = spend_spike_features(df).iloc[-1]
    # 00:01 starts a new day: today = $40. The $400 day is now in the baseline:
    # active days $40 x5 + $400 -> median $40 -> ratio 1.
    assert f["spend_ratio_day_vs_median_14d"] == pytest.approx(1.0)


def test_users_are_independent():
    df = _frame(_history("u1") + [("u2", TODAY, 500.0)])
    assert spend_spike_features(df).iloc[-1]["spend_ratio_day_vs_median_90d"] == -1
