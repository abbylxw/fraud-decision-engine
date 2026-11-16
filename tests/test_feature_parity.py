"""
Training-serving parity for velocity features.

The model is trained on features computed by training/build_features.py (a pandas
replay of the CSV). In production the same features are computed by
app/decisions/features/build_features() from Redis. If the two disagree, the model
sees inputs it was never trained on - no error, just silently worse decisions.

This test feeds the SAME transactions, in time order, through both code paths and
checks every numeric feature matches. Serving runs against an in-memory fake Redis
(fakeredis), so no real Redis or Postgres is needed.

What it caught when first written (2026-11-05): serving counted velocity windows back
from the server clock (time.time()) while training counts back from purchase_time, so
did_uid_*, did_txn_cnt_*, ip_uid_* and country_txn_count_* all disagreed.
"""
import datetime as dt
import random

import numpy as np
import pandas as pd
import pytest
from fakeredis import aioredis as fake_aioredis

import app.decisions.features as serving
from app.data.data_models import Transaction
from app.decisions.features import build_features, update_velocity_counters
from app.decisions.features.request import event_ts
from training.build_features import DATA_DIR, build_feature_frame, load_transactions

# Features that are deliberately not compared:
#   did_uid_days_on_file - comes from Postgres (stubbed here), not Redis.
NOT_COMPARED = {"did_uid_days_on_file"}

# Window edges in seconds, used to plant transactions exactly on boundaries.
EDGES = [5 * 60, 10 * 60, 3600, 2 * 3600, 24 * 3600, 7 * 86400, 30 * 86400, 90 * 86400]
COUNTRIES = ["United States", "China", "Japan", "Unknown"]


# ---------------------------------------------------------------------------
# Synthetic data: shared devices/IPs, repeat users, boundary timestamps
# ---------------------------------------------------------------------------

def synthetic_transactions(n=1500, seed=7) -> pd.DataFrame:
    rng = random.Random(seed)
    start = dt.datetime(2015, 1, 1)
    users = [f"u{i}" for i in range(500)]
    devices = [f"d{i}" for i in range(120)]    # fewer devices than users -> sharing
    ips = [1_000_000 + i for i in range(150)]   # fewer IPs than users -> sharing

    times = sorted(rng.uniform(0, 120 * 86400) for _ in range(n))
    # Plant exact window edges: a later row exactly EDGE seconds after an earlier one,
    # reusing the same device/IP/user so the boundary actually matters.
    rows = []
    for i, t in enumerate(times):
        rows.append({
            "user_id": rng.choice(users),
            "device_id": rng.choice(devices),
            "ip_address": rng.choice(ips),
            "secs": t,
        })
    for k, edge in enumerate(EDGES):
        base = rows[k * 7]
        rows.append({**base, "secs": base["secs"] + edge})       # exactly on the edge
        # A second purchase in that same second: the first one's 90-day purge must
        # not delete the entry that is exactly on the edge before this one is scored.
        rows.append({**base, "secs": base["secs"] + edge})
        rows.append({**base, "secs": base["secs"] + edge + 1})   # one second past it
    # Two purchases in the same second on the same device (collision check).
    rows.append({**rows[50], "secs": rows[50]["secs"]})

    df = pd.DataFrame(rows).sort_values("secs", kind="stable").reset_index(drop=True)
    # Whole seconds, like the real data; naive datetimes, like the client sends.
    df["purchase_time"] = [start + dt.timedelta(seconds=int(s)) for s in df["secs"]]
    df["signup_time"] = [
        p - dt.timedelta(seconds=rng.randint(120, 200 * 86400)) for p in df["purchase_time"]
    ]
    df["purchase_value"] = [rng.randint(9, 150) for _ in range(len(df))]
    df["source"] = [rng.choice(["SEO", "Ads", "Direct"]) for _ in range(len(df))]
    df["browser"] = [rng.choice(["Chrome", "IE", "Safari", "FireFox", "Opera"])
                     for _ in range(len(df))]
    df["sex"] = [rng.choice(["M", "F"]) for _ in range(len(df))]
    df["age"] = [rng.randint(18, 70) for _ in range(len(df))]
    df["class"] = [int(rng.random() < 0.1) for _ in range(len(df))]
    df["ip_country"] = [COUNTRIES[ip % len(COUNTRIES)] for ip in df["ip_address"]]
    return df.drop(columns="secs")


# ---------------------------------------------------------------------------
# Serving replay
# ---------------------------------------------------------------------------

async def serving_features(df: pd.DataFrame, monkeypatch) -> pd.DataFrame:
    """Run each row through the real build_features() + update_velocity_counters()."""
    country_by_ip = dict(zip(df["ip_address"].astype("int64"), df["ip_country"]))

    async def fake_ip_country(ip, db):
        return country_by_ip.get(int(ip), "Unknown")

    async def fake_days_on_file(device_id, user_id, db, as_of_date=None):
        return 0

    async def empty_purchase_history(user_id, db, as_of=None):
        # The DB fallback only runs when Redis has no purchase history for this user,
        # which in a sequential replay means the user has none -> all zeros.
        windows = ["5min", "10min", "1h", "24h", "7d", "30d", "90d"]
        out = {f"{p}_{w}": 0 for p in ("cnt", "amt", "success_cnt", "success_amt")
               for w in windows}
        out["median_amt_90d"] = 0.0
        return out

    monkeypatch.setattr(serving, "get_ip_country", fake_ip_country)
    monkeypatch.setattr(serving, "get_did_days_on_file", fake_days_on_file)
    async def no_device_history(device_id, db, as_of=None):
        # Same reasoning: no device history in Redis in a sequential replay = a new device.
        return {}

    monkeypatch.setattr(serving, "_purchase_counts_from_db", empty_purchase_history)
    monkeypatch.setattr(serving, "_device_history_from_db", no_device_history)

    redis = fake_aioredis.FakeRedis(decode_responses=True)
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        t = Transaction(
            user_id=str(r.user_id), signup_time=r.signup_time, purchase_time=r.purchase_time,
            purchase_value=float(r.purchase_value), device_id=str(r.device_id),
            source=r.source, browser=r.browser, sex=r.sex, age=int(r.age),
            ip_address=int(r.ip_address),
        )
        f = await build_features(t, db=None, redis=redis)
        out.append(f.model_dump())
        # Production fires this off after responding; here we await it so the next
        # transaction sees this one, exactly like the training replay does.
        # "Successful" purchase = not fraud, matching training's class == 0.
        await update_velocity_counters(
            user_id=t.user_id, device_id=t.device_id, ip=t.ip_address,
            ip_country=f.ip_country, transaction_id=f"txn-{i}",
            purchase_value=t.purchase_value,
            fraud_decision="deny" if r[df.columns.get_loc("class")] == 1 else "allow",
            event_ts=event_ts(t), redis=redis,
        )
    await redis.aclose()
    return pd.DataFrame(out)


def assert_parity(train: pd.DataFrame, serve: pd.DataFrame):
    common = [
        c for c in train.columns
        if c in serve.columns and c not in NOT_COMPARED
        and pd.api.types.is_numeric_dtype(serve[c])
    ]
    assert len(common) > 40, f"expected to compare most features, only found {len(common)}"
    problems = []
    for c in common:
        a = train[c].to_numpy(dtype=float)
        b = serve[c].to_numpy(dtype=float)
        bad = ~np.isclose(a, b, rtol=1e-9, atol=1e-6)
        if bad.any():
            i = int(np.argmax(bad))
            problems.append(f"{c}: {bad.sum()} rows differ, first at row {i} "
                            f"(training={a[i]}, serving={b[i]})")
    assert not problems, "training/serving feature mismatch:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_parity_on_synthetic_data(monkeypatch):
    df = synthetic_transactions()
    assert_parity(build_feature_frame(df), await serving_features(df, monkeypatch))


@pytest.mark.skipif(not (DATA_DIR / "Fraud_Data.csv").exists(),
                    reason="Fraud_Data.csv is gitignored; runs locally only")
async def test_parity_on_real_data(monkeypatch):
    df = load_transactions().head(5000).copy()
    # The API receives the integer IP (client_mimic.py truncates the CSV float).
    df["ip_address"] = df["ip_address"].astype("int64")
    assert_parity(build_feature_frame(df), await serving_features(df, monkeypatch))


ALL_TIME = ["did_txn_cnt_all", "did_other_uid_all"]


def test_all_time_device_features_use_only_earlier_rows():
    """Leakage check: an all-time device feature for a row must not change when the rows
    AFTER it change. Build features on the full frame and on its first half; the first
    half's values must be identical."""
    df = synthetic_transactions()
    half = len(df) // 2
    full = build_feature_frame(df)[ALL_TIME].to_numpy()[:half]
    head = build_feature_frame(df.iloc[:half].copy())[ALL_TIME].to_numpy()
    assert (full == head).all()
    assert full.max(axis=0).min() > 0   # devices repeat in the synthetic data: both vary
    # did_txn_cnt_all equals the number of earlier rows on the same device.
    assert (full[:, 0] == df.groupby("device_id").cumcount().to_numpy()[:half]).all()


def test_user_history_features_on_repeat_users():
    """The synthetic data has repeat users, so these two features take real values here
    (on Fraud_Data they are constant and the parity test would compare constants)."""
    df = synthetic_transactions()
    X = build_feature_frame(df)
    first = ~df["user_id"].duplicated().to_numpy()
    assert (X.loc[first, "amount_vs_user_avg_30d"] == 1.0).all()
    assert (X.loc[first, "time_since_last_txn_uid"] == -1.0).all()
    assert (X.loc[~first, "time_since_last_txn_uid"] >= 0).all()
    has_recent = (X["purchase_cnt_30d"] > 0).to_numpy()
    assert has_recent.sum() > 100
    expected = df.loc[has_recent, "purchase_value"] / (
        X.loc[has_recent, "purchase_amt_30d"] / X.loc[has_recent, "purchase_cnt_30d"]).clip(lower=1)
    assert np.allclose(X.loc[has_recent, "amount_vs_user_avg_30d"], expected)
    assert (X.loc[~has_recent, "amount_vs_user_avg_30d"] == 1.0).all()
    # The old definition: equal to the purchase amount when there is no history.
    assert not (X.loc[first, "amount_vs_user_avg_30d"] == df.loc[first, "purchase_value"]).all()


def test_other_accounts_excludes_the_buyer():
    """The synthetic data has repeat users (unlike Fraud_Data), so the two features differ:
    other accounts <= earlier transactions, strictly fewer when the buyer has been on the
    device before, and never counting the buyer themselves."""
    df = synthetic_transactions()
    X = build_feature_frame(df)
    assert (X["did_other_uid_all"] <= X["did_txn_cnt_all"]).all()
    assert (X["did_other_uid_all"] < X["did_txn_cnt_all"]).any()
    expected = []
    seen: dict[str, set] = {}
    for device, user in zip(df["device_id"], df["user_id"]):
        expected.append(len(seen.setdefault(device, set()) - {user}))
        seen[device].add(user)
    assert (X["did_other_uid_all"].to_numpy() == expected).all()


async def test_all_time_device_features_are_compared_by_the_parity_tests(monkeypatch):
    df = synthetic_transactions(n=300)
    serve = await serving_features(df, monkeypatch)
    for c in ALL_TIME:
        assert c in serve.columns and serve[c].max() > 0


# ---------------------------------------------------------------------------
# Set features include the buyer; count features do not (2026-11-09)
# ---------------------------------------------------------------------------

def _b_a_a() -> pd.DataFrame:
    """Users B, A, A on one device and one IP within an hour."""
    start = dt.datetime(2015, 3, 1, 12, 0, 0)
    rows = [{"user_id": u, "device_id": "d1", "ip_address": 1_000_001,
             "purchase_time": start + dt.timedelta(minutes=m)}
            for u, m in (("B", 0), ("A", 10), ("A", 20))]
    df = pd.DataFrame(rows)
    df["signup_time"] = df["purchase_time"] - dt.timedelta(days=30)
    df["purchase_value"] = 20
    df["source"], df["browser"], df["sex"], df["age"] = "SEO", "Chrome", "M", 30
    df["class"] = 0
    df["ip_country"] = "United States"
    return df


async def test_set_features_include_the_buyer(monkeypatch):
    df = _b_a_a()
    train = build_feature_frame(df)
    serve = await serving_features(df, monkeypatch)
    for frame in (train, serve):
        assert list(frame["did_uid_1h"]) == [1, 2, 2]
        assert list(frame["ip_uid_1h"]) == [1, 2, 2]
        assert list(frame["uid_did_1h"]) == [1, 1, 1]      # each user only ever used d1
        assert list(frame["did_txn_cnt_1h"]) == [0, 1, 2]  # counts stay strictly prior
        first = frame.iloc[0]
        set_cols = [c for c in frame.columns
                    if c.startswith(("did_uid_", "uid_did_", "ip_uid_"))
                    and c != "did_uid_days_on_file"]
        assert all(first[c] == 1 for c in set_cols), "first purchase ever: every set feature = 1"
    assert_parity(train, serve)
