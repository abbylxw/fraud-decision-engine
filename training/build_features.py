"""
Build the training feature matrix from Fraud_Data.csv.

This is the feature code from notebooks/train_model.ipynb (cells 3, 8-11, plus the
age_group / age_account_gap features from the version that produced model.joblib),
moved into a module so scripts and tests can reuse it. The logic is unchanged;
only the category encodings now come from app/decisions/models.py, so training and
serving share ONE definition (see tests/test_encoding_contract.py).

Key idea: velocity features look *backward* in time. We replay transactions in
purchase_time order, and each row only sees the history before it. A row is added
to the history AFTER its own features are computed, so it never "sees itself".
"""
import bisect
import json
import pathlib
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from app.decisions.models import (
    _BROWSER_CODES,
    _SEX_CODES,
    _SOURCE_CODES,
    _age_group_code,
    _country_code,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
SERVED_FEATURES = json.loads((ROOT / "app/decisions/model_features.json").read_text())

WINDOWS = {
    "5min": 5 * 60,
    "10min": 10 * 60,
    "1h": 3600,
    "2h": 2 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 86400,
    "30d": 30 * 86400,
    "90d": 90 * 86400,
}


def load_transactions(csv_path=DATA_DIR / "Fraud_Data.csv",
                      ip_csv_path=DATA_DIR / "IpAddress_to_Country.csv") -> pd.DataFrame:
    """Load transactions sorted by purchase_time, with ip_country attached."""
    df = pd.read_csv(csv_path, parse_dates=["signup_time", "purchase_time"])
    # CRITICAL: sort first, so "earlier rows" really are earlier in time.
    df = df.sort_values("purchase_time", kind="stable").reset_index(drop=True)

    ip = pd.read_csv(ip_csv_path)
    lo = ip["lower_bound_ip_address"].values
    hi = ip["upper_bound_ip_address"].values
    country = ip["country"].values

    def lookup(ip_float):
        if pd.isna(ip_float):
            return "Unknown"
        n = int(ip_float)
        i = bisect.bisect_right(lo, n) - 1
        return country[i] if i >= 0 and n <= hi[i] else "Unknown"

    df["ip_country"] = df["ip_address"].apply(lookup)
    return df


def _distinct_since(times, vals, cutoff):
    return len(set(vals[bisect.bisect_left(times, cutoff):]))


def _count_sum_since(times, vals, cutoff):
    v = vals[bisect.bisect_left(times, cutoff):]
    return len(v), sum(v)


def _count_since(times, cutoff):
    return len(times) - bisect.bisect_left(times, cutoff)


def velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replay transactions in time order and compute backward-looking counters
    (notebook cells 8-9)."""
    did_uid_t, did_uid_v = defaultdict(list), defaultdict(list)
    uid_did_t, uid_did_v = defaultdict(list), defaultdict(list)
    ip_uid_t, ip_uid_v = defaultdict(list), defaultdict(list)
    pur_t, pur_v = defaultdict(list), defaultdict(list)
    suc_t, suc_v = defaultdict(list), defaultdict(list)
    did_txn_t = defaultdict(list)
    country_t = defaultdict(list)
    uid_last_ts, did_first_seen = {}, {}

    rows = []
    for r in df.itertuples(index=False):
        ts = r.purchase_time.timestamp()
        uid, did, ip, val = str(r.user_id), str(r.device_id), r.ip_address, r.purchase_value
        f = {}
        for w in ["5min", "10min", "1h", "2h", "24h", "7d", "30d", "90d"]:
            f[f"did_uid_{w}"] = _distinct_since(did_uid_t[did], did_uid_v[did], ts - WINDOWS[w])
        for w in ["5min", "10min", "1h", "2h", "24h", "7d", "30d", "90d"]:
            f[f"uid_did_{w}"] = _distinct_since(uid_did_t[uid], uid_did_v[uid], ts - WINDOWS[w])
        for w in ["5min", "10min", "1h", "2h", "24h", "7d"]:
            f[f"ip_uid_{w}"] = _distinct_since(ip_uid_t[ip], ip_uid_v[ip], ts - WINDOWS[w])
        for w in ["5min", "10min", "1h", "24h", "7d", "30d", "90d"]:
            f[f"purchase_cnt_{w}"], f[f"purchase_amt_{w}"] = _count_sum_since(
                pur_t[uid], pur_v[uid], ts - WINDOWS[w])
        for w in ["5min", "10min", "1h", "24h", "7d", "30d", "90d"]:
            f[f"purchase_success_cnt_{w}"], f[f"purchase_success_amt_{w}"] = _count_sum_since(
                suc_t[uid], suc_v[uid], ts - WINDOWS[w])

        key = (did, uid)
        if key not in did_first_seen:
            did_first_seen[key] = r.purchase_time
            f["did_uid_days_on_file"] = 0
        else:
            f["did_uid_days_on_file"] = (r.purchase_time - did_first_seen[key]).days

        recent = pur_v[uid][bisect.bisect_left(pur_t[uid], ts - WINDOWS["90d"]):]
        f["pur_median_amt_90d"] = float(np.median(recent)) if recent else 0.0

        for w in ["5min", "10min", "1h", "2h", "24h", "7d", "30d"]:
            f[f"did_txn_cnt_{w}"] = _count_since(did_txn_t[did], ts - WINDOWS[w])
        f["time_since_last_txn_uid"] = ts - uid_last_ts.get(uid, ts)
        for w in ["5min", "10min", "1h", "24h", "7d", "30d"]:
            f[f"country_txn_count_{w}"] = _count_since(country_t[r.ip_country], ts - WINDOWS[w])
        rows.append(f)

        # Add this transaction to history only AFTER computing its features.
        for times, vals, k, v in [
            (did_uid_t, did_uid_v, did, uid),
            (uid_did_t, uid_did_v, uid, did),
            (ip_uid_t, ip_uid_v, ip, uid),
            (pur_t, pur_v, uid, val),
        ]:
            times[k].append(ts)
            vals[k].append(v)
        did_txn_t[did].append(ts)
        uid_last_ts[uid] = ts
        country_t[r.ip_country].append(ts)
        if r[df.columns.get_loc("class")] == 0:   # "successful" = not fraud
            suc_t[uid].append(ts)
            suc_v[uid].append(val)

    return pd.DataFrame(rows)


# Spend spike: is today's spend a multiple of the user's typical daily spend?
SPIKE_WINDOWS_DAYS = (14, 30, 90)
SPIKE_MULTIPLIERS = (3, 5)
MIN_ACTIVE_DAYS = 5   # fewer active days than this in the window = no baseline


def spend_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Today's spend vs the median daily spend on the user's active days.

    TRAINING ONLY: not computed by the live API and not used by the served model
    (measured in training/compare_datasets.py). Redis keeps 90 days of per-user purchases,
    so a live version is possible for these windows later.

    Per row, with days in UTC:
      today_spend   = the user's earlier purchases today + this purchase (its amount is
                      known at decision time, so this is not leakage)
      baseline (Wd) = median of the user's daily totals over days WITH purchases in
                      [today - W, today - 1]; today itself is excluded
      spend_ratio_day_vs_median_Wd = today_spend / baseline, or -1 with fewer than
                      MIN_ACTIVE_DAYS active days
      spend_spike_Kx_Wd = 1 if the ratio >= K
    Like velocity_features(), rows must be sorted by purchase_time and each row is added
    to the history only after its own features are computed.
    """
    days_of, totals_of = defaultdict(list), defaultdict(list)   # per user, oldest day first
    rows = []
    for uid, t, val in zip(df["user_id"].astype(str), df["purchase_time"], df["purchase_value"]):
        today = int(t.timestamp() // 86400)
        days, totals = days_of[uid], totals_of[uid]
        has_today = bool(days) and days[-1] == today
        today_spend = (totals[-1] if has_today else 0.0) + val
        end = len(days) - 1 if has_today else len(days)   # baseline stops before today

        f = {}
        for w in SPIKE_WINDOWS_DAYS:
            past = totals[bisect.bisect_left(days, today - w, 0, end):end]
            median = statistics.median(past) if len(past) >= MIN_ACTIVE_DAYS else 0.0
            ratio = today_spend / median if median > 0 else -1.0
            f[f"spend_ratio_day_vs_median_{w}d"] = ratio
            for k in SPIKE_MULTIPLIERS:
                f[f"spend_spike_{k}x_{w}d"] = int(ratio >= k)
        rows.append(f)

        if has_today:
            totals[-1] += val
        else:
            days.append(today)
            totals.append(val)
    return pd.DataFrame(rows)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All features, encoded exactly the way the live API encodes them.
    Columns are a superset of SERVED_FEATURES; select with X[SERVED_FEATURES]."""
    days = (df["purchase_time"] - df["signup_time"]).dt.days
    raw = pd.DataFrame({
        "purchase_value": df["purchase_value"],
        "days_since_signup": days.fillna(-999).astype(int),
        "sex": df["sex"].map(_SEX_CODES).fillna(-1).astype(int),
        "source": df["source"].map(_SOURCE_CODES).fillna(-1).astype(int),
        "browser": df["browser"].map(_BROWSER_CODES).fillna(-1).astype(int),
        "ip_country_code": df["ip_country"].map(_country_code),
        "hour_of_day": df["purchase_time"].dt.hour,
        "day_of_week": df["purchase_time"].dt.dayofweek,
        "month_of_year": df["purchase_time"].dt.month,
        "age_group": df["age"].map(_age_group_code),
        "age_account_gap": df["age"] - days.fillna(-999) / 365,
    })
    X = pd.concat([raw, velocity_features(df)], axis=1)

    # Derived features (notebook cell 11)
    user_avg_30d = X["purchase_amt_30d"] / X["purchase_cnt_30d"].clip(lower=1)
    X["amount_vs_user_avg_30d"] = X["purchase_value"] / user_avg_30d.clip(lower=1)
    X["high_amount_vs_hist"] = (
        (X["pur_median_amt_90d"] > 0) & (X["purchase_value"] > 2 * X["pur_median_amt_90d"])
    ).astype(int)
    return X
