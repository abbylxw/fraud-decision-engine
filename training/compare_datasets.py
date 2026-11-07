"""
Does the velocity layer help? The same velocity replay on two datasets, as an ablation.

    python -m training.compare_datasets                   # every dataset that is downloaded
    python -m training.compare_datasets sparkov           # one
    python -m training.compare_datasets --rebuild         # recompute cached features

Fraud_Data.csv has one transaction per user, so its per-user velocity features are always
0. Sparkov has repeat users (download: python -m training.datasets.download).
Each dataset is mapped onto the canonical columns (see training/datasets/), then run through
training/build_features.velocity_features() unchanged: the same replay the parity test
checks against the live API.

Per dataset, one XGBoost configuration is trained on four feature sets, each adding to
the previous one:
  static             columns of the transaction itself (amount, category, time of day, ...)
  static+velocity    + backward-looking counters (purchase counts/amounts, distinct
                     users per device/IP, time since last transaction, ...)
  +spend_spike       + today's spend vs the median active-day spend over 14/30/90 days
                     (build_features.spend_spike_features; training only, not served)
  +label_history     + purchase_success_*: counts of the user's past NON-fraud purchases.
                     Optimistic: the replay knows each past label instantly, but in
                     production fraud labels arrive weeks later (chargebacks).

Split: first 70% by time = train, next 10% = validation (early stopping, threshold),
last 20% = test, scored once. Writes results/datasets_<ts>/metrics.json and the committed
summary docs/evaluation/multi_dataset.json.
"""
import argparse
import datetime as dt
import json
import time

import pandas as pd
from xgboost import XGBClassifier

from app.decisions.models import COUNTRY_ONEHOT
from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import (
    ROOT,
    build_feature_frame,
    load_transactions,
    spend_spike_features,
    velocity_features,
)
from training.datasets import EXTERNAL_DIR, sparkov
from training.evaluate import SEED, dollar_cost, pick_threshold, score_report

CACHE_DIR = ROOT / "results" / "cache"
SUMMARY_PATH = ROOT / "docs" / "evaluation" / "multi_dataset.json"
META = ["purchase_time", "user_id", "purchase_value", "class"]


# ---------------------------------------------------------------------------
# Datasets -> (canonical frame, static features, rows to evaluate)
# ---------------------------------------------------------------------------

def _load_fraud_data():
    """Fraud_Data.csv as the served model sees it: bot-burst rows are denied by the
    signup_burst rule before the model runs, so they are dropped (after the replay,
    because in production they still enter the velocity history)."""
    df = load_transactions()
    static_cols = ["purchase_value", "days_since_signup", "sex", "source", "browser",
                   *COUNTRY_ONEHOT.values(), "hour_of_day", "day_of_week", "age_group",
                   "age_account_gap"]
    static = build_feature_frame(df)[static_cols]
    secs = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    keep = ~((secs >= 0) & (secs < SIGNUP_BURST_SECONDS)).to_numpy()
    return df, static, keep


DATASETS = {
    "fraud_data": (_load_fraud_data, ROOT / "app" / "data" / "Fraud_Data.csv"),
    "sparkov": (lambda: (*sparkov.load(), None), EXTERNAL_DIR / "sparkov"),
}


def add_derived(v: pd.DataFrame, purchase_value: pd.Series) -> pd.DataFrame:
    """The two derived velocity features from build_feature_frame()."""
    user_avg_30d = v["purchase_amt_30d"] / v["purchase_cnt_30d"].clip(lower=1)
    v["amount_vs_user_avg_30d"] = purchase_value / user_avg_30d.clip(lower=1)
    v["high_amount_vs_hist"] = (
        (v["pur_median_amt_90d"] > 0) & (purchase_value > 2 * v["pur_median_amt_90d"])
    ).astype(int)
    return v


def feature_table(name: str, rebuild: bool) -> tuple[pd.DataFrame, dict]:
    """One frame with META + static + velocity + spend-spike columns, cached as parquet
    (the Python replay takes minutes on the large datasets).
    Returns (frame, {"static": [...], "velocity": [...], "spend_spike": [...]})."""
    cache = CACHE_DIR / f"{name}.parquet"
    cols_file = CACHE_DIR / f"{name}.columns.json"
    if cache.exists() and cols_file.exists() and not rebuild:
        cols = json.loads(cols_file.read_text())
        if "spend_spike" in cols:   # caches from before the spend-spike features rebuild
            return pd.read_parquet(cache), cols

    loader, _ = DATASETS[name]
    t0 = time.time()
    canonical, static, keep = loader()
    print(f"  loaded {len(canonical):,} rows; velocity replay...", flush=True)
    velocity = add_derived(velocity_features(canonical), canonical["purchase_value"])
    spike = spend_spike_features(canonical)
    print(f"  replay done in {time.time() - t0:.0f}s", flush=True)

    static = static.add_prefix("s_")   # avoid name clashes (e.g. purchase_value)
    frame = pd.concat([canonical[META].astype({"user_id": str}), static, velocity, spike],
                      axis=1)
    if keep is not None:
        frame = frame[keep].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    cols = {"static": list(static.columns), "velocity": list(velocity.columns),
            "spend_spike": list(spike.columns)}
    cols_file.write_text(json.dumps(cols))
    return frame, cols


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def make_model(neg_pos_ratio):
    """One fixed configuration for every dataset and feature set, so differences come
    from the features. Early-stopped on validation."""
    return XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=neg_pos_ratio,
        eval_metric="aucpr", early_stopping_rounds=50, tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )


def spike_flag_summary(frame, y, rows):
    """How often a 30-day spike flag fires on test rows, and the fraud rate when it does:
    a readable check that the feature carries signal, independent of the model."""
    has_baseline = frame["spend_ratio_day_vs_median_30d"].iloc[rows] >= 0
    out = {"share_with_30d_baseline": round(float(has_baseline.mean()), 4)}
    for k in ("3x", "5x"):
        on = frame[f"spend_spike_{k}_30d"].iloc[rows].to_numpy() == 1
        out[f"spike_{k}_30d"] = {
            "share_of_rows": round(float(on.mean()), 4),
            "fraud_rate_when_on": round(float(y[rows][on].mean()), 4) if on.any() else None,
            "fraud_rate_when_off": round(float(y[rows][~on].mean()), 4),
        }
    return out


def run_dataset(name, frame, cols, n_boot, review_cost):
    static_cols, velocity_cols, spike_cols = cols["static"], cols["velocity"], cols["spend_spike"]
    y = frame["class"].to_numpy()
    value = frame["purchase_value"].to_numpy(dtype=float)
    t = frame["purchase_time"]
    n = len(frame)
    i_va, i_te = int(n * 0.70), int(n * 0.80)
    tr, va, te = slice(0, i_va), slice(i_va, i_te), slice(i_te, n)

    label_history = [c for c in velocity_cols if c.startswith("purchase_success_")]
    velocity = [c for c in velocity_cols if c not in label_history]
    live_velocity = [c for c in velocity if frame[c].iloc[tr].nunique() > 1]

    result = {
        "rows": n,
        "users": int(frame["user_id"].nunique()),
        "splits": {
            k: {"rows": s.stop - s.start, "from": str(t.iloc[s.start])[:10],
                "to": str(t.iloc[s.stop - 1])[:10],
                "fraud_rate": round(float(y[s].mean()), 4)}
            for k, s in [("train", tr), ("validation", va), ("test", te)]
        },
        "n_static_features": len(static_cols),
        "n_velocity_features": len(velocity),
        "n_velocity_features_non_constant_in_train": len(live_velocity),
        "n_spend_spike_features": len(spike_cols),
        "n_spend_spike_features_non_constant_in_train":
            sum(frame[c].iloc[tr].nunique() > 1 for c in spike_cols),
        "spend_spike_test": spike_flag_summary(frame, y, te),
        "feature_sets": {},
    }
    neg_pos = float((y[tr] == 0).sum() / max(1, y[tr].sum()))
    fraud_total = float(value[te][y[te] == 1].sum())

    for setup, feats in [
        ("static", static_cols),
        ("static+velocity", static_cols + velocity),
        ("+spend_spike", static_cols + velocity + spike_cols),
        ("+label_history", static_cols + velocity + spike_cols + label_history),
    ]:
        t0 = time.time()
        X = frame[feats]
        model = make_model(neg_pos)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        p_te = model.predict_proba(X.iloc[te])[:, 1]

        threshold, _, _ = pick_threshold(y[va], p_va, value[va], review_cost)
        alert = p_te >= threshold
        caught = alert & (y[te] == 1)
        report = score_report(y[te], p_te, n_boot)
        report["cost_threshold"] = {
            "threshold": round(threshold, 4),
            "alert_rate": round(float(alert.mean()), 4),
            "fraud_caught": f"{int(caught.sum())} / {int(y[te].sum())}",
            "fraud_dollars_caught_share": round(float(value[te][caught].sum() / fraud_total), 4),
            "total_cost": round(dollar_cost(y[te], p_te, value[te], threshold, review_cost)),
            "cost_if_no_model": round(fraud_total),
        }
        importance = pd.Series(model.feature_importances_, index=feats).nlargest(8)
        report["top_features"] = {k: round(float(v), 3) for k, v in importance.items()}
        report["best_iteration"] = int(model.best_iteration)
        result["feature_sets"][setup] = report
        print(f"  {setup:<16} PR-AUC {report['pr_auc']:.3f} {report['pr_auc_ci95']}  "
              f"ROC-AUC {report['roc_auc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--rebuild", action="store_true", help="recompute cached features")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--review-cost", type=float, default=5.0,
                    help="dollar cost of reviewing one alerted transaction")
    args = ap.parse_args()

    out = ROOT / "results" / f"datasets_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    summary.setdefault("datasets", {})

    for name in args.datasets:
        if not DATASETS[name][1].exists():
            print(f"{name}: not downloaded, skipping (python -m training.datasets.download)")
            continue
        print(f"{name}:", flush=True)
        frame, cols = feature_table(name, args.rebuild)
        summary["datasets"][name] = run_dataset(
            name, frame, cols, args.n_boot, args.review_cost)

    summary["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    summary["settings"] = {"split": "70/10/20 by time", "n_boot": args.n_boot,
                           "review_cost_per_alert": args.review_cost,
                           "model": "XGBoost depth 6, lr 0.1, early-stopped on validation"}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'metrics.json'} and {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
