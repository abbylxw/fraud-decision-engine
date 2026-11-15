"""
Feature ablation on the rolling-origin folds: does a new feature group really help?

    python -m training.feature_ablation [--n-boot 1000]

"Ablation" = fit the same model with and without a group of features and compare. Each
feature set is scored on the seven rolling-origin folds of training/rolling_eval.py
(train on the past, score the next month) with the tuned XGBoost, and compared with the
baseline set by a PAIRED bootstrap, so a gain is only claimed when its 95% interval
excludes zero.

The group under test here is all-time device history. Every other device counter looks
back at most 90 days; these have no window:

    did_txn_cnt_all            transactions ever seen on this device before this one
    did_secs_since_first_txn   seconds since the device's first transaction (-1 = new device)
    did_secs_since_last_txn    seconds since the device's previous transaction (-1 = new)
    did_uid_all                distinct users ever seen on the device, including this one

Diagnostic only: writes results/ablation_<timestamp>/ablation.json, never the served model.
"""
import argparse
import datetime as dt
import json
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import best_trees, fit_candidate
from training.rolling_eval import FOLDS, label, paired_mean_diff
from training.train import CANDIDATE_FEATURES, DROP_ALWAYS, make_candidates

warnings.filterwarnings("ignore", category=UserWarning)

NEW = ["did_txn_cnt_all", "did_secs_since_first_txn", "did_secs_since_last_txn", "did_uid_all"]
BASELINE = "baseline"


def feature_sets():
    base = [c for c in CANDIDATE_FEATURES if c not in NEW]
    sets = {BASELINE: base}
    for c in NEW:
        sets[f"+{c}"] = base + [c]
    sets["+count and recency (3)"] = base + NEW[:3]
    sets["+all four"] = base + NEW
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    print("Building features (velocity replay)...")
    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    month = df["purchase_time"].dt.month.to_numpy()
    gap = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().to_numpy()
    keep = ~((gap >= 0) & (gap < SIGNUP_BURST_SECONDS))

    # How much traffic the new features can say anything about, on the scored months.
    scored_any = keep & (month >= 5)
    seen = (X_all["did_txn_cnt_all"] > 0).to_numpy()
    coverage = {
        "share_of_scored_rows_on_a_seen_device": round(float(seen[scored_any].mean()), 4),
        "fraud_rate_seen_device": round(float(y[scored_any & seen].mean()), 4),
        "fraud_rate_new_device": round(float(y[scored_any & ~seen].mean()), 4),
    }
    print(f"  scored rows on a device seen before: {coverage['share_of_scored_rows_on_a_seen_device']:.1%}"
          f" (fraud rate {coverage['fraud_rate_seen_device']:.1%} vs "
          f"{coverage['fraud_rate_new_device']:.1%} on new devices)")

    sets = feature_sets()
    folds, per_fold = [], []
    for stop_month, scored_months in FOLDS:
        tr = keep & (month < stop_month)
        va = keep & (month == stop_month)
        sc = keep & np.isin(month, scored_months)
        ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        scores, row = {}, {"scored": label(scored_months)}
        for name, cols in sets.items():
            feats = [c for c in cols if c not in DROP_ALWAYS and X_all.loc[tr, c].nunique() > 1]
            X = X_all[feats]
            model = make_candidates(ratio)["xgboost"]
            fit_candidate("xgboost", model, X[tr], y[tr], X[va], y[va])
            scores[name] = model.predict_proba(X[sc])[:, 1]
            row[name] = {"pr_auc": round(float(average_precision_score(y[sc], scores[name])), 4),
                         "trees": best_trees("xgboost", model)}
        folds.append({"y": y[sc], "scores": scores})
        per_fold.append(row)
        print(f"  {row['scored']:<8} " + "  ".join(f"{row[n]['pr_auc']:.4f}" for n in sets))

    summary = {}
    for name in sets:
        summary[name] = {"mean_pr_auc": round(float(np.mean([r[name]["pr_auc"] for r in per_fold])), 4)}
        if name != BASELINE:
            summary[name]["vs_baseline"] = paired_mean_diff(folds, name, BASELINE, args.n_boot)

    out = ROOT / "results" / f"ablation_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps({
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": "tuned XGBoost (train.make_candidates), rolling-origin folds of rolling_eval.py",
        "n_boot": args.n_boot, "new_features": NEW, "coverage": coverage,
        "folds": per_fold, "summary": summary,
    }, indent=2))

    print(f"\n  {'feature set':<28} {'mean PR-AUC':>11}   paired difference vs baseline [95% CI]")
    for name, s in summary.items():
        d = s.get("vs_baseline")
        tail = (f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else "")
        print(f"  {name:<28} {s['mean_pr_auc']:>11.4f}   {tail}")
    print(f"Wrote {out.relative_to(ROOT)}/ablation.json")


if __name__ == "__main__":
    main()
