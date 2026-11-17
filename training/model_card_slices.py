"""
Numbers for docs/model_card.md: how the served model behaves for different groups.

    python -m training.model_card_slices [--n-boot 500]

Two things are measured on the test period (the last 20% by time, never trained on):

1. Per group (sex, age band, country group, traffic source, browser): how large the group
   is, its fraud rate, what share of its orders the served model sends to review, how many
   of those reviews are fraud (precision) and what share of the group's fraud is reviewed
   (recall), with bootstrap 95% intervals. A group can be reviewed more often for a good
   reason (it has more fraud) or a bad one; the fraud rate column is there to tell which.
2. What the model loses if sex and age are not inputs at all (an ablation): the same
   training recipe as train.py without `sex`, `age_group` and `age_account_gap`.

Writes docs/evaluation/model_card_slices.json. Reads the served model; changes nothing.
"""
import argparse
import datetime as dt
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from app.decisions.models import (
    MODEL_VERSION,
    REVIEW_THRESHOLD,
    THRESHOLD,
    _features,
    _model,
    country_group,
)
from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import best_trees, fit_candidate, fix_trees, time_split
from training.evaluate import SEED
from training.policy import pick_thresholds, policy_outcome
from training.train import CANDIDATE_FEATURES, DROP_ALWAYS, make_candidates

warnings.filterwarnings("ignore", category=UserWarning)
DEMOGRAPHIC_INPUTS = ("sex", "age_group", "age_account_gap")
AGE_BANDS = [(0, 25, "under 25"), (25, 35, "25 to 34"), (35, 45, "35 to 44"), (45, 200, "45 and over")]


def age_band(age):
    return next(label for lo, hi, label in AGE_BANDS if lo <= age < hi)


def ci(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def slice_stats(y, acted, n_boot, rng):
    """Review rate, precision and recall of one group, with bootstrap intervals."""
    n = len(y)
    rates, recalls, precisions = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi, ai = y[i], acted[i]
        rates.append(ai.mean())
        if yi.sum():
            recalls.append(ai[yi == 1].mean())
        if ai.sum():
            precisions.append(yi[ai].mean())
    return {
        "orders": int(n), "frauds": int(y.sum()), "fraud_rate": round(float(y.mean()), 4),
        "review_rate": round(float(acted.mean()), 4), "review_rate_ci95": ci(rates),
        "precision": round(float(y[acted].mean()), 4) if acted.any() else None,
        "precision_ci95": ci(precisions) if precisions else None,
        "recall": round(float(acted[y == 1].mean()), 4) if y.sum() else None,
        "recall_ci95": ci(recalls) if recalls else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    tr, va, te, _, features, _, _ = time_split(df, X_all, CANDIDATE_FEATURES, DROP_ALWAYS)

    # --- 1. Served model by group, test period -----------------------------------------
    score = _model.predict_proba(X_all.loc[te, _features])[:, 1]
    acted = score >= REVIEW_THRESHOLD          # reviewed (or denied, if the model ever denies)
    yt = y[te]
    groups = {
        "sex": df.loc[te, "sex"].to_numpy(),
        "age band": df.loc[te, "age"].map(age_band).to_numpy(),
        "country group": df.loc[te, "ip_country"].map(country_group).to_numpy(),
        "traffic source": df.loc[te, "source"].to_numpy(),
        "browser": df.loc[te, "browser"].to_numpy(),
    }
    slices = {"all": {"all": slice_stats(yt, acted, args.n_boot, rng)}}
    for dimension, labels in groups.items():
        slices[dimension] = {}
        for label in sorted(pd.unique(labels)):
            mask = labels == label
            if mask.sum() >= 300:   # smaller groups: intervals too wide to say anything
                slices[dimension][str(label)] = slice_stats(yt[mask], acted[mask], args.n_boot, rng)
        print(f"  {dimension}: " + "  ".join(
            f"{k} review {v['review_rate']:.1%} (fraud {v['fraud_rate']:.1%})"
            for k, v in slices[dimension].items()))

    # --- 2. The same recipe without sex and age ------------------------------------------
    def fit(feature_list):
        X = X_all[feature_list]
        ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        model = make_candidates(ratio)["xgboost"]
        fit_candidate("xgboost", model, X[tr], y[tr], X[va], y[va])
        p_va = model.predict_proba(X[va])[:, 1]
        final = fix_trees("xgboost", clone(model), best_trees("xgboost", model))
        final.fit(X[tr | va], y[tr | va])
        p_te = final.predict_proba(X[te])[:, 1]
        thresholds = pick_thresholds(y[va], p_va, value[va], 10.0, 5.0, 1.0)
        return p_te, policy_outcome(yt, p_te, value[te], *thresholds, 10.0, 5.0, 1.0)

    # The served model no longer uses these (2026-11-16), so "with" puts them back.
    without = [f for f in features if f not in DEMOGRAPHIC_INPUTS]
    with_them = without + [f for f in DEMOGRAPHIC_INPUTS if X_all[f].nunique() > 1]
    p_with, policy_with = fit(with_them)
    p_without, policy_without = fit(without)
    diffs = []
    for _ in range(args.n_boot):
        i = rng.integers(0, len(yt), len(yt))
        diffs.append(average_precision_score(yt[i], p_without[i])
                     - average_precision_score(yt[i], p_with[i]))
    ablation = {
        "removed": list(DEMOGRAPHIC_INPUTS),
        "with": {"features": len(with_them), "test_pr_auc": round(float(average_precision_score(yt, p_with)), 4),
                 "policy_cost": policy_with["total_cost"], "review_rate": policy_with["review_rate"]},
        "without": {"features": len(without),
                    "test_pr_auc": round(float(average_precision_score(yt, p_without)), 4),
                    "policy_cost": policy_without["total_cost"],
                    "review_rate": policy_without["review_rate"]},
        "paired_pr_auc_difference_ci95": ci(diffs),
        "paired_pr_auc_difference": round(float(np.mean(diffs)), 4),
    }
    print(f"  without sex and age: PR-AUC {ablation['without']['test_pr_auc']:.4f} vs "
          f"{ablation['with']['test_pr_auc']:.4f} (difference {ablation['paired_pr_auc_difference']:+.4f} "
          f"{ablation['paired_pr_auc_difference_ci95']}); cost ${policy_without['total_cost']:,.0f} "
          f"vs ${policy_with['total_cost']:,.0f}")

    out = ROOT / "docs" / "evaluation" / "model_card_slices.json"
    out.write_text(json.dumps({
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "thresholds": {"review": REVIEW_THRESHOLD, "deny": THRESHOLD},
        "period": "test (last 20% by time)", "n_boot": args.n_boot,
        "slices": slices, "ablation_without_sex_and_age": ablation,
    }, indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
