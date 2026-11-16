"""
Honest model evaluation.

    python -m training.evaluate [--review-cost 5] [--n-boot 500]

Writes results/<timestamp>/metrics.json plus PNG charts. Never touches
app/decisions/model.joblib - shipping a new model is a separate decision.

Rules this script follows (each one prevents a way of fooling yourself):
  * Split by TIME: train = first 70%, validation = next 10%, test = last 20%.
    The test period is the same one the notebook used, so numbers are comparable.
  * Early stopping and threshold choice use VALIDATION only. Test is scored once.
  * Every headline metric has a bootstrap 95% confidence interval, so we can tell
    a real improvement from noise.
"""
import argparse
import datetime as dt
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# This script's cost view is "alert on an order at $review_cost each". The served engine
# acts on an order from its review threshold upwards (review, or deny above the deny
# threshold), so that is the served alert threshold here. For a model shipped without a
# review band the two thresholds are equal.
from app.decisions.models import REVIEW_THRESHOLD as SERVED_THRESHOLD
from app.decisions.models import _model as deployed_model
from training.build_features import ROOT, SERVED_FEATURES, build_feature_frame, load_transactions
from training.candidates import BOOSTERS, best_trees, fit_candidate

MONOTONE = {"did_other_uid_all": 1}   # same constraint as training/train.py

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42

# Chart colors: categorical slots 1-3 of the validated reference palette, plus a gray
# for "baseline / reference" lines. Text stays in neutral ink, never series color.
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
C_PURPLE, C_BROWN = "#7a5bd6", "#a06a2c"
INK, INK_2 = "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bootstrap_ci(y, p, metric, n_boot, seed=SEED):
    """95% CI by resampling test rows with replacement (percentile method)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], p[idx]))
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def at_alert_rate(y, p, rate):
    """If we review the top `rate` share of transactions by score, what do we catch?"""
    k = max(1, int(round(len(p) * rate)))
    top = np.argsort(-p, kind="stable")[:k]
    caught = y[top].sum()
    return {"recall": round(float(caught / y.sum()), 4), "precision": round(float(caught / k), 4)}


def precision_at_recall(y, p, target=0.5):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec >= target
    return round(float(prec[ok].max()), 4) if ok.any() else 0.0


def dollar_cost(y, p, value, threshold, review_cost):
    """Cost = every alert costs a manual review; every missed fraud loses its value."""
    alert = p >= threshold
    missed = (y == 1) & ~alert
    return float(alert.sum() * review_cost + value[missed].sum())


def pick_threshold(y, p, value, review_cost):
    """Cheapest threshold on the validation set (searched over score quantiles)."""
    grid = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    grid = np.append(grid, 1.01)   # 1.01 = "alert on nothing"
    costs = [dollar_cost(y, p, value, t, review_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), grid, np.array(costs)


def score_report(y, p, n_boot):
    y = np.asarray(y)
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "pr_auc_ci95": bootstrap_ci(y, p, average_precision_score, n_boot),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "roc_auc_ci95": bootstrap_ci(y, p, roc_auc_score, n_boot),
        "at_1pct_alerts": at_alert_rate(y, p, 0.01),
        "at_5pct_alerts": at_alert_rate(y, p, 0.05),
        "precision_at_50pct_recall": precision_at_recall(y, p, 0.5),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models(neg_pos_ratio):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        ),
        # Sweep winner (docs/evaluation/tune_sweep.json); same settings as train.py.
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=100,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.05, min_child_weight=100,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg_pos_ratio,
            # More other accounts on the device must never LOWER the score (see
            # docs/feature_ablation.md section 8: unconstrained, a device with 5+ other
            # accounts scored as low risk, an artefact of the January bot devices).
            monotone_constraints=MONOTONE,
            random_state=SEED, n_jobs=-1,
        ),
        # Sweep winners (docs/evaluation/tune_sweep.json); same settings as train.py.
        "lightgbm": LGBMClassifier(
            n_estimators=1000, num_leaves=7, learning_rate=0.05, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio, metric="average_precision",
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000, depth=4, learning_rate=0.03, l2_leaf_reg=10,
            scale_pos_weight=neg_pos_ratio, random_seed=SEED, verbose=False, thread_count=-1,
            allow_writing_files=False,
        ),
    }


fit = fit_candidate  # boosters early-stop on validation; see training/candidates.py


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")


def chart_fraud_by_month(df, out):
    m = df.groupby(df["purchase_time"].dt.month)["class"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(m.index, m.values, color=C_BLUE, width=0.7)
    for x, v in zip(m.index, m.values):
        ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    _style(ax, "Fraud rate by month (2015) - January is a bot attack", "", "Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_pr_curves(y, curves, base_rate, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (label, p), color in zip(curves.items(),
                                 [C_GRAY, C_BLUE, C_ORANGE, C_AQUA, C_PURPLE, C_BROWN]):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{label} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(base_rate, color=C_GRAY, linewidth=1, linestyle="--",
               label=f"random guessing ({base_rate:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Precision-recall on the test period (Sep-Dec)", "Recall", "Precision")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_cost(grid, costs, chosen, label, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(grid[:-1], costs[:-1] / 1000, color=C_BLUE, linewidth=2)
    ax.axhline(costs[-1] / 1000, color=C_GRAY, linestyle="--", linewidth=1,
               label="alert on nothing")
    ax.axvline(chosen, color=C_ORANGE, linewidth=1.5, label=f"chosen threshold {chosen:.2f}")
    _style(ax, f"Validation cost vs threshold - {label}", "Score threshold", "Total cost ($k)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--review-cost", type=float, default=5.0,
                    help="dollar cost of manually reviewing one alerted transaction")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples")
    args = ap.parse_args()

    out = ROOT / "results" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print("Building features (velocity replay, ~10 s)...")
    df = load_transactions()
    X_all = build_feature_frame(df)
    X, y = X_all[SERVED_FEATURES], df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    secs_to_purchase = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().to_numpy()

    n = len(df)
    i_va, i_te = int(n * 0.70), int(n * 0.80)
    tr, va, te = slice(0, i_va), slice(i_va, i_te), slice(i_te, n)
    t = df["purchase_time"]
    splits = {
        name: {"rows": s.stop - s.start, "from": str(t.iloc[s.start]),
               "to": str(t.iloc[s.stop - 1]), "fraud_rate": round(float(y[s].mean()), 4)}
        for name, s in [("train", tr), ("validation", va), ("test", te)]
    }
    for name, s in splits.items():
        print(f"  {name:<10} {s['from'][:10]} -> {s['to'][:10]}  "
              f"rows={s['rows']:,}  fraud={s['fraud_rate']:.1%}")

    # --- Diagnostics: why is this hard? -----------------------------------
    dead = [c for c in SERVED_FEATURES if X.iloc[tr][c].nunique() == 1]
    bot = secs_to_purchase <= 1
    diagnostics = {
        "users_with_more_than_one_txn": int((df["user_id"].value_counts() > 1).sum()),
        "constant_features_in_train": dead,
        "n_constant_features": len(dead),
        "n_served_features": len(SERVED_FEATURES),
        "fraud_rate_by_month": {
            int(k): round(float(v), 4)
            for k, v in df.groupby(t.dt.month)["class"].mean().items()
        },
        "bot_burst": {
            "definition": "purchase within 1 second of signup",
            "rows_total": int(bot.sum()),
            "fraud_rate": round(float(y[bot].mean()), 4),
            "share_of_train_fraud": round(float(y[tr][bot[tr]].sum() / y[tr].sum()), 4),
            "rows_in_test": int(bot[te].sum()),
        },
    }
    chart_fraud_by_month(df, out / "fraud_by_month.png")

    # --- Train & score ---------------------------------------------------
    setups = {
        "all_train_data": np.ones(i_va, dtype=bool),
        "without_bot_burst": ~bot[tr],
    }
    results, test_scores, val_scores = {}, {}, {}

    p_deployed = deployed_model.predict_proba(X.iloc[te])[:, 1]
    results["deployed_model"] = {
        "note": "app/decisions/model.joblib as served today; trained on the first 80% "
                "(train + validation), so it is scored on test only.",
        "test": score_report(y[te], p_deployed, args.n_boot),
    }
    test_scores["deployed_model"] = p_deployed
    results["base_rate"] = {"test": {"pr_auc": round(float(y[te].mean()), 4), "roc_auc": 0.5},
                            "note": "score every transaction the same; PR-AUC = fraud rate"}

    for setup, keep in setups.items():
        X_tr, y_tr = X.iloc[tr][keep], y[tr][keep]
        ratio = float((y_tr == 0).sum() / (y_tr == 1).sum())
        for name, model in make_models(ratio).items():
            key = f"{name}/{setup}"
            print(f"Training {key} (train rows={len(y_tr):,}, fraud={y_tr.mean():.1%})...")
            fit(name, model, X_tr, y_tr, X.iloc[va], y[va])
            p_va = model.predict_proba(X.iloc[va])[:, 1]
            p_te = model.predict_proba(X.iloc[te])[:, 1]
            val_scores[key], test_scores[key] = p_va, p_te
            results[key] = {
                "train_rows": int(len(y_tr)),
                "validation_pr_auc": round(float(average_precision_score(y[va], p_va)), 4),
                "test": score_report(y[te], p_te, args.n_boot),
            }
            if name in BOOSTERS:
                results[key]["trees"] = best_trees(name, model)
                imp = pd.Series(model.feature_importances_, index=SERVED_FEATURES)
                results[key]["top_features"] = {
                    k: round(float(v), 4) for k, v in imp.nlargest(10).items()}
            print(f"   val PR-AUC={results[key]['validation_pr_auc']:.4f}  "
                  f"test PR-AUC={results[key]['test']['pr_auc']:.4f} "
                  f"{results[key]['test']['pr_auc_ci95']}")

    # --- Pick the best candidate on VALIDATION, then report it on test ------
    best = max(val_scores, key=lambda k: results[k]["validation_pr_auc"])
    print(f"Best on validation: {best}")
    chosen, grid, costs = pick_threshold(y[va], val_scores[best], value[va], args.review_cost)
    chart_cost(grid, costs, chosen, best, out / "cost_vs_threshold.png")

    def cost_block(p, thr):
        c = dollar_cost(y[te], p, value[te], thr, args.review_cost)
        alert = p >= thr
        caught = int((alert & (y[te] == 1)).sum())
        return {"threshold": round(thr, 4), "total_cost": round(c, 2),
                "alerts": int(alert.sum()), "fraud_caught": caught,
                "fraud_total": int(y[te].sum())}

    nothing = float(value[te][y[te] == 1].sum())
    threshold_report = {
        "review_cost_per_alert": args.review_cost,
        "chosen_on": "validation",
        "cost_of_alerting_on_nothing": round(nothing, 2),
        "best_model": best,
        "best_model_at_chosen_threshold": cost_block(test_scores[best], chosen),
        "best_model_at_served_threshold": cost_block(test_scores[best], SERVED_THRESHOLD),
        "deployed_model_at_served_threshold": cost_block(p_deployed, SERVED_THRESHOLD),
    }
    for k in ("best_model_at_chosen_threshold", "best_model_at_served_threshold",
              "deployed_model_at_served_threshold"):
        b = threshold_report[k]
        b["saved_vs_nothing"] = round(nothing - b["total_cost"], 2)

    chart_pr_curves(
        y[te],
        {"deployed model": p_deployed,
         **{f"{m.replace('_', ' ')} (no bot rows)": test_scores[f"{m}/without_bot_burst"]
            for m in ("random_forest", "xgboost", "lightgbm", "catboost")}},
        float(y[te].mean()),
        out / "pr_curves_test.png",
    )

    metrics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "splits": splits,
        "diagnostics": diagnostics,
        "models": results,
        "threshold": threshold_report,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}/metrics.json and charts.")
    print(json.dumps(threshold_report, indent=2))


if __name__ == "__main__":
    main()
