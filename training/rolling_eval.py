"""
Rolling-origin evaluation of the four tree candidates.

    python -m training.rolling_eval [--n-boot 1000]

One train/validation/test split gives one ranking of the models, and on this data that
ranking is mostly noise (see docs/model_candidates.md). Here the same question is asked
seven times: train on the past, score the next period, then move the cut-off (the
"origin") forward one month and repeat.

    fold   train      early-stop month   scored period
    1      Jan-Mar    Apr                May
    2      Jan-Apr    May                Jun
    ...
    7      Jan-Sep    Oct                Nov + Dec   (Dec alone has 35 frauds)

Rules:
  * The boosters pick their tree count on the month BEFORE the scored one, so the scored
    period is never seen while fitting. Same as production: stop on last month, predict next.
  * XGBoost's tree count is picked here with scikit-learn's average precision, not with
    XGBoost's own early stopping. Its built-in "aucpr" over-credits a model of one or two
    trees (few distinct scores, many ties: 0.088 where scikit-learn says 0.077), so early
    stopping quit at tree 1 in several folds and under-fitted. Same trap, same fix as
    CatBoost in training/candidates.py. LightGBM's "average_precision" agrees with
    scikit-learn and is left alone.
  * Hyper-parameters are the tuned ones from train.make_candidates, fixed for every fold.
  * Bot rows (signup -> purchase < 60 s, January only) are removed, as in train.py.
  * Models are compared against XGBoost (the served algorithm) with a PAIRED bootstrap:
    both models are scored on the same resampled rows, so noise they share cancels out.

Diagnostic only: writes results/rolling_<timestamp>/ and never touches the served model.
"""
import argparse
import datetime as dt
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import CATBOOST_EVAL_PERIOD, best_trees, fit_candidate
from training.evaluate import (
    C_AQUA,
    C_BLUE,
    C_GRAY,
    C_ORANGE,
    C_PURPLE,
    SEED,
    _style,
    bootstrap_ci,
)
from training.train import CANDIDATE_FEATURES, DROP_ALWAYS, INCUMBENT, make_candidates

warnings.filterwarnings("ignore", category=UserWarning)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# (early-stop month, scored months). Training = every month before the early-stop month.
FOLDS = [(4, [5]), (5, [6]), (6, [7]), (7, [8]), (8, [9]), (9, [10]), (10, [11, 12])]
COLORS = {"random_forest": C_ORANGE, "xgboost": C_BLUE, "lightgbm": C_AQUA, "catboost": C_PURPLE}


def fit_xgboost(model, X_tr, y_tr, X_va, y_va):
    """Train the full tree budget, then keep the tree count with the best scikit-learn
    PR-AUC on the early-stop month (checked every CATBOOST_EVAL_PERIOD trees).
    Returns that tree count; predict with iteration_range=(0, n)."""
    model.set_params(early_stopping_rounds=None)
    model.fit(X_tr, y_tr, verbose=False)
    counts = range(CATBOOST_EVAL_PERIOD, model.n_estimators + 1, CATBOOST_EVAL_PERIOD)
    curve = [average_precision_score(y_va, model.predict_proba(X_va, iteration_range=(0, k))[:, 1])
             for k in counts]
    return counts[int(np.argmax(curve))]


def label(months):
    return "+".join(MONTH_NAMES[m - 1] for m in months)


def paired_mean_diff(folds, challenger, incumbent, n_boot, seed=SEED):
    """Mean over folds of PR-AUC(challenger) - PR-AUC(incumbent), with a 95% CI.

    Each bootstrap round resamples the scored rows inside every fold (the same rows for
    both models) and averages the per-fold differences."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        per_fold = []
        for f in folds:
            y, a, b = f["y"], f["scores"][challenger], f["scores"][incumbent]
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                continue
            per_fold.append(average_precision_score(y[idx], a[idx])
                            - average_precision_score(y[idx], b[idx]))
        diffs.append(np.mean(per_fold))
    diffs = np.array(diffs)
    return {"mean_diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "share_of_resamples_challenger_ahead": round(float((diffs > 0).mean()), 3)}


def chart(report, names, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(report))
    for name in names:
        ax.plot(x, [f["models"][name]["pr_auc"] for f in report], color=COLORS[name],
                linewidth=2, marker="o", markersize=4, label=name.replace("_", " "))
    ax.plot(x, [f["base_rate"] for f in report], color=C_GRAY, linewidth=1, linestyle="--",
            label="random guessing (fraud rate)")
    ax.set_xticks(list(x), [f["scored"] for f in report])
    ax.set_ylim(0, None)
    _style(ax, "Rolling-origin PR-AUC: train on the past, score the next period",
           "Scored period (2015)", "PR-AUC")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


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
    keep = ~((gap >= 0) & (gap < SIGNUP_BURST_SECONDS))  # not a bot row
    print(f"  rows={len(df):,}, bot rows removed={int((~keep).sum()):,}")

    folds, report, names = [], [], None
    for i, (stop_month, scored_months) in enumerate(FOLDS, start=1):
        tr = keep & (month < stop_month)
        va = keep & (month == stop_month)
        sc = keep & np.isin(month, scored_months)
        assert not (sc & (tr | va)).any(), "scored rows leaked into fitting"
        assert month[tr].max() < stop_month < month[sc].min(), "folds must run forward in time"

        constant = [c for c in CANDIDATE_FEATURES if X_all.loc[tr, c].nunique() == 1]
        features = [c for c in CANDIDATE_FEATURES if c not in constant and c not in DROP_ALWAYS]
        X = X_all[features]
        ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        candidates = make_candidates(ratio)
        names = list(candidates)

        scores, models = {}, {}
        for name, model in candidates.items():
            if name == "xgboost":
                trees = fit_xgboost(model, X[tr], y[tr], X[va], y[va])
                p = model.predict_proba(X[sc], iteration_range=(0, trees))[:, 1]
            else:
                fit_candidate(name, model, X[tr], y[tr], X[va], y[va])
                trees = best_trees(name, model)
                p = model.predict_proba(X[sc])[:, 1]
            scores[name] = p
            pr = float(average_precision_score(y[sc], p))
            models[name] = {
                "pr_auc": round(pr, 4),
                "ci95": bootstrap_ci(y[sc], p, average_precision_score, args.n_boot),
                "lift": round(pr / float(y[sc].mean()), 2),
                "trees": trees,
            }
        winner = max(models, key=lambda n: models[n]["pr_auc"])
        ranked = sorted(names, key=lambda n: -models[n]["pr_auc"])
        for n in names:
            models[n]["rank"] = ranked.index(n) + 1

        folds.append({"y": y[sc], "scores": scores})
        report.append({
            "fold": i,
            "train": f"Jan-{MONTH_NAMES[stop_month - 2]}",
            "early_stop": MONTH_NAMES[stop_month - 1],
            "scored": label(scored_months),
            "train_rows": int(tr.sum()), "scored_rows": int(sc.sum()),
            "scored_frauds": int(y[sc].sum()), "base_rate": round(float(y[sc].mean()), 4),
            "features": len(features), "winner": winner, "models": models,
        })
        line = "  ".join(f"{n}={models[n]['pr_auc']:.4f}({models[n]['trees'] or '-'})"
                         for n in names)
        print(f"  fold {i}: train {report[-1]['train']:<8} stop {report[-1]['early_stop']} "
              f"score {report[-1]['scored']:<8} n={sc.sum():>6,} frauds={y[sc].sum():>4} "
              f"base={y[sc].mean():.3f} | {line} | winner {winner}")

    summary = {
        n: {"mean_pr_auc": round(float(np.mean([f["models"][n]["pr_auc"] for f in report])), 4),
            "mean_lift": round(float(np.mean([f["models"][n]["lift"] for f in report])), 2),
            "mean_rank": round(float(np.mean([f["models"][n]["rank"] for f in report])), 2),
            "folds_won": sum(f["winner"] == n for f in report)}
        for n in names
    }
    print(f"\nPaired bootstrap vs {INCUMBENT} ({args.n_boot} rounds)...")
    paired = {n: paired_mean_diff(folds, n, INCUMBENT, args.n_boot)
              for n in names if n != INCUMBENT}

    out = ROOT / "results" / f"rolling_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rolling_origin.json").write_text(json.dumps({
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "design": "expanding window; boosters early-stop on the month before the scored "
                  "period, by scikit-learn average precision; hyper-parameters fixed (train.make_candidates); bot rows removed",
        "n_boot": args.n_boot, "folds": report, "summary": summary,
        f"paired_mean_diff_vs_{INCUMBENT}": paired,
    }, indent=2))
    chart(report, names, out / "rolling_origin.png")

    print(f"\n  {'model':<14} {'mean PR-AUC':>11} {'mean lift':>9} {'mean rank':>9} {'folds won':>9}")
    for n, s in summary.items():
        print(f"  {n:<14} {s['mean_pr_auc']:>11.4f} {s['mean_lift']:>9.2f} "
              f"{s['mean_rank']:>9.2f} {s['folds_won']:>9}")
    print(f"\n  Mean PR-AUC difference vs {INCUMBENT} over {len(report)} folds (95% CI):")
    for n, d in paired.items():
        print(f"  {n:<14} {d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
              f"ahead in {d['share_of_resamples_challenger_ahead']:.0%} of resamples")
    print(f"Wrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
