"""
Validation-only hyper-parameter sweep for the four tree candidates.

    python -m training.tune            # full grids, ~5-10 min
    python -m training.tune --quick    # 2 configs per algorithm, smoke test

Each config is fit on the training rows and scored by PR-AUC on the validation rows
(the boosters early-stop on validation as well). Test rows are never touched here.
Writes results/tune_<timestamp>/sweep.json and prints the top 3 per algorithm; the
winning settings are copied by hand into train.make_candidates / evaluate.make_models
with a comment naming the sweep folder, so the served hyper-parameters stay explicit.
"""
import argparse
import datetime as dt
import itertools
import json
import time
import warnings

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import best_trees, fit_candidate, time_split
from training.evaluate import SEED
from training.train import CANDIDATE_FEATURES, DROP_ALWAYS

warnings.filterwarnings("ignore", category=UserWarning)


def grids(ratio):
    """name -> (constructor(**cfg), list of cfg dicts). Fixed settings live in the
    constructor; only the swept ones appear in cfg."""
    def rf(**c):
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      n_jobs=-1, random_state=SEED, **c)

    def xgb(**c):
        return XGBClassifier(n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=ratio, random_state=SEED, n_jobs=-1, **c)

    def lgbm(**c):
        return LGBMClassifier(n_estimators=1000, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=ratio,
                              metric="average_precision",  # what early stopping watches
                              random_state=SEED, n_jobs=-1, verbose=-1, **c)

    def cat(**c):
        return CatBoostClassifier(iterations=1000, scale_pos_weight=ratio, random_seed=SEED,
                                  verbose=False, thread_count=-1, allow_writing_files=False, **c)

    def product(**space):
        keys = list(space)
        return [dict(zip(keys, vals)) for vals in itertools.product(*space.values())]

    return {
        "random_forest": (rf, product(max_depth=[8, 12, 16], min_samples_leaf=[20, 50, 100])),
        "xgboost": (xgb, product(max_depth=[2, 3, 5], learning_rate=[0.03, 0.05, 0.1],
                                 min_child_weight=[1, 50, 100])),
        "lightgbm": (lgbm, product(num_leaves=[7, 15, 31], learning_rate=[0.03, 0.05],
                                   min_child_samples=[20, 50, 100])),
        "catboost": (cat, product(depth=[3, 4, 6], learning_rate=[0.03, 0.05],
                                  l2_leaf_reg=[3, 10])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="2 configs per algorithm")
    args = ap.parse_args()

    print("Building features (velocity replay)...")
    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    tr, va, te, bot, features, dropped, _ = time_split(df, X_all, CANDIDATE_FEATURES, DROP_ALWAYS)
    X = X_all[features]
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())

    sweep = {}
    for name, (make, cfgs) in grids(ratio).items():
        if args.quick:
            cfgs = cfgs[:2]
        rows = []
        for cfg in cfgs:
            t0 = time.perf_counter()
            model = fit_candidate(name, make(**cfg), X[tr], y[tr], X[va], y[va])
            p_va = model.predict_proba(X[va])[:, 1]
            rows.append({
                **cfg,
                "validation_pr_auc": round(float(average_precision_score(y[va], p_va)), 4),
                "trees": best_trees(name, model),
                "fit_seconds": round(time.perf_counter() - t0, 1),
            })
            print(f"  {name:<14} {cfg} -> PR-AUC={rows[-1]['validation_pr_auc']:.4f} "
                  f"trees={rows[-1]['trees']} ({rows[-1]['fit_seconds']}s)")
        rows.sort(key=lambda r: -r["validation_pr_auc"])
        sweep[name] = rows

    out = ROOT / "results" / f"tune_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(
        {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
         "selection_metric": "validation PR-AUC (test rows never used)",
         "train_rows": int(tr.sum()), "validation_rows": int(va.sum()),
         "features": features, "sweep": sweep}, indent=2))

    print("\nTop 3 per algorithm (validation PR-AUC):")
    for name, rows in sweep.items():
        for r in rows[:3]:
            cfg = {k: v for k, v in r.items()
                   if k not in ("validation_pr_auc", "trees", "fit_seconds")}
            print(f"  {name:<14} {r['validation_pr_auc']:.4f}  trees={r['trees']}  {cfg}")
    print(f"Wrote {out.relative_to(ROOT)}/sweep.json")


if __name__ == "__main__":
    main()
