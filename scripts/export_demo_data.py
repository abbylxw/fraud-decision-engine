"""
Export the numbers behind the project-tour page to docs/demo/tour_data.json.

    python scripts/export_demo_data.py

Everything is aggregated: per-fold PR-AUC, the ablation summary, and cost curves (for each
of ~400 score thresholds: how many orders, legitimate orders, frauds and fraud dollars sit
at or above it, on validation and on test). No transaction-level rows are written. The
curves let the page re-pick the review/deny thresholds for any cost assumptions, the same
way training/policy.py does.

The previous model's curve needs its model file, which is read from git history
(PREVIOUS_MODEL_COMMIT); if that fails the page simply omits the comparison.
"""
import io
import json
import subprocess
import warnings

import joblib
import numpy as np
from sklearn.base import clone

from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import best_trees, fit_candidate, fix_trees, time_split
from training.train import CANDIDATE_FEATURES, DROP_ALWAYS, make_candidates

warnings.filterwarnings("ignore", category=UserWarning)
EVAL = ROOT / "docs" / "evaluation"
PREVIOUS_MODEL_COMMIT = "03eb99e"   # last commit that still served the 41-feature model


def curve(grid, p, y, value):
    fraud = y == 1
    return {
        "n": [int((p >= t).sum()) for t in grid],
        "legit": [int(((p >= t) & ~fraud).sum()) for t in grid],
        "fraud": [int(((p >= t) & fraud).sum()) for t in grid],
        "fraud_value": [round(float(value[(p >= t) & fraud].sum()), 2) for t in grid],
    }


def model_curves(p_va, p_te, y, value, va, te):
    grid = np.append(np.unique(np.quantile(p_va, np.linspace(0, 1, 401))), 1.01)
    return {"thresholds": [round(float(t), 5) for t in grid],
            "validation": curve(grid, p_va, y[va], value[va]),
            "test": curve(grid, p_te, y[te], value[te])}


def git_blob(path):
    return subprocess.check_output(["git", "show", f"{PREVIOUS_MODEL_COMMIT}:{path}"], cwd=ROOT)


def main():
    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    tr, va, te, _, features, _, _ = time_split(df, X_all, CANDIDATE_FEATURES, DROP_ALWAYS)
    X = X_all[features]

    # Served model, rebuilt the way train.py builds it: validation scores from the
    # train-only fit (thresholds are picked on those), test scores from the refit.
    model = make_candidates(float((y[tr] == 0).sum() / (y[tr] == 1).sum()))["xgboost"]
    fit_candidate("xgboost", model, X[tr], y[tr], X[va], y[va])
    final = fix_trees("xgboost", clone(model), best_trees("xgboost", model))
    final.fit(X[tr | va], y[tr | va])
    curves = {"served": model_curves(model.predict_proba(X[va])[:, 1],
                                     final.predict_proba(X[te])[:, 1], y, value, va, te)}
    try:
        old = joblib.load(io.BytesIO(git_blob("app/decisions/model.joblib")))
        old_features = json.loads(git_blob("app/decisions/model_features.json"))
        Xo = X_all[old_features]
        curves["previous"] = model_curves(old.predict_proba(Xo[va])[:, 1],
                                          old.predict_proba(Xo[te])[:, 1], y, value, va, te)
    except Exception as e:
        print(f"  previous model not exported ({e})")

    meta = json.loads((ROOT / "app/decisions/model_meta.json").read_text())
    before = json.loads((EVAL / "rolling_origin.json").read_text())
    after = json.loads((EVAL / "rolling_origin_42_features.json").read_text())
    ablation = json.loads((EVAL / "feature_ablation.json").read_text())

    def folds(r):
        return [{"scored": f["scored"], "base_rate": f["base_rate"],
                 **{m: f["models"][m]["pr_auc"] for m in f["models"]}} for f in r["folds"]]

    out = {
        "model_version": meta["model_version"],
        "test": {"orders": int(te.sum()), "frauds": int(y[te].sum()),
                 "fraud_value": round(float(value[te][y[te] == 1].sum()), 2),
                 "avg_fraud_value": round(float(value[te][y[te] == 1].mean()), 2)},
        "validation": {"orders": int(va.sum()), "frauds": int(y[va].sum()),
                       "fraud_value": round(float(value[va][y[va] == 1].sum()), 2)},
        "served_thresholds": {"review": meta["review_threshold"], "deny": meta["threshold"]},
        "top_features": meta["top_features"],
        "candidates_on_test": meta["candidates_on_test"],
        "served_test": meta["evaluation"]["new_model"]["test"],
        "rolling_before": {"folds": folds(before), "summary": before["summary"]},
        "rolling_after": {"folds": folds(after), "summary": after["summary"]},
        "ablation": {"coverage": ablation["coverage"],
                     "summary": ablation["summary"],
                     "folds": [{"scored": f["scored"], "baseline": f["baseline"]["pr_auc"],
                                "with_feature": f["+did_other_uid_all"]["pr_auc"]}
                               for f in ablation["folds"]]},
        "cost_curves": curves,
    }
    path = ROOT / "docs" / "demo" / "tour_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")

    # The tour page is one self-contained HTML file: the template with the data inlined.
    template = ROOT / "docs" / "demo" / "tour_template.html"
    if template.exists():
        page = template.read_text().replace("/*TOUR_DATA*/null", json.dumps(out, separators=(",", ":")))
        (ROOT / "docs" / "demo" / "tour.html").write_text(page)
        print("Wrote docs/demo/tour.html")


if __name__ == "__main__":
    main()
