"""
Retrain the served fraud model.

    python -m training.train                      # train + report only (writes results/)
    python -m training.train --ship               # also replace app/decisions/model.*

What changed vs the original notebook model, and why:
  * Bot-burst rows (purchase < SIGNUP_BURST_SECONDS after signup) are excluded from
    training. The `signup_burst` hard rule denies them before the model runs, so the
    model never scores them in production; training on them taught it "January = fraud".
  * `month_of_year` is dropped. The test months (Sep-Dec) never appear in training, so
    the feature can only memorise the January attack.
  * Features that are constant on the training rows are dropped. Every user in
    Fraud_Data.csv has one transaction, so all per-user counters are always 0.
  * The decision threshold is chosen on VALIDATION by dollar cost instead of hardcoded.

Split (same as training/evaluate.py): train = first 70% by time, validation = next 10%,
test = last 20%. Model choice, early stopping and threshold use validation; test is
scored once. The shipped model is then refit on train + validation (the first 80%, same
data window as the model it replaces) with the hyperparameters fixed.
"""
import argparse
import datetime as dt
import json
import subprocess
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from app.decisions.models import COUNTRY_ONEHOT
from app.decisions.models import THRESHOLD as OLD_THRESHOLD
from app.decisions.models import _model as deployed_model
from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import ROOT, build_feature_frame, load_transactions
from training.evaluate import SEED, score_report

warnings.filterwarnings("ignore", category=UserWarning)

DECISIONS = ROOT / "app" / "decisions"

# The 52 features the original model was trained on, except that the alphabetical
# ip_country_code is replaced by the grouped one-hot country columns (2026-11-06).
# Fixed here (not read from model_features.json) so re-running this script after
# shipping starts from the same candidate list instead of the already-pruned one.
CANDIDATE_FEATURES = [
    "purchase_value", "days_since_signup", "sex", "source", "browser",
    *COUNTRY_ONEHOT.values(),
    "hour_of_day", "day_of_week", "month_of_year",
    "did_uid_2h", "did_uid_24h", "did_uid_7d", "did_uid_30d", "did_uid_90d",
    "uid_did_2h", "uid_did_24h", "uid_did_7d", "uid_did_30d", "uid_did_90d",
    "ip_uid_2h", "ip_uid_24h", "ip_uid_7d",
    "purchase_cnt_24h", "purchase_amt_24h", "purchase_cnt_7d", "purchase_amt_7d",
    "purchase_cnt_30d", "purchase_amt_30d", "purchase_cnt_90d", "purchase_amt_90d",
    "purchase_success_cnt_24h", "purchase_success_amt_24h",
    "purchase_success_cnt_7d", "purchase_success_amt_7d",
    "purchase_success_cnt_30d", "purchase_success_amt_30d",
    "purchase_success_cnt_90d", "purchase_success_amt_90d",
    "did_uid_days_on_file", "pur_median_amt_90d",
    "did_txn_cnt_2h", "did_txn_cnt_24h", "did_txn_cnt_7d", "did_txn_cnt_30d",
    "time_since_last_txn_uid",
    "country_txn_count_24h", "country_txn_count_7d", "country_txn_count_30d",
    "amount_vs_user_avg_30d", "age_account_gap", "high_amount_vs_hist", "age_group",
]
DROP_ALWAYS = {"month_of_year": "test months never appear in training; memorises January"}

# The soft rate-limiter rules combine "elevated model score" with a velocity signal.
# "Elevated" is defined on this model's own score scale as the riskiest 3% of validation
# traffic, instead of a fixed probability: with scale_pos_weight the scores are not
# calibrated (the shipped XGBoost's median score is ~0.47, so a fixed 0.5 flags ~20%).
ELEVATED_ALERT_RATE = 0.03


def make_candidates(neg_pos_ratio):
    """Two candidates; the better one on validation PR-AUC ships.

    XGBoost is shallow on purpose. With the notebook settings (depth 7, lr 0.02) early
    stopping fired after 3 trees: the remaining signal is weak (mostly device sharing)
    and deep trees overfit it immediately. A small validation-only sweep (depth 2-5,
    lr 0.03-0.1, min_child_weight 1-100) put every shallow config at PR-AUC ~0.110-0.112,
    i.e. within noise; depth 3 / lr 0.05 / min_child_weight 50 stops at ~145 trees, the
    most stable of them.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.05, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg_pos_ratio,
            eval_metric="aucpr", early_stopping_rounds=100, random_state=SEED, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Threshold: the served model DENIES, so a false positive is a lost legit order.
# ---------------------------------------------------------------------------

def deny_cost(y, p, value, threshold, false_decline_cost):
    deny = p >= threshold
    missed_fraud = value[(y == 1) & ~deny].sum()
    false_declines = ((y == 0) & deny).sum()
    return float(missed_fraud + false_declines * false_decline_cost)


def pick_threshold(y, p, value, false_decline_cost):
    grid = np.append(np.unique(np.quantile(p, np.linspace(0, 1, 401))), 1.01)  # 1.01 = deny nothing
    costs = np.array([deny_cost(y, p, value, t, false_decline_cost) for t in grid])
    return float(grid[int(np.argmin(costs))])


def outcome(y, p, value, threshold, false_decline_cost):
    deny = p >= threshold
    return {
        "threshold": round(threshold, 4),
        "denied": int(deny.sum()),
        "deny_rate": round(float(deny.mean()), 4),
        "fraud_caught": int((deny & (y == 1)).sum()),
        "fraud_total": int(y.sum()),
        "false_declines": int((deny & (y == 0)).sum()),
        "fraud_dollars_stopped": round(float(value[deny & (y == 1)].sum()), 2),
        "total_cost": round(deny_cost(y, p, value, threshold, false_decline_cost), 2),
    }


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--false-decline-cost", type=float, default=10.0,
                    help="dollar cost of wrongly denying a legitimate order "
                         "(lost margin + customer friction)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--ship", action="store_true",
                    help="overwrite app/decisions/model.joblib, model_features.json, model_meta.json")
    args = ap.parse_args()

    print("Building features (velocity replay)...")
    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    gap = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().to_numpy()
    bot = (gap >= 0) & (gap < SIGNUP_BURST_SECONDS)

    n = len(df)
    i_va, i_te = int(n * 0.70), int(n * 0.80)
    idx = np.arange(n)
    tr = (idx < i_va) & ~bot
    va = (idx >= i_va) & (idx < i_te) & ~bot
    te = idx >= i_te
    print(f"  bot-burst rows removed: {bot.sum():,} (train {bot[:i_va].sum():,}, "
          f"validation {bot[i_va:i_te].sum():,}, test {bot[i_te:].sum():,})")
    print(f"  train rows={tr.sum():,} fraud={y[tr].mean():.1%} | "
          f"validation rows={va.sum():,} fraud={y[va].mean():.1%} | "
          f"test rows={te.sum():,} fraud={y[te].mean():.1%}")

    # --- Feature selection -------------------------------------------------
    constant = [c for c in CANDIDATE_FEATURES if X_all.loc[tr, c].nunique() == 1]
    dropped = {c: "constant on training rows" for c in constant} | DROP_ALWAYS
    features = [c for c in CANDIDATE_FEATURES if c not in dropped]
    print(f"  features: {len(CANDIDATE_FEATURES)} candidates -> {len(features)} kept "
          f"({len(constant)} constant, {len(DROP_ALWAYS)} dropped by design)")
    X = X_all[features]

    # --- Choose model on validation ---------------------------------------
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    candidates = make_candidates(ratio)
    val = {}
    for name, model in candidates.items():
        if name == "xgboost":
            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        else:
            model.fit(X[tr], y[tr])
        p_va = model.predict_proba(X[va])[:, 1]
        val[name] = {"model": model, "p_va": p_va,
                     "pr_auc": float(average_precision_score(y[va], p_va))}
        print(f"  {name:<14} validation PR-AUC={val[name]['pr_auc']:.4f}")
    best = max(val, key=lambda k: val[k]["pr_auc"])
    print(f"  selected: {best}")

    # --- Refit on train + validation with fixed hyperparameters -----------
    fit_rows = tr | va
    final = clone(candidates[best])
    if best == "xgboost":
        n_trees = int(val[best]["model"].best_iteration) + 1
        final.set_params(n_estimators=n_trees, early_stopping_rounds=None)
        print(f"  xgboost refit with n_estimators={n_trees} (validation early stop)")
    final.fit(X[fit_rows], y[fit_rows])

    # Threshold picked on validation scores of the train-only model, then applied to
    # the refit model. The refit sees 10% more data, so its scores shift slightly;
    # acceptable here, and the test-period numbers below are what the shipped model does.
    threshold = pick_threshold(y[va], val[best]["p_va"], value[va], args.false_decline_cost)
    print(f"  cost-based threshold (validation, false decline=${args.false_decline_cost:.0f}): "
          f"{threshold:.4f}")
    p_va_best = val[best]["p_va"]
    elevated = float(np.quantile(p_va_best, 1 - ELEVATED_ALERT_RATE))
    elevated_prec = float(y[va][p_va_best >= elevated].mean())
    print(f"  elevated-score level (top {ELEVATED_ALERT_RATE:.0%} of validation): {elevated:.4f} "
          f"(validation precision {elevated_prec:.1%})")

    # --- Score test once, new vs deployed ---------------------------------
    old_features = json.loads((DECISIONS / "model_features.json").read_text())
    p_new = final.predict_proba(X[te])[:, 1]
    p_old = deployed_model.predict_proba(X_all.loc[te, old_features])[:, 1]
    yt, vt = y[te], value[te]
    report = {
        "new_model": {
            "test": score_report(yt, p_new, args.n_boot),
            "at_new_threshold": outcome(yt, p_new, vt, threshold, args.false_decline_cost),
        },
        "previous_model": {
            "test": score_report(yt, p_old, args.n_boot),
            "at_old_threshold": outcome(yt, p_old, vt, OLD_THRESHOLD, args.false_decline_cost),
            "at_new_threshold_rule": outcome(
                yt, p_old, vt,
                pick_threshold(y[va], deployed_model.predict_proba(X_all.loc[va, old_features])[:, 1],
                               value[va], args.false_decline_cost),
                args.false_decline_cost),
        },
        "deny_nothing_cost": round(float(vt[yt == 1].sum()), 2),
    }
    elevated_test = p_new >= elevated
    report["new_model"]["elevated_score_on_test"] = {
        "share": round(float(elevated_test.mean()), 4),
        "precision": round(float(yt[elevated_test].mean()), 4),
    }

    # Ship gate: the new model must not be meaningfully worse on test. It removes
    # artefacts rather than adding signal, so "no worse" is the bar, not "better".
    new_pr, old_ci = report["new_model"]["test"]["pr_auc"], report["previous_model"]["test"]["pr_auc_ci95"]
    gate_ok = new_pr >= old_ci[0]
    report["ship_gate"] = {"rule": "new test PR-AUC >= previous model's 95% CI lower bound",
                           "passed": bool(gate_ok)}

    imp = dict(sorted(zip(features, final.feature_importances_), key=lambda kv: -kv[1])[:10])

    trained_at = dt.datetime.now().isoformat(timespec="seconds")
    meta = {
        "model_version": f"{best}-{dt.datetime.now():%Y%m%d%H%M%S}-{git_sha()}",
        "trained_at": trained_at,
        "algorithm": best,
        "params": {k: v for k, v in final.get_params().items()
                   if isinstance(v, (int, float, str, bool)) or v is None},
        "threshold": round(threshold, 6),
        "threshold_basis": {"chosen_on": "validation",
                            "false_decline_cost": args.false_decline_cost,
                            "missed_fraud_cost": "purchase_value"},
        "elevated_score_threshold": round(elevated, 6),
        "elevated_score_basis": {"chosen_on": "validation",
                                 "rule": f"top {ELEVATED_ALERT_RATE:.0%} of scores",
                                 "validation_precision": round(elevated_prec, 4),
                                 "used_by": "rules.rate_limiter_control (score + velocity combos)"},
        "training_data": {
            "source": "app/data/Fraud_Data.csv",
            "rows": int(fit_rows.sum()),
            "window": [str(df["purchase_time"].iloc[0]), str(df["purchase_time"].iloc[i_te - 1])],
            "excluded": f"{int(bot[:i_te].sum())} bot-burst rows "
                        f"(signup->purchase < {SIGNUP_BURST_SECONDS}s; denied by signup_burst rule)",
        },
        "features": features,
        "dropped_features": dropped,
        "top_features": {k: round(float(v), 4) for k, v in imp.items()},
        "validation_pr_auc": {k: round(v["pr_auc"], 4) for k, v in val.items()},
        "evaluation": report,
        "seed": SEED,
    }

    out = ROOT / "results" / f"train_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nTest period (scored once):")
    for k in ("new_model", "previous_model"):
        t = report[k]["test"]
        print(f"  {k:<15} PR-AUC={t['pr_auc']:.4f} {t['pr_auc_ci95']}  "
              f"ROC-AUC={t['roc_auc']:.4f} {t['roc_auc_ci95']}")
    print(f"  new model at threshold {threshold:.3f}: {report['new_model']['at_new_threshold']}")
    print(f"  old model at threshold {OLD_THRESHOLD}: {report['previous_model']['at_old_threshold']}")
    print(f"  top features: {meta['top_features']}")
    print(f"  ship gate passed: {gate_ok}")
    print(f"Wrote {out.relative_to(ROOT)}/model_meta.json")

    if args.ship:
        if not gate_ok:
            raise SystemExit("Ship gate failed; not replacing the served model.")
        joblib.dump(final, DECISIONS / "model.joblib")
        (DECISIONS / "model_features.json").write_text(json.dumps(features, indent=2))
        (DECISIONS / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print("Shipped: app/decisions/model.joblib, model_features.json, model_meta.json")


if __name__ == "__main__":
    main()
