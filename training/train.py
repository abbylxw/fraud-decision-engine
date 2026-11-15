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
  * The decision thresholds are chosen on VALIDATION by dollar cost instead of hardcoded.
    There are two: scores above the deny threshold are declined, scores between the
    review and deny thresholds get "review" (step-up check or manual look). See
    pick_thresholds below and docs/feature_ablation.md for why a middle band is needed.

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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from app.decisions.models import COUNTRY_ONEHOT
from app.decisions.models import REVIEW_THRESHOLD as OLD_REVIEW_THRESHOLD
from app.decisions.models import THRESHOLD as OLD_THRESHOLD
from app.decisions.models import _model as deployed_model
from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import ROOT, build_feature_frame, load_transactions
from training.candidates import (
    best_trees,
    fit_candidate,
    fix_trees,
    time_split,
)
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
    # All-time device transaction count (no window). Added 2026-11-15 after
    # training/feature_ablation.py: +0.033 mean rolling-origin PR-AUC [+0.028, +0.038].
    "did_txn_cnt_all",
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


INCUMBENT = "xgboost"


def make_candidates(neg_pos_ratio):
    """Four tree candidates: one bagging model and three gradient boosters.

    Hyper-parameters come from the validation-only sweep `python -m training.tune`
    (docs/evaluation/tune_sweep.json; summary in docs/model_candidates.md). Rule:
    each algorithm's validation winner. (An earlier sweep needed an extra "reject configs
    under 50 trees" rule; that was XGBoost's own early stopping misfiring, fixed in
    candidates.fit_candidate on 2026-11-15, and the same config wins without the rule.)
    Validation PR-AUC (sweep of 2026-11-15, with did_txn_cnt_all): RF 0.129, XGBoost 0.149,
    LightGBM 0.148, CatBoost 0.143; the top three configs of each booster are within 0.002
    of each other, i.e. noise. All boosters are shallow:
    the signal left after the January bot rows are removed is weak (mostly device sharing)
    and deep trees overfit it within a handful of rounds.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.05, min_child_weight=100,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg_pos_ratio,
            random_state=SEED, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=1000, num_leaves=7, learning_rate=0.03, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio, metric="average_precision",
            random_state=SEED, n_jobs=-1, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=1000, depth=3, learning_rate=0.05, l2_leaf_reg=10,
            scale_pos_weight=neg_pos_ratio, random_seed=SEED, verbose=False, thread_count=-1,
            allow_writing_files=False,
        ),
    }


def select_candidate(val, test_pr_auc, incumbent=INCUMBENT):
    """Which candidate becomes the served model.

    The incumbent algorithm (XGBoost, the only booster the API image ships with) keeps
    its place unless a challenger is clearly better: higher validation PR-AUC AND a test
    PR-AUC above the incumbent's 95% CI upper bound. A win inside the noise band is not
    worth a new runtime dependency. Returns (name, reason).
    """
    best = max(val, key=lambda k: val[k]["pr_auc"])
    if best == incumbent:
        return best, "highest validation PR-AUC"
    inc_hi = test_pr_auc[incumbent]["pr_auc_ci95"][1]
    if test_pr_auc[best]["pr_auc"] > inc_hi:
        return best, (f"validation PR-AUC above {incumbent} and test PR-AUC "
                      f"{test_pr_auc[best]['pr_auc']:.4f} > {incumbent} CI upper {inc_hi:.4f}")
    return incumbent, (f"{best} won validation ({val[best]['pr_auc']:.4f} vs "
                       f"{val[incumbent]['pr_auc']:.4f}) but its test PR-AUC "
                       f"{test_pr_auc[best]['pr_auc']:.4f} is inside {incumbent}'s 95% CI "
                       f"(upper {inc_hi:.4f}); kept the incumbent")


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


# ---------------------------------------------------------------------------
# Three-way policy: allow / review / deny.
#
# Cost of a policy on a set of orders, in dollars:
#   fraud that is allowed            -> its purchase value is lost
#   legitimate order that is denied  -> false_decline_cost (lost margin + an angry customer)
#   any order sent to review         -> review_cost (a step-up check or a minute of an analyst)
#   fraud sent to review             -> stopped with probability review_catch_rate, else lost
# A denied fraud costs nothing. Review is cheaper than a wrong decline, so it is the right
# action for scores that are risky but not risky enough for a decline to pay for itself.
# ---------------------------------------------------------------------------

def policy_cost(y, p, value, review_t, deny_t, false_decline_cost, review_cost, catch_rate=1.0):
    deny = p >= deny_t
    review = (p >= review_t) & ~deny
    fraud = y == 1
    return float(value[fraud & ~deny & ~review].sum()
                 + (1 - catch_rate) * value[fraud & review].sum()
                 + false_decline_cost * (deny & ~fraud).sum()
                 + review_cost * review.sum())


def pick_thresholds(y, p, value, false_decline_cost, review_cost, catch_rate=1.0):
    """(review_threshold, deny_threshold) with the lowest policy_cost, review <= deny.

    Same candidate grid as pick_threshold. 1.01 means "never": a deny threshold of 1.01
    is a model that only allows or reviews; equal thresholds mean no review band.
    The cost splits into a part that depends only on the review threshold and a part that
    depends only on the deny threshold, so the best pair is found with one pass (a running
    minimum) instead of trying all ~160,000 pairs."""
    grid = np.append(np.unique(np.quantile(p, np.linspace(0, 1, 401))), 1.01)
    fraud = y == 1
    above = [p >= t for t in grid]
    fraud_value = np.array([value[a & fraud].sum() for a in above])   # fraud $ at or above t
    legit = np.array([(a & ~fraud).sum() for a in above])             # legit orders at or above t
    count = np.array([a.sum() for a in above])
    # total = constant + review_part[r] + deny_part[d], for grid indices r <= d
    review_part = -catch_rate * fraud_value + review_cost * count
    deny_part = -(1 - catch_rate) * fraud_value + false_decline_cost * legit - review_cost * count
    best_r = np.array([int(np.argmin(review_part[:d + 1])) for d in range(len(grid))])
    d = int(np.argmin(review_part[best_r] + deny_part))
    return float(grid[best_r[d]]), float(grid[d])


def policy_outcome(y, p, value, review_t, deny_t, false_decline_cost, review_cost,
                   catch_rate=1.0):
    deny = p >= deny_t
    review = (p >= review_t) & ~deny
    fraud = y == 1
    return {
        "review_threshold": round(review_t, 4), "deny_threshold": round(deny_t, 4),
        "denied": int(deny.sum()), "deny_rate": round(float(deny.mean()), 4),
        "reviewed": int(review.sum()), "review_rate": round(float(review.mean()), 4),
        "fraud_total": int(fraud.sum()),
        "fraud_denied": int((deny & fraud).sum()),
        "fraud_reviewed": int((review & fraud).sum()),
        "fraud_allowed": int((~deny & ~review & fraud).sum()),
        "false_declines": int((deny & ~fraud).sum()),
        "legit_reviewed": int((review & ~fraud).sum()),
        "review_precision": round(float(y[review].mean()), 4) if review.any() else None,
        "deny_precision": round(float(y[deny].mean()), 4) if deny.any() else None,
        "total_cost": round(policy_cost(y, p, value, review_t, deny_t, false_decline_cost,
                                        review_cost, catch_rate), 2),
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
    ap.add_argument("--review-cost", type=float, default=5.0,
                    help="dollar cost of sending one order to review (step-up check or "
                         "analyst time), charged on every reviewed order")
    ap.add_argument("--review-catch-rate", type=float, default=1.0,
                    help="share of reviewed fraud that the review stops")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--ship", action="store_true",
                    help="overwrite app/decisions/model.joblib, model_features.json, model_meta.json")
    args = ap.parse_args()

    print("Building features (velocity replay)...")
    df = load_transactions()
    X_all = build_feature_frame(df)
    y = df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    tr, va, te, bot, features, dropped, i_te = time_split(df, X_all, CANDIDATE_FEATURES, DROP_ALWAYS)
    X = X_all[features]

    # --- Choose model on validation ---------------------------------------
    ratio = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    candidates = make_candidates(ratio)
    val = {}
    for name, model in candidates.items():
        fit_candidate(name, model, X[tr], y[tr], X[va], y[va])
        p_va = model.predict_proba(X[va])[:, 1]
        val[name] = {"model": model, "p_va": p_va, "trees": best_trees(name, model),
                     "pr_auc": float(average_precision_score(y[va], p_va))}
        print(f"  {name:<14} validation PR-AUC={val[name]['pr_auc']:.4f}"
              + (f"  trees={val[name]['trees']}" if val[name]["trees"] else ""))

    # Every candidate's train-only fit is scored on test ONCE, for the comparison table
    # and the "clearly better" selection rule. The chosen one is then refit below.
    candidate_test = {name: score_report(y[te], v["model"].predict_proba(X[te])[:, 1], args.n_boot)
                      for name, v in val.items()}
    best, why = select_candidate(val, candidate_test)
    print(f"  selected: {best} ({why})")
    if best != INCUMBENT:
        print(f"  NOTE: {best} is not in the api dependency group; add it to pyproject.toml "
              f"before --ship or the container cannot load model.joblib.")

    # --- Refit on train + validation with fixed hyperparameters -----------
    fit_rows = tr | va
    final = clone(candidates[best])
    if val[best]["trees"]:
        fix_trees(best, final, val[best]["trees"])
        print(f"  {best} refit with {val[best]['trees']} trees (validation early stop)")
    final.fit(X[fit_rows], y[fit_rows])

    # Threshold picked on validation scores of the train-only model, then applied to
    # the refit model. The refit sees 10% more data, so its scores shift slightly;
    # acceptable here, and the test-period numbers below are what the shipped model does.
    costs = (args.false_decline_cost, args.review_cost, args.review_catch_rate)
    review_threshold, threshold = pick_thresholds(y[va], val[best]["p_va"], value[va], *costs)
    print(f"  cost-based thresholds (validation; false decline=${args.false_decline_cost:.0f}, "
          f"review=${args.review_cost:.0f}, review catches {args.review_catch_rate:.0%}): "
          f"review >= {review_threshold:.4f}, deny >= {threshold:.4f}")
    p_va_best = val[best]["p_va"]
    elevated = float(np.quantile(p_va_best, 1 - ELEVATED_ALERT_RATE))
    elevated_prec = float(y[va][p_va_best >= elevated].mean())
    print(f"  elevated-score level (top {ELEVATED_ALERT_RATE:.0%} of validation): {elevated:.4f} "
          f"(validation precision {elevated_prec:.1%})")

    # --- Score test once, new vs deployed ---------------------------------
    old_features = json.loads((DECISIONS / "model_features.json").read_text())
    p_new = final.predict_proba(X[te])[:, 1]
    p_old = deployed_model.predict_proba(X_all.loc[te, old_features])[:, 1]
    p_old_va = deployed_model.predict_proba(X_all.loc[va, old_features])[:, 1]
    yt, vt = y[te], value[te]
    report = {
        "new_model": {
            "test": score_report(yt, p_new, args.n_boot),
            "policy": policy_outcome(yt, p_new, vt, review_threshold, threshold, *costs),
            # Deny-only policy with a single threshold, for comparison with earlier runs.
            "deny_only": outcome(
                yt, p_new, vt,
                pick_threshold(y[va], val[best]["p_va"], value[va], args.false_decline_cost),
                args.false_decline_cost),
        },
        "previous_model": {
            "test": score_report(yt, p_old, args.n_boot),
            # As it is served today (its own thresholds from model_meta.json).
            "as_served": policy_outcome(yt, p_old, vt, OLD_REVIEW_THRESHOLD, OLD_THRESHOLD, *costs),
            # Given the same freedom as the new model: its best review/deny pair on validation.
            "best_policy": policy_outcome(
                yt, p_old, vt, *pick_thresholds(y[va], p_old_va, value[va], *costs), *costs),
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
    # Second condition (2026-11-15): the new policy must not cost more on test than what is
    # served today. A model once passed the PR-AUC gate while being worth $0 at the decline
    # threshold; ranking quality alone does not say a model earns its keep.
    new_pr, old_ci = report["new_model"]["test"]["pr_auc"], report["previous_model"]["test"]["pr_auc_ci95"]
    new_cost = report["new_model"]["policy"]["total_cost"]
    served_cost = report["previous_model"]["as_served"]["total_cost"]
    pr_ok, cost_ok = new_pr >= old_ci[0], new_cost <= served_cost
    gate_ok = pr_ok and cost_ok
    report["ship_gate"] = {
        "rule": "new test PR-AUC >= previous model's 95% CI lower bound AND new policy's "
                "test cost <= the served policy's test cost (same cost parameters)",
        "pr_auc_ok": bool(pr_ok), "cost_ok": bool(cost_ok),
        "new_policy_cost": new_cost, "served_policy_cost": served_cost,
        "passed": bool(gate_ok)}

    # How much the conclusion depends on the two assumptions about review.
    sensitivity = []
    for rc in (3.0, 5.0, 8.0):
        for cr in (1.0, 0.8, 0.6):
            c = (args.false_decline_cost, rc, cr)
            new = policy_outcome(yt, p_new, vt, *pick_thresholds(y[va], val[best]["p_va"], value[va], *c), *c)
            old = policy_outcome(yt, p_old, vt, *pick_thresholds(y[va], p_old_va, value[va], *c), *c)
            sensitivity.append({"review_cost": rc, "review_catch_rate": cr,
                                "new": {k: new[k] for k in ("review_rate", "deny_rate", "total_cost")},
                                "served_model_best_policy": {
                                    k: old[k] for k in ("review_rate", "deny_rate", "total_cost")}})
    report["review_sensitivity"] = sensitivity

    imp = dict(sorted(zip(features, final.feature_importances_), key=lambda kv: -kv[1])[:10])

    trained_at = dt.datetime.now().isoformat(timespec="seconds")
    meta = {
        "model_version": f"{best}-{dt.datetime.now():%Y%m%d%H%M%S}-{git_sha()}",
        "trained_at": trained_at,
        "algorithm": best,
        "params": {k: v for k, v in final.get_params().items()
                   if isinstance(v, (int, float, str, bool)) or v is None},
        "threshold": round(threshold, 6),
        "review_threshold": round(review_threshold, 6),
        "threshold_basis": {"chosen_on": "validation",
                            "decisions": "score >= threshold: deny; "
                                         "review_threshold <= score < threshold: review",
                            "false_decline_cost": args.false_decline_cost,
                            "review_cost": args.review_cost,
                            "review_catch_rate": args.review_catch_rate,
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
        "candidates_on_test": {k: {"trees": val[k]["trees"], "pr_auc": t["pr_auc"],
                                   "pr_auc_ci95": t["pr_auc_ci95"], "roc_auc": t["roc_auc"]}
                               for k, t in candidate_test.items()},
        "selection_rule": {"incumbent": INCUMBENT, "selected": best, "reason": why,
                           "rule": "highest validation PR-AUC, unless the winner is a challenger "
                                   "whose test PR-AUC is inside the incumbent's 95% CI"},
        "evaluation": report,
        "seed": SEED,
    }

    out = ROOT / "results" / f"train_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nCandidates on test (train-only fits):")
    for k, t in candidate_test.items():
        print(f"  {k:<14} PR-AUC={t['pr_auc']:.4f} {t['pr_auc_ci95']}  ROC-AUC={t['roc_auc']:.4f}")
    print("\nTest period (refit model vs deployed):")
    for k in ("new_model", "previous_model"):
        t = report[k]["test"]
        print(f"  {k:<15} PR-AUC={t['pr_auc']:.4f} {t['pr_auc_ci95']}  "
              f"ROC-AUC={t['roc_auc']:.4f} {t['roc_auc_ci95']}")
    print(f"\nPolicy on test (cost of doing nothing ${report['deny_nothing_cost']:,.0f}):")
    cols = ("review_rate", "deny_rate", "fraud_reviewed", "fraud_denied", "fraud_allowed",
            "legit_reviewed", "false_declines", "review_precision", "total_cost")
    for label, o in (("new model, new thresholds", report["new_model"]["policy"]),
                     ("served model, as served", report["previous_model"]["as_served"]),
                     ("served model, its best pair", report["previous_model"]["best_policy"])):
        print(f"  {label:<28} review>={o['review_threshold']:.3f} deny>={o['deny_threshold']:.3f}  "
              + "  ".join(f"{c}={o[c]}" for c in cols))
    print("\nSensitivity (total test cost; thresholds re-picked on validation for each row):")
    for r in sensitivity:
        print(f"  review ${r['review_cost']:.0f}, catches {r['review_catch_rate']:.0%}: "
              f"new ${r['new']['total_cost']:,.0f} (review {r['new']['review_rate']:.1%}, "
              f"deny {r['new']['deny_rate']:.1%})  vs served model's best "
              f"${r['served_model_best_policy']['total_cost']:,.0f}")
    print(f"  top features: {meta['top_features']}")
    print(f"  ship gate: PR-AUC ok={pr_ok}, cost ok={cost_ok} "
          f"(${new_cost:,.0f} vs served ${served_cost:,.0f}) -> passed: {gate_ok}")
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
