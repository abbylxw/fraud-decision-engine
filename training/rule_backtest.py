"""
Backtest the decision rules on Fraud_Data.csv.

    python -m training.rule_backtest [--false-decline-cost 10]

Runs the real functions from app/decisions/rules.py on every replayed row (same features
as training, same 70/10/20 time split as training/train.py) and reports, per rule, on the
validation and test periods:
  fires / precision  how often it would deny, and what share of those are fraud
                     (Wilson 95% interval)
  alone              fraud caught, fraud $ stopped and false declines if it ran by itself
  on top of model    what it adds to the served model's denies (score >= THRESHOLD):
                     extra fraud caught, extra false declines, change in dollar cost
Cost is the same as the ship report (train.py): missed fraud $ + false declines x cost.

Also checks whether combining a rule condition with the model score helps: "deny when in
segment S and score >= t", with t below the global threshold (a segment threshold). This
needs honest validation scores, and the served model was refit on train + validation, so a
model with the served hyperparameters is fit on the train period only. Both the global
threshold and each segment's t are chosen on validation; test is scored once, with a
bootstrap 95% CI on the cost change.

Writes results/rule_backtest_<ts>/rules.json. Never changes any served file.
"""
import argparse
import datetime as dt
import json
import math
from types import SimpleNamespace

import numpy as np
from xgboost import XGBClassifier

from app.decisions.models import THRESHOLD
from app.decisions.models import _model as deployed_model
from app.decisions.rules import (
    _OFAC_BLOCKED,
    SIGNUP_BURST_SECONDS,
    age_limit,
    ip_country_block,
    rate_limiter_control,
    signup_burst,
)
from training.build_features import ROOT, SERVED_FEATURES, build_feature_frame, load_transactions
from training.train import deny_cost, pick_threshold

SHADOW_BRANCHES = ["device_many_users_90d", "user_many_devices_90d", "ip_many_users_7d",
                   "elevated_score_purchases_24h", "elevated_score_devices_24h"]
SHADOW_INPUTS = ["did_uid_days_on_file", "did_uid_90d", "uid_did_90d", "ip_uid_7d",
                 "purchase_cnt_24h", "uid_did_24h"]


def wilson(k, n):
    """95% interval for a proportion k/n; stays sensible for tiny n."""
    if n == 0:
        return [None, None]
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def rule_flags(df, X, scores):
    """Run each rule function row by row. Rules only read attributes, so a SimpleNamespace
    per row stands in for the Transaction / Features objects the API passes."""
    flags = {name: np.zeros(len(df), dtype=bool)
             for name in ["signup_burst", "age_limit", "ip_country_block", *SHADOW_BRANCHES]}
    cols = df[["signup_time", "purchase_time", "age", "ip_country"]].to_dict("records")
    shadow = X[SHADOW_INPUTS].to_dict("records")
    for i, (row, feats, score) in enumerate(zip(cols, shadow, scores)):
        t = SimpleNamespace(**row)
        f = SimpleNamespace(**feats, ip_country=row["ip_country"])
        flags["signup_burst"][i] = signup_burst(t) == "deny"
        flags["age_limit"][i] = age_limit(t, f) == "deny"
        flags["ip_country_block"][i] = ip_country_block(t, f) == "deny"
        for branch in rate_limiter_control(t, f, score):
            flags[branch][i] = True
    return flags


def report(rule, model_deny, y, value, cost):
    fires = int(rule.sum())
    tp = int((rule & (y == 1)).sum())
    new = rule & ~model_deny
    new_tp, new_fp = int((new & (y == 1)).sum()), int((new & (y == 0)).sum())
    new_fraud_dollars = float(value[new & (y == 1)].sum())
    return {
        "fires": fires,
        "fire_rate": round(fires / len(y), 5),
        "precision": round(tp / fires, 4) if fires else None,
        "precision_ci95": wilson(tp, fires),
        "alone": {"fraud_caught": tp, "fraud_dollars_stopped": round(float(value[rule & (y == 1)].sum()), 2),
                  "false_declines": fires - tp},
        "on_top_of_model": {"extra_fraud_caught": new_tp, "extra_false_declines": new_fp,
                            "extra_fraud_dollars_stopped": round(new_fraud_dollars, 2),
                            # negative = cheaper than the model alone
                            "cost_change": round(new_fp * cost - new_fraud_dollars, 2)},
    }


def segment_thresholds(df, X, y, value, tr, va, te, cost, n_boot=500):
    """For each segment, the best lower score cut-off on validation, scored on test."""
    meta = json.loads((ROOT / "app/decisions/model_meta.json").read_text())
    params = {k: v for k, v in meta["params"].items() if v is not None and k != "missing"}
    s = XGBClassifier(**params).fit(X.loc[tr, SERVED_FEATURES], y[tr]) \
        .predict_proba(X[SERVED_FEATURES])[:, 1]
    T = pick_threshold(y[va], s[va], value[va], cost)
    base = s >= T

    def cost_of(deny, rows):
        return deny_cost(y[rows], deny[rows].astype(float), value[rows], 0.5, cost)

    segments = {
        "ofac_country": df["ip_country"].isin(_OFAC_BLOCKED).to_numpy(),
        "russia": (df["ip_country"] == "Russian Federation").to_numpy(),
        "device_shared (did_uid_90d>=1)": X["did_uid_90d"].to_numpy() >= 1,
        "ip_shared (ip_uid_7d>=1)": X["ip_uid_7d"].to_numpy() >= 1,
        "unknown_ip_country": (df["ip_country"] == "Unknown").to_numpy(),
        "new_account (<1 day)": X["days_since_signup"].to_numpy() < 1,
        "high_amount (>= $100)": value >= 100,
    }
    rng = np.random.default_rng(42)
    te_idx = np.where(te)[0]
    boots = [rng.choice(te_idx, len(te_idx)) for _ in range(n_boot)]
    out = {"model": "served hyperparameters, fit on train period only",
           "global_threshold": round(T, 4), "test_cost_model_alone": cost_of(base, te),
           "segments": {}}
    for name, seg in segments.items():
        res = {"test_share": round(float(seg[te].mean()), 4),
               "test_fraud_rate": round(float(y[te][seg[te]].mean()), 4) if seg[te].any() else None,
               "cut_off": None}
        best_change = 0.0
        if (va & seg).any():
            for t in np.unique(np.quantile(s[va & seg], np.linspace(0, 1, 101))):
                if t < T:
                    change = cost_of(base | (seg & (s >= t)), va) - cost_of(base, va)
                    if change < best_change:
                        res["cut_off"], best_change = round(float(t), 4), change
        if res["cut_off"] is not None:
            deny = base | (seg & (s >= res["cut_off"]))
            new = deny & ~base & te
            bs = [cost_of(deny, b) - cost_of(base, b) for b in boots]
            res |= {"validation_cost_change": round(best_change, 2),
                    "test_cost_change": round(cost_of(deny, te) - cost_of(base, te), 2),
                    "test_cost_change_ci95": [round(float(np.percentile(bs, 2.5)), 2),
                                              round(float(np.percentile(bs, 97.5)), 2)],
                    "extra_fraud_caught": int((new & (y == 1)).sum()),
                    "extra_false_declines": int((new & (y == 0)).sum())}
        out["segments"][name] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--false-decline-cost", type=float, default=10.0)
    args = ap.parse_args()
    cost = args.false_decline_cost

    print("Building features (velocity replay)...")
    df = load_transactions()
    X = build_feature_frame(df)
    y = df["class"].to_numpy()
    value = df["purchase_value"].to_numpy(dtype=float)
    scores = deployed_model.predict_proba(X[SERVED_FEATURES])[:, 1]

    n = len(df)
    i_va, i_te = int(n * 0.70), int(n * 0.80)
    periods = {"validation": slice(i_va, i_te), "test": slice(i_te, n)}

    print("Running rules on every row...")
    flags = rule_flags(df, X, scores)

    out = {"false_decline_cost": cost, "served_threshold": THRESHOLD,
           "signup_burst_seconds": SIGNUP_BURST_SECONDS, "periods": {}}
    for period, s in periods.items():
        yy, vv = y[s], value[s]
        model_deny = scores[s] >= THRESHOLD
        res = {"rows": int(len(yy)), "fraud": int(yy.sum()),
               "model_alone": report(model_deny, np.zeros_like(model_deny), yy, vv, cost),
               "rules": {name: report(f[s], model_deny, yy, vv, cost) for name, f in flags.items()}}
        out["periods"][period] = res

    # Whole-file check of the bot rule: does it still deny only fraud?
    sb = flags["signup_burst"]
    out["signup_burst_all_rows"] = {"fires": int(sb.sum()), "fraud": int(y[sb].sum()),
                                    "months": sorted(set(df.loc[sb, "purchase_time"].dt.month))}

    # Rule condition + model score. Bot rows are left out of the fit, as in train.py.
    print("Checking segment thresholds (fits one model on the train period)...")
    gap = (df["purchase_time"] - df["signup_time"]).dt.total_seconds().to_numpy()
    bot = (gap >= 0) & (gap < SIGNUP_BURST_SECONDS)
    idx = np.arange(n)
    tr, va, te = (idx < i_va) & ~bot, (idx >= i_va) & (idx < i_te) & ~bot, idx >= i_te
    out["segment_thresholds"] = segment_thresholds(df, X, y, value, tr, va, te, cost)

    folder = ROOT / "results" / f"rule_backtest_{dt.datetime.now():%Y%m%dT%H%M%S}"
    folder.mkdir(parents=True)
    (folder / "rules.json").write_text(json.dumps(out, indent=2, default=int))

    for period, res in out["periods"].items():
        print(f"\n{period}: rows={res['rows']:,} fraud={res['fraud']:,}  (+cost = worse)")
        print(f"  {'rule':<30}{'fires':>7}{'prec':>7}  {'95% CI':<16}{'+fraud':>7}{'+FD':>6}{'cost chg':>10}")
        for name, r in [("MODEL (alone)", res["model_alone"]), *res["rules"].items()]:
            ci = r["precision_ci95"]
            ci_txt = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci[0] is not None else "-"
            top = r["on_top_of_model"]
            prec = f"{r['precision']:.3f}" if r["precision"] is not None else "-"
            print(f"  {name:<30}{r['fires']:>7}{prec:>7}  {ci_txt:<16}"
                  f"{top['extra_fraud_caught']:>7}{top['extra_false_declines']:>6}{top['cost_change']:>10.0f}")
    print(f"\nsignup_burst on all rows: {out['signup_burst_all_rows']}")
    st = out["segment_thresholds"]
    print(f"\nSegment + score (global threshold {st['global_threshold']:.3f}, chosen on validation)")
    print(f"  {'segment':<32}{'test %':>7}{'fraud %':>8}{'cut-off':>9}{'test cost chg':>15}  95% CI")
    for name, r in st["segments"].items():
        fraud = f"{100 * r['test_fraud_rate']:.1f}" if r["test_fraud_rate"] is not None else "-"
        head = f"  {name:<32}{100 * r['test_share']:>7.1f}{fraud:>8}"
        if r["cut_off"] is None:
            print(f"{head}{'none':>9}   no lower cut-off helps on validation")
        else:
            ci = r["test_cost_change_ci95"]
            print(f"{head}{r['cut_off']:>9.3f}{r['test_cost_change']:>15.0f}  [{ci[0]:.0f}, {ci[1]:.0f}]")
    print(f"\nWrote {folder.relative_to(ROOT)}/rules.json")


if __name__ == "__main__":
    main()
