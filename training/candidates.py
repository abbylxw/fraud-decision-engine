"""
Shared pieces for the model-candidate scripts (train.py, evaluate.py, tune.py).

Four tree candidates are compared: random forest (bagging) and three gradient boosters
(XGBoost, LightGBM, CatBoost). Each library spells "early-stop on a validation set" and
"how many trees did you keep" differently; the helpers below hide that so the scripts can
treat the candidates uniformly. See docs/model_candidates.md for how the four differ.
"""
import numpy as np
from lightgbm import early_stopping
from sklearn.metrics import average_precision_score

from app.decisions.rules import SIGNUP_BURST_SECONDS

BOOSTERS = ("xgboost", "lightgbm", "catboost")
EARLY_STOP_ROUNDS = 100  # XGBoost takes this in the constructor, LightGBM in fit()
CATBOOST_EVAL_PERIOD = 10  # CatBoost: validation PR-AUC checked every 10 trees (see fit_candidate)


def time_split(df, X_all, candidate_features, drop_always):
    """70/10/20 split by row order (rows are sorted by purchase_time), bot rows removed
    from train and validation, constant features dropped.

    Returns (tr, va, te, bot, features, dropped, i_te) with tr/va/te as boolean masks.
    """
    y = df["class"].to_numpy()
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
    constant = [c for c in candidate_features if X_all.loc[tr, c].nunique() == 1]
    dropped = {c: "constant on training rows" for c in constant} | drop_always
    features = [c for c in candidate_features if c not in dropped]
    print(f"  features: {len(candidate_features)} candidates -> {len(features)} kept "
          f"({len(constant)} constant, {len(drop_always)} dropped by design)")
    return tr, va, te, bot, features, dropped, i_te


def fit_candidate(name, model, X_tr, y_tr, X_va, y_va):
    """Fit; the boosters watch the validation set for early stopping.

    LightGBM: the model must be built with ``metric="average_precision"`` so the callback
    watches PR-AUC only (by default it also watches logloss, which with scale_pos_weight
    gets worse from tree 1 and stops training immediately).

    CatBoost: its own ``PRAUC`` metric disagrees with scikit-learn's average precision
    (0.108 vs 0.076 after one tree on this data) and its overfitting detector therefore
    stopped at tree 1. So CatBoost trains its full ``iterations`` budget, validation PR-AUC
    is computed with scikit-learn every CATBOOST_EVAL_PERIOD trees, and the model is
    shrunk to the best checkpoint. Same idea as early stopping, patience = whole budget.
    """
    if name == "lightgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[early_stopping(EARLY_STOP_ROUNDS, first_metric_only=True,
                                            verbose=False)])
    elif name == "catboost":
        model.fit(X_tr, y_tr, verbose=False)
        curve = [average_precision_score(y_va, p[:, 1])
                 for p in model.staged_predict_proba(X_va, eval_period=CATBOOST_EVAL_PERIOD)]
        n_best = (int(np.argmax(curve)) + 1) * CATBOOST_EVAL_PERIOD
        model.shrink(min(n_best, model.tree_count_))
    elif name in BOOSTERS:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model


def best_trees(name, model):
    """Number of trees the early-stopped booster kept (None for non-boosters)."""
    if name == "xgboost":
        return int(model.best_iteration) + 1
    if name == "lightgbm":
        return int(model.best_iteration_)
    if name == "catboost":
        return int(model.tree_count_)  # after shrink() in fit_candidate
    return None


def fix_trees(name, model, n_trees):
    """Turn early stopping off and pin the tree count, for the refit on train+validation."""
    if name == "xgboost":
        model.set_params(n_estimators=n_trees, early_stopping_rounds=None)
    elif name == "lightgbm":
        model.set_params(n_estimators=n_trees)  # early stopping is a fit() callback, not a param
    elif name == "catboost":
        model.set_params(iterations=n_trees)
    return model
