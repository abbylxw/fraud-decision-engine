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
EARLY_STOP_ROUNDS = 100  # LightGBM's early-stopping patience (a fit() callback)
EVAL_PERIOD = 10  # XGBoost, CatBoost: validation PR-AUC checked every 10 trees (see fit_candidate)


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

    CatBoost and XGBoost: their built-in PR-AUC metrics (``PRAUC``, ``aucpr``) disagree
    with scikit-learn's average precision when a model has few distinct scores. After one
    tree CatBoost reported 0.108 and XGBoost 0.088 where scikit-learn computes 0.076 and
    0.077, so their own early stopping saw tree 1 as the peak and stopped there (found by
    training/rolling_eval.py, 2026-11-15). So both train their full tree budget, validation
    PR-AUC is computed with scikit-learn every EVAL_PERIOD trees, and the model is cut back
    to the best checkpoint: CatBoost with ``shrink()``, XGBoost by setting the booster's
    ``best_iteration``, which ``predict_proba`` then honours. Same idea as early stopping
    with patience = the whole budget.
    """
    if name == "lightgbm":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[early_stopping(EARLY_STOP_ROUNDS, first_metric_only=True,
                                            verbose=False)])
    elif name == "catboost":
        model.fit(X_tr, y_tr, verbose=False)
        curve = [average_precision_score(y_va, p[:, 1])
                 for p in model.staged_predict_proba(X_va, eval_period=EVAL_PERIOD)]
        n_best = (int(np.argmax(curve)) + 1) * EVAL_PERIOD
        model.shrink(min(n_best, model.tree_count_))
    elif name == "xgboost":
        # A monotone constraint given as {feature: direction} is turned into one number
        # per column here. The dict form trains fine but cannot be re-applied to a model
        # loaded from disk: the API calls set_params(n_jobs=1) on the loaded model and
        # XGBoost then rejects the dict ("Invalid Parameter format"). The tuple form
        # survives. Features missing from this fit (an ablation baseline) are skipped.
        constraint = model.get_params().get("monotone_constraints")
        if isinstance(constraint, dict):
            model.set_params(monotone_constraints=tuple(
                int(constraint.get(c, 0)) for c in X_tr.columns))
        model.fit(X_tr, y_tr, verbose=False)
        counts = range(EVAL_PERIOD, model.n_estimators + 1, EVAL_PERIOD)
        curve = [average_precision_score(
                     y_va, model.predict_proba(X_va, iteration_range=(0, k))[:, 1])
                 for k in counts]
        model.get_booster().best_iteration = counts[int(np.argmax(curve))] - 1
    else:
        model.fit(X_tr, y_tr)
    return model


def best_trees(name, model):
    """Number of trees the early-stopped booster kept (None for non-boosters)."""
    if name == "xgboost":
        return int(model.best_iteration) + 1  # set in fit_candidate
    if name == "lightgbm":
        return int(model.best_iteration_)
    if name == "catboost":
        return int(model.tree_count_)  # after shrink() in fit_candidate
    return None


def fix_trees(name, model, n_trees):
    """Pin the tree count, for the refit on train+validation."""
    if name == "xgboost":
        model.set_params(n_estimators=n_trees)
    elif name == "lightgbm":
        model.set_params(n_estimators=n_trees)  # early stopping is a fit() callback, not a param
    elif name == "catboost":
        model.set_params(iterations=n_trees)
    return model
