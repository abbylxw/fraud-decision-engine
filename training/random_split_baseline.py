"""
Reproduce the "precision 0.98 / recall 0.54" Random Forest that public Fraud_Data write-ups
report, and show where the recall comes from.

    python -m training.random_split_baseline

Recipe copied from those write-ups: calendar parts of signup_time / purchase_time (year, month,
day, hour, minute, second) as numeric features, label-encoded device_id / browser / source / sex,
a random 80/20 train_test_split(random_state=42), RandomForest(100 trees). SMOTE is left out: it
only moves the operating point, not the ranking.

The script then splits the test recall into bot rows (purchase < 60 s after signup; the January
attack, 54% of all fraud) and everything else, and repeats the fit with a time split. See
README "Why these numbers look low next to other Fraud_Data projects".
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.decisions.rules import SIGNUP_BURST_SECONDS
from training.build_features import DATA_DIR


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.copy()
    for col in ["signup_time", "purchase_time"]:
        for part in ["year", "month", "day", "hour", "minute", "second"]:
            X[f"{col}_{part}"] = getattr(X[col].dt, part)
        X = X.drop(columns=[col])
    for col in ["sex", "browser", "source", "device_id"]:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    y = X.pop("class")
    return X.drop(columns=["user_id"]), y


def report(label, y, pred, prob):
    tp = (pred == 1) & (y == 1)
    prec = tp.sum() / max((pred == 1).sum(), 1)
    print(f"{label:46s} precision={prec:.3f} recall={tp.sum() / y.sum():.3f} "
          f"ROC-AUC={roc_auc_score(y, prob):.3f} PR-AUC={average_precision_score(y, prob):.3f}")


def main():
    df = pd.read_csv(DATA_DIR / "Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
    gap = (df.purchase_time - df.signup_time).dt.total_seconds()
    bot = (gap >= 0) & (gap < SIGNUP_BURST_SECONDS)
    X, y = prepare(df)

    # 1) the public recipe: random split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    bte = bot.loc[yte.index].to_numpy()
    yte = yte.to_numpy()
    print(f"random split: test fraud={yte.sum()}, bot rows among them={int((bte & (yte == 1)).sum())} "
          f"({(bte & (yte == 1)).sum() / yte.sum():.1%})")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred, prob = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
    report("  all test rows", yte, pred, prob)
    tp = (pred == 1) & (yte == 1)
    print(f"  recall on bot fraud={tp[bte].sum() / (yte[bte] == 1).sum():.3f}   "
          f"recall on non-bot fraud={tp[~bte].sum() / (yte[~bte] == 1).sum():.3f}")
    report("  non-bot test rows only", yte[~bte], pred[~bte], prob[~bte])

    # 2) same recipe, time split (first 80% by purchase_time -> train)
    order = df.purchase_time.sort_values(kind="stable").index
    cut = int(len(df) * 0.8)
    tr, te = order[:cut], order[cut:]
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X.loc[tr], y.loc[tr])
    report("time split (test = last 20%, no bot rows)", y.loc[te].to_numpy(),
           rf2.predict(X.loc[te]), rf2.predict_proba(X.loc[te])[:, 1])


if __name__ == "__main__":
    main()
