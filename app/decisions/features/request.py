import json, bisect, pathlib
from app.data.data_models import Transaction

# Load age bin thresholds produced by the notebook (app/decisions/age_bins.json).
# Falls back to empty list (age_group = 0 for all) if the file doesn't exist yet.
_bins_path = pathlib.Path(__file__).parent.parent / "age_bins.json"
try:
    _raw_bins = json.loads(_bins_path.read_text())["bins"]
    # bins = [None, t1, t2, ..., None]; extract the finite interior thresholds
    _thresholds = [b for b in _raw_bins if b is not None]
except Exception:
    _thresholds = []


def get_days_since_signup(t: Transaction) -> int:
    if t.signup_time is None:
        return -999
    return max(0, (t.purchase_time - t.signup_time).days)


def get_age_group(t: Transaction) -> int:
    """Map age to integer bin index matching pd.cut(age, bins, labels) from training."""
    age = t.age if t.age is not None else -1
    if not _thresholds:
        return 0
    return bisect.bisect_right(_thresholds, age)


def get_age_account_gap(t: Transaction, days_since_signup: int) -> float:
    age = t.age if t.age is not None else -1
    return age - days_since_signup / 365


def get_hour_of_day(t: Transaction) -> int:
    return t.purchase_time.hour


def get_day_of_week(t: Transaction) -> int:
    return t.purchase_time.weekday()   # 0 = Monday, 6 = Sunday


def get_month_of_year(t: Transaction) -> int:
    return t.purchase_time.month


def get_hours_to_purchase(t: Transaction) -> float:
    """Hours from account signup to this purchase. -1.0 if signup_time is missing.
    Captures same-session fraud that days_since_signup rounds to 0."""
    if t.signup_time is None:
        return -1.0
    return (t.purchase_time - t.signup_time).total_seconds() / 3600
