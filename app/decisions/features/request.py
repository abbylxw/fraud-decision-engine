import bisect
from datetime import timezone

from app.data.data_models import Transaction


def event_ts(t: Transaction) -> float:
    """The transaction's own time as a Unix timestamp - the clock for every velocity window.

    Training counts windows back from purchase_time, so serving must too (not the server
    clock). A naive datetime (no timezone, which is what the client sends) is treated as
    UTC, because that is what pandas does in training; Python's .timestamp() would
    otherwise read it as the server's LOCAL time and shift every window by hours.
    """
    pt = t.purchase_time
    if pt.tzinfo is None:
        pt = pt.replace(tzinfo=timezone.utc)
    return pt.timestamp()


def get_days_since_signup(t: Transaction) -> int:
    if t.signup_time is None:
        return -999
    return max(0, (t.purchase_time - t.signup_time).days)


# Fixed thresholds and labels for days_since_signup binning.
# Mirrors the pd.cut bins used in the notebook (right-inclusive, so 7 → '0-7d', 8 → '8-14d').
_SIGNUP_THRESHOLDS = [7, 14, 30, 90, 180, 360, 720]
_SIGNUP_LABELS     = ['0-7d', '8-14d', '15-30d', '31-90d', '91-180d', '181-360d', '361-720d', '720+d']


def get_days_since_signup_group(days_since_signup: int) -> str:
    """Map days_since_signup to a human-readable bin label.
    Missing (days=-999) is treated as 0 days (brand-new account → '0-7d')."""
    d = max(0, days_since_signup)  # clip -999 sentinel to 0
    return _SIGNUP_LABELS[bisect.bisect_left(_SIGNUP_THRESHOLDS, d)]


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
