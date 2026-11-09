from app.data.data_models import Features, Transaction
from app.decisions.models import ELEVATED_SCORE, THRESHOLD


def age_limit(transaction: Transaction, feature: Features):
    """Data check only: deny an age that can't be real. The request requires age (HTTP 422
    if missing); the None check only guards callers that build a Transaction without
    validation. The old 'age >= 80 + new account' clause was removed (rule-logic-review):
    it had no supporting data and denied by age alone."""
    age = transaction.age
    if age is not None and (age < 0 or age >= 110):
        return 'deny'
    return 'allow'

# OFAC-sanctioned countries — transactions originating here are denied.
# ip_country comes from the IP table (IpAddress_to_Country.csv), which uses its own
# spellings; the short names never match it. tests/test_rules.py checks every name here
# against the table so a spelling mismatch fails loudly.
_OFAC_BLOCKED = frozenset({
    'Cuba', 'Iran (ISLAMIC Republic Of)', 'Russian Federation', 'Syrian Arab Republic',
    # Short aliases, and North Korea, which has no rows in the IP table.
    'Iran', 'Russia', 'Syria', 'North Korea',
})

def ip_country_block(transaction: Transaction, feature: Features):
    country = feature.ip_country
    if country in _OFAC_BLOCKED:
        return 'deny'
    # No 'Unknown IP + new device' branch (removed in rule-logic-review): it denied ~15% of
    # traffic with no fraud lift. ipc_Unknown is still a model input.
    return 'allow'

# Signup-to-purchase faster than this is scripted, not human.
# In Fraud_Data.csv all 7,600 purchases within 1s of signup are fraud (a January 2015
# bot attack), and the fastest legitimate purchase is 137s after signup. 60s leaves
# headroom for slow bots while staying well clear of real customers.
SIGNUP_BURST_SECONDS = 60


def signup_burst(transaction: Transaction) -> str:
    """Deny purchases made within SIGNUP_BURST_SECONDS of account signup.
    Handled as a rule rather than a model feature: the pattern only occurs in
    January, so a model trained on it learns 'January = fraud' and nothing that
    carries forward (see training/train.py)."""
    if transaction.signup_time is None:
        return 'allow'
    gap = (transaction.purchase_time - transaction.signup_time).total_seconds()
    if 0 <= gap < SIGNUP_BURST_SECONDS:
        return 'deny'
    return 'allow'


def model_score_threshold(score: float) -> str:
    """Deny if the model's fraud probability meets or exceeds the threshold."""
    return 'deny' if score >= THRESHOLD else 'allow'


def rate_limiter_control(transaction: Transaction, feature: Features, score: float) -> list[str]:
    """SHADOW MODE: never denies. Returns the names of the branches that WOULD deny, so the
    caller can log and count them (fraud_rule_shadow_total{rule}).

    The velocity cut-offs below are expert-set and NOT validated: Fraud_Data.csv has one
    transaction per user, so every input here except did_uid_90d / ip_uid_7d is always 0
    in the replay, and none of these branches fires on the validation or test period.
    They need data with repeat users (or production labels) to tune. The old
    'purchase_cnt_90d >= 20 and purchase_amt_90d >= $1M' branch was removed (dead on both
    datasets)."""
    would_deny = []
    # did_uid_days_on_file is the age of the (device, user) PAIR, not of the device.
    # Pair seen <= 90 d and 10+ distinct users on this device in 90 d.
    if feature.did_uid_days_on_file <= 90 and feature.did_uid_90d >= 10:
        would_deny.append('device_many_users_90d')
    if feature.uid_did_90d >= 20:
        would_deny.append('user_many_devices_90d')
    if feature.ip_uid_7d >= 50:       # ip_uid only tracked to 7d window
        would_deny.append('ip_many_users_7d')
    # Combined: elevated model score + high 24h purchase velocity.
    # ELEVATED_SCORE = top 3% of validation scores for the served model (model_meta.json).
    if score >= ELEVATED_SCORE and feature.purchase_cnt_24h >= 5:
        would_deny.append('elevated_score_purchases_24h')
    # Combined: elevated model score + multiple devices today (account takeover signal)
    if score >= ELEVATED_SCORE and feature.uid_did_24h >= 3:
        would_deny.append('elevated_score_devices_24h')
    return would_deny
