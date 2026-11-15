from app.data.data_models import Features, Transaction
from app.decisions.models import ELEVATED_SCORE, REVIEW_THRESHOLD, THRESHOLD


def age_limit(transaction: Transaction, feature: Features):
    """Data check only: deny an age that can't be real. The request requires age (HTTP 422
    if missing); the None check only guards callers that build a Transaction without
    validation. The old 'age >= 80 + new account' clause was removed (rule-logic-review):
    it had no supporting data and denied by age alone."""
    age = transaction.age
    if age is not None and (age < 0 or age >= 110):
        return 'deny'
    return 'allow'

# Countries under comprehensive OFAC sanctions — transactions originating here are denied.
# This is a compliance rule, not a fraud rule: it is not tuned on fraud cost.
# Russia was removed (2026-11): its sanctions are targeted (sectoral / listed parties), not a
# country-wide embargo, and blocking every Russian IP denied 332 test rows at a 7% fraud rate
# (+$3.1k cost; training/rule_backtest.py). Syria stays until checked: its program was
# reported as lifted in 2025, but sources disagree. Confirm the list with the sanctions owner.
# ip_country comes from the IP table (IpAddress_to_Country.csv), which uses its own
# spellings; the short names never match it. tests/test_rules.py checks every name here
# against the table so a spelling mismatch fails loudly.
_OFAC_BLOCKED = frozenset({
    'Cuba', 'Iran (ISLAMIC Republic Of)', 'Syrian Arab Republic',
    # Short aliases, and North Korea, which has no rows in the IP table.
    'Iran', 'Syria', 'North Korea',
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


# Deny the SECOND distinct user on a device within one hour (event time). did_uid_1h includes
# the buyer (set features count the current purchase's member), so B first -> 1, allow;
# A after B -> 2, deny on A's first attempt; A again after only A -> 1, allow.
# Scripted, not human: the January 2015 bot used 760 devices x 10 user_ids within ~10 s, and a
# bot that waits past SIGNUP_BURST_SECONDS still trips this. Backtest
# (results/agents/retry-after-decline/fd-device-users-k1/result.json): catches 6,840 of 7,600
# bot rows (9 of 10 per device); legit rows hit 1 / 1 / 0 in train / validation / test. A 10 min
# window would catch the same with 0 / 0 / 0 legit hits; 1 h was chosen. No bots in validation /
# test, so this is insurance, not a measured held-out gain.
DEVICE_USERS_1H_LIMIT = 2


def device_many_users_1h(transaction: Transaction, feature: Features) -> str:
    """Deny when distinct users on this device in the last hour (buyer included) reach
    DEVICE_USERS_1H_LIMIT."""
    if feature.did_uid_1h >= DEVICE_USERS_1H_LIMIT:
        return 'deny'
    return 'allow'


def model_score_threshold(score: float) -> str:
    """Three-way decision on the model score:

        score >= THRESHOLD          deny    a decline pays for itself at this risk
        score >= REVIEW_THRESHOLD   review  the caller should step up (3-D Secure, SMS
                                            code) or queue the order for a person
        otherwise                   allow

    Both thresholds come from model_meta.json. Why a middle band exists: at $10 per wrongly
    declined order and a $37 average fraud, a decline only pays above ~27% precision, and
    most of the fraud the model finds sits just below that (docs/feature_ablation.md)."""
    if score >= THRESHOLD:
        return 'deny'
    if score >= REVIEW_THRESHOLD:
        return 'review'
    return 'allow'


def rate_limiter_control(transaction: Transaction, feature: Features, score: float) -> list[str]:
    """SHADOW MODE: never denies. Returns the names of the branches that WOULD deny, so the
    caller can log and count them (fraud_rule_shadow_total{rule}).

    The velocity cut-offs below are expert-set and NOT validated: Fraud_Data.csv has one
    transaction per user, so in the replay the user-keyed set features are always 1 (the
    buyer; set features include the current purchase) and the per-user counts are always 0,
    and none of these branches fires on the validation or test period.
    They need data with repeat users (or production labels) to tune. The old
    'purchase_cnt_90d >= 20 and purchase_amt_90d >= $1M' branch was removed (dead on both
    datasets)."""
    would_deny = []
    # did_uid_days_on_file is the age of the (device, user) PAIR, not of the device.
    # Pair seen <= 90 d and 10+ distinct users on this device in 90 d (buyer included).
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
