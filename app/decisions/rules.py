from app.data.data_models import Transaction, Features
from app.decisions.models import THRESHOLD


def transaction_limit(transaction: Transaction):
    if transaction.purchase_value >= 5000:
        return 'deny'
    return 'allow'

def age_limit(transaction: Transaction, feature: Features):
    age = transaction.age
    # Invalid or missing age
    if age is None or age < 0 or age >= 110:
        return 'deny'
    # Elderly customers with very new accounts are high-risk
    if age >= 80 and feature.days_since_signup <= 90:
        return 'deny'
    return 'allow'

# OFAC-sanctioned countries — transactions originating here are denied
_OFAC_BLOCKED = frozenset({
    'Cuba', 'Iran', 'North Korea', 'Russia', 'Syria',
})

def ip_country_block(transaction: Transaction, feature: Features):
    country = feature.ip_country
    if country in _OFAC_BLOCKED:
        return 'deny'
    if country == 'Unknown' and feature.did_uid_days_on_file <= 90:
        return 'deny'   # unresolvable IP + new device — deny
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


def rate_limiter_control(transaction: Transaction, feature: Features, score: float):
    if feature.did_uid_days_on_file <= 90 and feature.did_uid_90d >= 10:
        return 'deny'
    if feature.uid_did_90d >= 20:
        return 'deny'
    if feature.ip_uid_7d >= 50:       # ip_uid only tracked to 7d window
        return 'deny'
    if feature.purchase_cnt_90d >= 20 and feature.purchase_amt_90d >= 1000000:
        return 'deny'
    # Combined: moderate model score + elevated 24h purchase velocity
    if score >= 0.5 and feature.purchase_cnt_24h >= 5:
        return 'deny'
    # Combined: moderate model score + multiple devices today (account takeover signal)
    if score >= 0.5 and feature.uid_did_24h >= 3:
        return 'deny'
    return 'allow'
