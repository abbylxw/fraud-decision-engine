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

def ip_country_block(transaction: Transaction, feature: Features):
    if feature.ip_country != 'US':
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
    return 'allow'
