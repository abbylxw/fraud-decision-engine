from . import rules
from app.data.data_models import Transaction, Features


def execute_hard_rules(t: Transaction, f: Features) -> str:
    """Fast rules that don't need the model score. Short-circuit on clear denials."""
    if (rules.transaction_limit(t) == 'deny'
            or rules.age_limit(t, f) == 'deny'
            or rules.ip_country_block(t, f) == 'deny'):
        return 'deny'
    return 'allow'


def execute_soft_rules(t: Transaction, f: Features, score: float) -> str:
    """Velocity rules and model threshold — all receive the model score."""
    if (rules.rate_limiter_control(t, f, score) == 'deny'
            or rules.model_score_threshold(score) == 'deny'):
        return 'deny'
    return 'allow'
