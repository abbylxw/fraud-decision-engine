from . import rules
from app.data.data_models import Transaction, Features


def execute_hard_rules(t: Transaction, f: Features) -> tuple[str, str | None]:
    """Fast rules that don't need the model score. Returns (decision, rule_name)."""
    for rule_fn, args in [
        (rules.transaction_limit, (t,)),
        (rules.signup_burst,      (t,)),
        (rules.age_limit,         (t, f)),
        (rules.ip_country_block,  (t, f)),
    ]:
        if rule_fn(*args) == 'deny':
            return 'deny', rule_fn.__name__
    return 'allow', None


def execute_soft_rules(t: Transaction, f: Features, score: float) -> tuple[str, str | None]:
    """Velocity rules and model threshold — all receive the model score. Returns (decision, rule_name)."""
    for rule_fn, args in [
        (rules.rate_limiter_control,  (t, f, score)),
        (rules.model_score_threshold, (score,)),
    ]:
        if rule_fn(*args) == 'deny':
            return 'deny', rule_fn.__name__
    return 'allow', None
