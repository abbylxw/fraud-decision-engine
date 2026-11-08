from app.data.data_models import Features, Transaction

from . import rules


def execute_hard_rules(t: Transaction, f: Features) -> tuple[str, str | None]:
    """Fast rules that don't need the model score. Returns (decision, rule_name)."""
    for rule_fn, args in [
        (rules.signup_burst,      (t,)),
        (rules.age_limit,         (t, f)),
        (rules.ip_country_block,  (t, f)),
    ]:
        if rule_fn(*args) == 'deny':
            return 'deny', rule_fn.__name__
    return 'allow', None


def execute_soft_rules(t: Transaction, f: Features, score: float) -> tuple[str, str | None]:
    """Rules that receive the model score. Returns (decision, rule_name)."""
    for rule_fn, args in [
        (rules.model_score_threshold, (score,)),
    ]:
        if rule_fn(*args) == 'deny':
            return 'deny', rule_fn.__name__
    return 'allow', None


def execute_shadow_rules(t: Transaction, f: Features, score: float) -> list[str]:
    """Rules in shadow mode: never change the decision. Returns the names of the branches
    that would have denied, for logging and the fraud_rule_shadow_total counter."""
    return rules.rate_limiter_control(t, f, score)
