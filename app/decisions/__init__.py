from . import rules
from app.data.data_models import Transaction, Features

def execute_rules(t: Transaction, f: Features) -> str:
    if (rules.transaction_limit(t) == 'deny'
            or rules.age_limit(t, f) == 'deny'
            or rules.ip_country_block(t, f) == 'deny'
            or rules.rate_limiter_control(t, f) == 'deny'):
        return 'deny'
    else:
        return 'allow'
