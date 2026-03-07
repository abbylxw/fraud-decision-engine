from . import rules, models
from app.data.data_models import Transaction, Features

def execute_rules(t: Transaction, f: Features) -> str:
    if rules.transaction_limit(t) == 'deny' or rules.age_limit(t,f) == 'deny':
        return 'deny'
    else:
        return 'allow'
