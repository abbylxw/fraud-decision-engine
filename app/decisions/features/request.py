from app.data.data_models import Transaction


def get_days_since_signup(t: Transaction) -> int:
    if t.signup_time is None:
        return -999
    time_diff = t.purchase_time - t.signup_time
    return max(0, time_diff.days)


def get_age_group(t: Transaction) -> str:
    if t.age is None:
        return 'missing'
    if t.age < 0 or t.age >= 110:
        return 'invalid_age'
    lower_age = (t.age // 10.0) * 10
    higher_age = lower_age + 10
    if lower_age <= t.age < higher_age:
        return f'{lower_age}-{higher_age}'
