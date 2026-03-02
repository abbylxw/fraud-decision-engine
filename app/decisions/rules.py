from app.data.data_models import Transaction


def transaction_limit(transaction: Transaction):
    if transaction.purchase_value >= 5000:
        return 'deny'
