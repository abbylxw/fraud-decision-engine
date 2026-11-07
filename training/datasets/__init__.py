"""
External fraud datasets with repeat users, used for training/evaluation only (never served).

Fraud_Data.csv has exactly one transaction per user, so per-user velocity features are
always 0 on it. Each loader here maps another dataset onto the column names that
training/build_features.velocity_features() reads, so the SAME velocity replay runs on
every dataset:

    purchase_time, user_id, device_id, ip_address, ip_country, purchase_value, class

On datasets without devices or IPs, those columns hold the closest "shared entity" the
dataset has (e.g. merchant, zip). Each loader documents its mapping.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "data" / "external"

CANONICAL_COLUMNS = [
    "purchase_time", "user_id", "device_id", "ip_address", "ip_country", "purchase_value", "class",
]
