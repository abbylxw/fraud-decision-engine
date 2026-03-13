import bisect
import json
import pathlib

import joblib
import numpy as np

from app.data.data_models import Features

_base = pathlib.Path(__file__).parent

_model     = joblib.load(_base / "model.joblib")
_features  = json.loads((_base / "model_features.json").read_text())
_countries = json.loads((_base / "ip_country_categories.json").read_text())

# Probability threshold: transactions above this are flagged as fraud.
# 0.5 is the default decision boundary; tune upward to reduce false positives.
THRESHOLD = 0.5


def _country_code(ip_country: str) -> int:
    """Map a country name string to its pd.Categorical integer code.

    During training: ip_country_code = pd.Categorical(df["ip_country"]).codes
    pd.Categorical sorts categories alphabetically and assigns 0, 1, 2, …
    We saved that sorted list to ip_country_categories.json, so a bisect
    lookup reproduces the exact same integer without needing pandas at runtime.
    Returns -1 for countries not seen in training (matches pd.Categorical behaviour).
    """
    idx = bisect.bisect_left(_countries, ip_country)
    if idx < len(_countries) and _countries[idx] == ip_country:
        return idx
    return -1


def score(features: Features) -> float:
    """Return the raw fraud probability (0.0–1.0).

    Builds a 1-row numpy array in the exact column order the model was trained on
    (from model_features.json) and returns predict_proba for the fraud class.
    """
    _SIGNUP_LABELS = ['0-7d', '8-14d', '15-30d', '31-90d', '91-180d', '181-360d', '361-720d', '720+d']

    row = {name: 0 for name in _features}
    for name in _features:
        if name == "ip_country_code":
            row[name] = _country_code(features.ip_country)
        elif name == "days_since_signup_group":
            label = features.days_since_signup_group
            row[name] = _SIGNUP_LABELS.index(label) if label in _SIGNUP_LABELS else 0
        elif hasattr(features, name):
            row[name] = getattr(features, name)

    X = np.array([[row[name] for name in _features]], dtype=np.float32)
    return float(_model.predict_proba(X)[0][1])


def predict(features: Features) -> str:
    """Run the trained RandomForest model and return 'deny' or 'allow'."""
    return 'deny' if score(features) >= THRESHOLD else 'allow'
