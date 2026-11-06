import bisect
import json
import pathlib

import joblib
import pandas as pd

from app.data.data_models import Features

_base = pathlib.Path(__file__).parent

_model     = joblib.load(_base / "model.joblib")
_features  = json.loads((_base / "model_features.json").read_text())
_countries = json.loads((_base / "ip_country_categories.json").read_text())

# Written by training/train.py alongside model.joblib: version, the cost-based
# threshold chosen on validation, training window, and evaluation results.
_meta_path = _base / "model_meta.json"
_meta = json.loads(_meta_path.read_text()) if _meta_path.exists() else {}

# Deny when the fraud probability is at or above this. Comes from the model's own
# metadata because a threshold only means something for the model it was tuned on.
THRESHOLD: float = _meta.get("threshold", 0.7)
MODEL_VERSION: str = _meta.get("model_version", "legacy-notebook-rf")
# "Elevated" score for the rate-limiter combos (score + velocity): the riskiest 3% of
# validation traffic on this model's scale. 0.5 was the old hand-picked value.
ELEVATED_SCORE: float = _meta.get("elevated_score_threshold", 0.5)


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


# These maps MUST match the encodings used in training (training/build_features.py imports
# them from here). tests/test_encoding_contract.py fails if the two ever drift apart.
_SEX_CODES     = {"M": 0, "F": 1}
_SOURCE_CODES  = {"Ads": 0, "Direct": 1, "SEO": 2}
_BROWSER_CODES = {"Chrome": 0, "FireFox": 1, "IE": 2, "Opera": 3, "Safari": 4}

# Mirrors pd.cut(df["age"], bins=age_bins.json["bins"]).cat.codes
_AGE_BINS = json.loads((_base / "age_bins.json").read_text())["bins"]
_AGE_EDGES = [b if b is not None else float("inf") for b in _AGE_BINS[1:]]  # right edges


def _age_group_code(age: int | None) -> int:
    """Return the pd.cut bin index for an age value, or -1 if unclassifiable."""
    if age is None:
        return -1
    return bisect.bisect_left(_AGE_EDGES, age)


_SIGNUP_LABELS = ['0-7d', '8-14d', '15-30d', '31-90d', '91-180d', '181-360d', '361-720d', '720+d']
_ENCODED = {"ip_country_code", "days_since_signup_group", "sex", "source", "browser", "age_group"}

# Fail at startup, not silently at request time: every feature the model expects must
# be either encoded below or a field on Features. Otherwise score() would feed it 0.
_unserved = [f for f in _features if f not in _ENCODED and f not in Features.model_fields]
if _unserved:
    raise RuntimeError(f"model_features.json lists features the API does not produce: {_unserved}")


def score(features: Features) -> float:
    """Return the raw fraud probability (0.0–1.0).

    Builds a 1-row numpy array in the exact column order the model was trained on
    (from model_features.json) and returns predict_proba for the fraud class.
    """
    row = {name: 0 for name in _features}
    for name in _features:
        if name == "ip_country_code":
            row[name] = _country_code(features.ip_country)
        elif name == "days_since_signup_group":
            label = features.days_since_signup_group
            row[name] = _SIGNUP_LABELS.index(label) if label in _SIGNUP_LABELS else 0
        elif name == "sex":
            row[name] = _SEX_CODES.get(features.sex or "", -1)
        elif name == "source":
            row[name] = _SOURCE_CODES.get(features.source or "", -1)
        elif name == "browser":
            row[name] = _BROWSER_CODES.get(features.browser or "", -1)
        elif name == "age_group":
            row[name] = _age_group_code(features.age)
        else:
            row[name] = getattr(features, name)

    X = pd.DataFrame([row], columns=_features)
    return float(_model.predict_proba(X)[0][1])


def predict(features: Features) -> str:
    """Run the served model and return 'deny' or 'allow'."""
    return 'deny' if score(features) >= THRESHOLD else 'allow'
