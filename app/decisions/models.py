import bisect
import json
import pathlib
import re

import joblib
import pandas as pd

from app.data.data_models import Features

_base = pathlib.Path(__file__).parent

_model     = joblib.load(_base / "model.joblib")
# Trained with n_jobs=-1 (all cores). Serving scores one row per call from 8 uvicorn workers
# at once, so all-cores means ~80 threads fighting over 10 CPUs: 6.7 ms vs 1.2 ms per
# prediction, ~4x lower throughput (docs/performance_2026-11.md). Scores are unchanged.
if hasattr(_model, "set_params") and "n_jobs" in _model.get_params():
    _model.set_params(n_jobs=1)
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
    """Return the pd.cut bin index for an age value, or -1 if unclassifiable.
    NaN (a missing age in the training replay) is -1 too, same as a live None;
    bisect would otherwise put it in bin 0."""
    if age is None or age != age:
        return -1
    return bisect.bisect_left(_AGE_EDGES, age)


# Grouped country, one-hot: the 10 highest-volume countries keep their own group,
# "Unknown" (IP not in the table) is its own group, everything else is its continent,
# and a name missing from country_groups.json is "Other". One 0/1 column per group, so
# the model sees no fake order between countries (unlike ip_country_code, which numbers
# them alphabetically and is kept only so older models can still be scored).
_country_groups = json.loads((_base / "country_groups.json").read_text())
_TOP_COUNTRIES = frozenset(_country_groups["top_countries"])
_CONTINENT: dict[str, str] = _country_groups["continent"]
COUNTRY_GROUPS = [*_country_groups["top_countries"], "Unknown",
                  *sorted(set(_CONTINENT.values())), "Other"]


def country_group(ip_country: str | None) -> str:
    if ip_country in _TOP_COUNTRIES or ip_country == "Unknown":
        return ip_country
    return _CONTINENT.get(ip_country, "Other")


def _onehot_column(group: str) -> str:
    prefix = "ipc_" if group in _TOP_COUNTRIES or group == "Unknown" else "ipc_grp_"
    return prefix + re.sub(r"\W+", "_", group).strip("_")


# group -> column name, e.g. "United States" -> "ipc_United_States", "Asia" -> "ipc_grp_Asia"
COUNTRY_ONEHOT = {g: _onehot_column(g) for g in COUNTRY_GROUPS}


_SIGNUP_LABELS = ['0-7d', '8-14d', '15-30d', '31-90d', '91-180d', '181-360d', '361-720d', '720+d']
_ENCODED = {"ip_country_code", "days_since_signup_group", "sex", "source", "browser", "age_group",
            *COUNTRY_ONEHOT.values()}

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
    country_column = COUNTRY_ONEHOT[country_group(features.ip_country)]
    for name in _features:
        if name.startswith("ipc_"):
            row[name] = int(name == country_column)
        elif name == "ip_country_code":
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
