"""
Training-serving encoding contract.

The model only ever sees integers. Training turns strings like "Chrome" or "M" into
integers one way (pandas, in the notebook / training/build_features.py); the live API
does it another way (plain dicts in app/decisions/models.py). If the two disagree,
the model silently gets wrong inputs — no error, just worse decisions.

That actually happened: training used sex M=0/F=1 while serving used F=0/M=1.
These tests pin the original training encodings (how model.joblib was built) and
check the serving dicts reproduce them for every value.
"""
import json
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest

from app.data.data_models import Features
from app.decisions import models
from app.decisions.features.request import get_age_account_gap
from app.decisions.models import (
    _BROWSER_CODES,
    _SEX_CODES,
    _SOURCE_CODES,
    COUNTRY_GROUPS,
    COUNTRY_ONEHOT,
    _age_group_code,
    country_group,
)
from training.build_features import build_feature_frame

IP_TABLE_COUNTRIES = json.loads(
    (pathlib.Path(models.__file__).parent / "ip_country_categories.json").read_text())

# Every category value present in Fraud_Data.csv (checked 2026-11-04).
SOURCES = ["SEO", "Ads", "Direct"]
BROWSERS = ["Chrome", "IE", "Safari", "FireFox", "Opera"]
SEXES = ["M", "F"]


def _training_codes(values: list[str]) -> dict[str, int]:
    """How the training notebook encodes source/browser: pd.Categorical(...).codes,
    which sorts categories alphabetically and numbers them 0, 1, 2, ..."""
    cat = pd.Categorical(values)
    return dict(zip(values, cat.codes.tolist()))


def test_source_matches_training():
    assert _SOURCE_CODES == _training_codes(SOURCES)


def test_browser_matches_training():
    assert _BROWSER_CODES == _training_codes(BROWSERS)


def test_sex_matches_training():
    # The notebook uses an explicit map, not pd.Categorical:
    #   df["sex"].map({"M": 0, "F": 1}).fillna(-1)
    training_sex = {"M": 0, "F": 1}
    assert _SEX_CODES == training_sex


@pytest.mark.parametrize("missing", [None, "", "X"])
def test_unknown_sex_is_minus_one(missing):
    # Training fills unmapped sex with -1; serving must do the same.
    assert _SEX_CODES.get(missing or "", -1) == -1


def test_missing_age_features_match_training():
    # The API requires age, but the replay can still see NaN ages. Both paths must give the
    # same age_group and age_account_gap: training sees NaN, serving sees None.
    assert _age_group_code(None) == _age_group_code(float("nan")) == -1
    df = pd.DataFrame({
        "user_id": ["u0", "u1"], "device_id": ["d0", "d1"], "ip_address": [1, 2],
        "purchase_value": 10.0, "class": 0, "ip_country": "United States",
        "purchase_time": pd.date_range("2015-01-01", periods=2, freq="h"),
        "signup_time": pd.Timestamp("2014-01-01"),
        "sex": "M", "source": "SEO", "browser": "Chrome", "age": [float("nan"), 30.0],
    })
    train = build_feature_frame(df)
    assert train["age_group"].tolist() == [_age_group_code(None), _age_group_code(30)]
    days = int(train["days_since_signup"].iloc[0])
    serving = [get_age_account_gap(SimpleNamespace(age=a), days) for a in (None, 30)]
    assert train["age_account_gap"].tolist() == pytest.approx(serving)


# ---------------------------------------------------------------------------
# Grouped country (one-hot): top 10 countries, Unknown, continents, Other
# ---------------------------------------------------------------------------

def test_every_ip_table_country_has_a_real_group():
    # A country silently falling into "Other" would lose its continent signal.
    assert [c for c in IP_TABLE_COUNTRIES if country_group(c) == "Other"] == []


def test_group_rules():
    assert country_group("United States") == "United States"     # top 10
    assert country_group("Korea Republic of") == "Korea Republic of"
    assert country_group("Unknown") == "Unknown"
    assert country_group("Netherlands") == "Europe"               # not top 10
    assert country_group("Russian Federation") == "Europe"
    assert country_group("Atlantis") == "Other"                   # not in the table
    assert len(COUNTRY_GROUPS) == 18 and len(set(COUNTRY_ONEHOT.values())) == 18


def _serving_row(country):
    """The 0/1 values score() would feed the model for each one-hot column."""
    column = COUNTRY_ONEHOT[country_group(country)]
    return {c: int(c == column) for c in COUNTRY_ONEHOT.values()}


def test_training_and_serving_one_hot_match_for_every_country():
    n = len(IP_TABLE_COUNTRIES)
    df = pd.DataFrame({
        "ip_country": IP_TABLE_COUNTRIES,
        "user_id": [f"u{i}" for i in range(n)], "device_id": [f"d{i}" for i in range(n)],
        "ip_address": range(n), "purchase_value": 10.0, "class": 0,
        "purchase_time": pd.date_range("2015-01-01", periods=n, freq="h"),
        "signup_time": pd.Timestamp("2014-01-01"),
        "sex": "M", "source": "SEO", "browser": "Chrome", "age": 30,
    })
    train = build_feature_frame(df)[list(COUNTRY_ONEHOT.values())]
    assert (train.sum(axis=1) == 1).all()                        # exactly one group each
    for i, country in enumerate(IP_TABLE_COUNTRIES):
        assert train.iloc[i].to_dict() == _serving_row(country), country


def test_score_uses_the_one_hot_columns(monkeypatch):
    # score() must set exactly the matching column to 1 when the model asks for them.
    seen = {}

    class Spy:
        def predict_proba(self, X):
            seen.update(X.iloc[0].to_dict())
            return [[0.5, 0.5]]

    monkeypatch.setattr(models, "_model", Spy())
    monkeypatch.setattr(models, "_features", list(COUNTRY_ONEHOT.values()))
    models.score(Features(user_id="u", signup_time="2014-12-01T00:00:00Z",
                          purchase_time="2015-01-01T00:00:00Z",
                          purchase_value=1.0, device_id="d", source="SEO",
                          browser="Chrome", sex="M", age=30, ip_address=1,
                          ip_country="Netherlands"))
    assert seen == _serving_row("Netherlands")
    assert seen["ipc_grp_Europe"] == 1
