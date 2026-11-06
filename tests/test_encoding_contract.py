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
import pandas as pd
import pytest

from app.decisions.models import _BROWSER_CODES, _SEX_CODES, _SOURCE_CODES

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
