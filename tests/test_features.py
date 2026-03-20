"""
Unit tests for the pure (no DB, no Redis) feature functions in features/request.py.

"Pure" means the function only reads from the Transaction object and does math —
no external calls. These tests are the simplest and fastest in the suite.
"""
from datetime import datetime, timezone, timedelta
import pytest
from app.data.data_models import Transaction
from app.decisions.features.request import (
    get_days_since_signup,
    get_days_since_signup_group,
    get_hours_to_purchase,
    get_hour_of_day,
    get_day_of_week,
    get_month_of_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transaction(**overrides) -> Transaction:
    """Build a minimal Transaction, overriding only specified fields."""
    defaults = dict(
        user_id="u1",
        signup_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        purchase_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        purchase_value=50.0,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


# ---------------------------------------------------------------------------
# get_days_since_signup
# ---------------------------------------------------------------------------

class TestDaysSinceSignup:
    def test_exact_days(self):
        t = make_transaction(
            signup_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            purchase_time=datetime(2024, 1, 11, tzinfo=timezone.utc),
        )
        assert get_days_since_signup(t) == 10

    def test_same_day_returns_zero(self):
        t = make_transaction(
            signup_time=datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc),
            purchase_time=datetime(2024, 6, 1, 23, 59, tzinfo=timezone.utc),
        )
        assert get_days_since_signup(t) == 0

    def test_missing_signup_returns_sentinel(self):
        # -999 is the sentinel value meaning "we don't know when they signed up"
        t = make_transaction(signup_time=None)
        assert get_days_since_signup(t) == -999


# ---------------------------------------------------------------------------
# get_days_since_signup_group
# ---------------------------------------------------------------------------

class TestDaysSinceSignupGroup:
    @pytest.mark.parametrize("days,expected_label", [
        (0,   "0-7d"),
        (7,   "0-7d"),
        (8,   "8-14d"),
        (14,  "8-14d"),
        (15,  "15-30d"),
        (30,  "15-30d"),
        (31,  "31-90d"),
        (90,  "31-90d"),
        (91,  "91-180d"),
        (180, "91-180d"),
        (181, "181-360d"),
        (360, "181-360d"),
        (361, "361-720d"),
        (720, "361-720d"),
        (721, "720+d"),
        (999, "720+d"),
    ])
    def test_bin_labels(self, days, expected_label):
        assert get_days_since_signup_group(days) == expected_label

    def test_sentinel_treated_as_new_account(self):
        # -999 (missing signup) is clipped to 0, falling into the '0-7d' bin
        assert get_days_since_signup_group(-999) == "0-7d"


# ---------------------------------------------------------------------------
# get_hours_to_purchase
# ---------------------------------------------------------------------------

class TestHoursToPurchase:
    def test_one_hour(self):
        t = make_transaction(
            signup_time=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            purchase_time=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        )
        assert get_hours_to_purchase(t) == pytest.approx(1.0)

    def test_thirty_minutes(self):
        t = make_transaction(
            signup_time=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            purchase_time=datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc),
        )
        assert get_hours_to_purchase(t) == pytest.approx(0.5)

    def test_missing_signup_returns_negative_one(self):
        t = make_transaction(signup_time=None)
        assert get_hours_to_purchase(t) == -1.0


# ---------------------------------------------------------------------------
# Temporal features — hour, weekday, month
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_hour_of_day(self):
        t = make_transaction(purchase_time=datetime(2025, 3, 14, 15, 30, tzinfo=timezone.utc))
        assert get_hour_of_day(t) == 15

    def test_day_of_week_monday(self):
        # 2025-03-10 is a Monday → weekday() == 0
        t = make_transaction(purchase_time=datetime(2025, 3, 10, 12, 0, tzinfo=timezone.utc))
        assert get_day_of_week(t) == 0

    def test_day_of_week_sunday(self):
        # 2025-03-16 is a Sunday → weekday() == 6
        t = make_transaction(purchase_time=datetime(2025, 3, 16, 12, 0, tzinfo=timezone.utc))
        assert get_day_of_week(t) == 6

    def test_month_of_year(self):
        t = make_transaction(purchase_time=datetime(2025, 7, 4, 12, 0, tzinfo=timezone.utc))
        assert get_month_of_year(t) == 7
