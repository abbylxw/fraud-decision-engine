"""
Unit tests for each rule function in app/decisions/rules.py.

These tests call the rule functions directly — no database, no Redis, no HTTP.
That makes them fast and deterministic: the only thing under test is the rule logic itself.

Each test mutates specific fields of the shared fixtures to put the transaction
into a known state, then asserts that the rule returns the expected verdict.
"""
import pytest
from app.decisions.rules import (
    transaction_limit,
    age_limit,
    ip_country_block,
    model_score_threshold,
    rate_limiter_control,
)


# ---------------------------------------------------------------------------
# transaction_limit — deny if purchase_value >= 5000
# ---------------------------------------------------------------------------

class TestTransactionLimit:
    def test_deny_at_threshold(self, sample_transaction):
        sample_transaction.purchase_value = 5000.0
        assert transaction_limit(sample_transaction) == "deny"

    def test_deny_above_threshold(self, sample_transaction):
        sample_transaction.purchase_value = 9999.99
        assert transaction_limit(sample_transaction) == "deny"

    def test_allow_just_below_threshold(self, sample_transaction):
        sample_transaction.purchase_value = 4999.99
        assert transaction_limit(sample_transaction) == "allow"

    def test_allow_zero(self, sample_transaction):
        sample_transaction.purchase_value = 0.0
        assert transaction_limit(sample_transaction) == "allow"


# ---------------------------------------------------------------------------
# age_limit — deny on invalid age or elderly + new account
# ---------------------------------------------------------------------------

class TestAgeLimit:
    def test_deny_negative_age(self, sample_transaction, sample_features):
        sample_transaction.age = -1
        assert age_limit(sample_transaction, sample_features) == "deny"

    def test_deny_age_exactly_110(self, sample_transaction, sample_features):
        sample_transaction.age = 110
        assert age_limit(sample_transaction, sample_features) == "deny"

    def test_deny_age_over_110(self, sample_transaction, sample_features):
        sample_transaction.age = 120
        assert age_limit(sample_transaction, sample_features) == "deny"

    def test_deny_none_age(self, sample_transaction, sample_features):
        sample_transaction.age = None
        assert age_limit(sample_transaction, sample_features) == "deny"

    def test_deny_elderly_with_new_account(self, sample_transaction, sample_features):
        # age >= 80 AND account < 90 days old → high-risk combination
        sample_transaction.age = 80
        sample_features.days_since_signup = 30
        assert age_limit(sample_transaction, sample_features) == "deny"

    def test_allow_elderly_with_established_account(self, sample_transaction, sample_features):
        # age >= 80 but account is older than 90 days → OK
        sample_transaction.age = 80
        sample_features.days_since_signup = 91
        assert age_limit(sample_transaction, sample_features) == "allow"

    def test_allow_age_109(self, sample_transaction, sample_features):
        # 109 is valid (< 110), and with a long-established account it passes
        sample_transaction.age = 109
        sample_features.days_since_signup = 400
        assert age_limit(sample_transaction, sample_features) == "allow"

    def test_allow_typical(self, sample_transaction, sample_features):
        assert age_limit(sample_transaction, sample_features) == "allow"


# ---------------------------------------------------------------------------
# ip_country_block — deny OFAC-sanctioned countries; Unknown + new device
# ---------------------------------------------------------------------------

class TestIpCountryBlock:
    def test_deny_russia(self, sample_transaction, sample_features):
        sample_features.ip_country = "Russia"
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_deny_iran(self, sample_transaction, sample_features):
        sample_features.ip_country = "Iran"
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_deny_north_korea(self, sample_transaction, sample_features):
        sample_features.ip_country = "North Korea"
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_deny_cuba(self, sample_transaction, sample_features):
        sample_features.ip_country = "Cuba"
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_deny_syria(self, sample_transaction, sample_features):
        sample_features.ip_country = "Syria"
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_deny_unknown_with_new_device(self, sample_transaction, sample_features):
        # Unknown country + new device (≤ 90d) — deny
        sample_features.ip_country = "Unknown"
        sample_features.did_uid_days_on_file = 30
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_allow_unknown_with_established_device(self, sample_transaction, sample_features):
        # Unknown country but device older than 90d — allow
        sample_features.ip_country = "Unknown"
        sample_features.did_uid_days_on_file = 91
        assert ip_country_block(sample_transaction, sample_features) == "allow"

    def test_allow_us(self, sample_transaction, sample_features):
        sample_features.ip_country = "United States"
        assert ip_country_block(sample_transaction, sample_features) == "allow"

    def test_allow_non_ofac_country(self, sample_transaction, sample_features):
        # France, Japan, Sweden etc. are not OFAC-blocked — allow
        sample_features.ip_country = "France"
        assert ip_country_block(sample_transaction, sample_features) == "allow"


# ---------------------------------------------------------------------------
# model_score_threshold — deny if fraud probability >= 0.5
# ---------------------------------------------------------------------------

class TestModelScoreThreshold:
    def test_deny_at_threshold(self):
        assert model_score_threshold(0.7) == "deny"

    def test_deny_above_threshold(self):
        assert model_score_threshold(0.99) == "deny"

    def test_allow_just_below_threshold(self):
        assert model_score_threshold(0.699) == "allow"

    def test_allow_zero(self):
        assert model_score_threshold(0.0) == "allow"


# ---------------------------------------------------------------------------
# rate_limiter_control — velocity-based anomaly detection
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_deny_device_flood_on_new_device(self, sample_transaction, sample_features):
        # 10+ distinct users on same device within 90d, device is < 90d old
        sample_features.did_uid_days_on_file = 30
        sample_features.did_uid_90d = 10
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "deny"

    def test_allow_device_flood_on_old_device(self, sample_transaction, sample_features):
        # Same count, but device has been around > 90d — not suspicious
        sample_features.did_uid_days_on_file = 91
        sample_features.did_uid_90d = 10
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "allow"

    def test_deny_user_device_count(self, sample_transaction, sample_features):
        # User has used 20+ distinct devices in 90d
        sample_features.uid_did_90d = 20
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "deny"

    def test_deny_ip_velocity(self, sample_transaction, sample_features):
        # 50+ distinct users from same IP in 7d
        sample_features.ip_uid_7d = 50
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "deny"

    def test_deny_high_purchase_volume(self, sample_transaction, sample_features):
        # 20+ transactions totalling $1M+ in 90d
        sample_features.purchase_cnt_90d = 20
        sample_features.purchase_amt_90d = 1_000_000.0
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "deny"

    def test_allow_normal(self, sample_transaction, sample_features):
        # All counters at defaults (0) — no velocity signals
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == "allow"
