"""
Unit tests for each rule function in app/decisions/rules.py.

These tests call the rule functions directly — no database, no Redis, no HTTP.
That makes them fast and deterministic: the only thing under test is the rule logic itself.

Each test mutates specific fields of the shared fixtures to put the transaction
into a known state, then asserts that the rule returns the expected verdict.
"""
import json
import pathlib
from datetime import timedelta

import pytest

from app.data.data_models import Features
from app.decisions import execute_hard_rules, execute_shadow_rules, execute_soft_rules, rules
from app.decisions.models import ELEVATED_SCORE, THRESHOLD
from app.decisions.rules import (
    SIGNUP_BURST_SECONDS,
    age_limit,
    ip_country_block,
    model_score_threshold,
    rate_limiter_control,
    signup_burst,
)

# No amount cap: transaction_limit ($5,000) was removed (rule-logic-review, Q1)
# ---------------------------------------------------------------------------

class TestNoAmountCap:
    def test_rule_removed(self):
        assert not hasattr(rules, "transaction_limit")

    def test_large_purchase_not_hard_denied(self, sample_transaction, sample_features):
        sample_transaction.purchase_value = 9999.99
        sample_features.purchase_value = 9999.99
        assert execute_hard_rules(sample_transaction, sample_features) == ("allow", None)


# ---------------------------------------------------------------------------
# age_limit — data check: deny only an age that can't be real
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

    def test_allow_elderly_with_new_account(self, sample_transaction, sample_features):
        # The age >= 80 + new-account clause was removed.
        sample_transaction.age = 80
        sample_features.days_since_signup = 0
        assert age_limit(sample_transaction, sample_features) == "allow"

    def test_allow_elderly_with_missing_signup(self, sample_transaction, sample_features):
        # Was the -999 bug: a missing signup_time made every 80+ customer a deny.
        sample_transaction.age = 80
        sample_transaction.signup_time = None
        sample_features.days_since_signup = -999
        assert age_limit(sample_transaction, sample_features) == "allow"

    def test_allow_age_109(self, sample_transaction, sample_features):
        # 109 is valid (< 110)
        sample_transaction.age = 109
        assert age_limit(sample_transaction, sample_features) == "allow"

    def test_allow_typical(self, sample_transaction, sample_features):
        assert age_limit(sample_transaction, sample_features) == "allow"


# ---------------------------------------------------------------------------
# ip_country_block — deny OFAC-sanctioned countries only
# ---------------------------------------------------------------------------

class TestIpCountryBlock:
    # The spellings ip_country actually takes (from IpAddress_to_Country.csv).
    @pytest.mark.parametrize("country", [
        "Cuba", "Iran (ISLAMIC Republic Of)", "Syrian Arab Republic",
    ])
    def test_deny_ip_table_spellings(self, sample_transaction, sample_features, country):
        sample_features.ip_country = country
        assert ip_country_block(sample_transaction, sample_features) == "deny"

    def test_every_blocked_name_is_in_ip_table_or_a_known_alias(self):
        # The short names alone matched nothing in real traffic, so only Cuba was blocked.
        ip_table = set(json.loads(
            (pathlib.Path(rules.__file__).parent / "ip_country_categories.json").read_text()))
        aliases_not_in_table = {"Iran", "Syria", "North Korea"}
        assert rules._OFAC_BLOCKED - aliases_not_in_table <= ip_table
        assert aliases_not_in_table.isdisjoint(ip_table)

    @pytest.mark.parametrize("country", ["Russian Federation", "Russia"])
    def test_allow_russia(self, sample_transaction, sample_features, country):
        # Targeted sanctions, not a country-wide embargo: no longer blocked by IP (2026-11).
        sample_features.ip_country = country
        assert ip_country_block(sample_transaction, sample_features) == "allow"

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

    def test_allow_unknown_with_new_device(self, sample_transaction, sample_features):
        # The 'Unknown IP + new device' clause was removed (Q5): no fraud lift on the data.
        sample_features.ip_country = "Unknown"
        sample_features.did_uid_days_on_file = 0
        assert ip_country_block(sample_transaction, sample_features) == "allow"

    def test_ofac_still_denied_through_hard_rules(self, sample_transaction, sample_features):
        sample_features.ip_country = "Cuba"
        assert execute_hard_rules(sample_transaction, sample_features) == ("deny", "ip_country_block")

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
# signup_burst — deny if purchase is < SIGNUP_BURST_SECONDS after signup
# ---------------------------------------------------------------------------

class TestSignupBurst:
    def _at(self, t, seconds):
        t.purchase_time = t.signup_time + timedelta(seconds=seconds)
        return t

    def test_deny_same_second(self, sample_transaction):
        assert signup_burst(self._at(sample_transaction, 0)) == "deny"

    def test_deny_one_second(self, sample_transaction):
        # The exact pattern of the January 2015 bot attack.
        assert signup_burst(self._at(sample_transaction, 1)) == "deny"

    def test_deny_just_below_limit(self, sample_transaction):
        assert signup_burst(self._at(sample_transaction, SIGNUP_BURST_SECONDS - 0.1)) == "deny"

    def test_allow_at_limit(self, sample_transaction):
        assert signup_burst(self._at(sample_transaction, SIGNUP_BURST_SECONDS)) == "allow"

    def test_allow_fastest_legit_in_data(self, sample_transaction):
        # Fastest legitimate signup -> purchase in Fraud_Data.csv is 137s.
        assert signup_burst(self._at(sample_transaction, 137)) == "allow"

    def test_allow_missing_signup_time(self, sample_transaction):
        sample_transaction.signup_time = None
        assert signup_burst(sample_transaction) == "allow"

    def test_allow_purchase_before_signup(self, sample_transaction):
        # Bad data (clock skew), not a bot signal; other rules/model still run.
        assert signup_burst(self._at(sample_transaction, -5)) == "allow"

    def test_wired_into_hard_rules(self, sample_transaction, sample_features):
        self._at(sample_transaction, 1)
        assert execute_hard_rules(sample_transaction, sample_features) == ("deny", "signup_burst")


# ---------------------------------------------------------------------------
# model_score_threshold — deny if fraud probability >= THRESHOLD (from model_meta.json)
# ---------------------------------------------------------------------------

class TestModelScoreThreshold:
    # THRESHOLD comes from app/decisions/model_meta.json (chosen on validation by
    # training/train.py), so test against it rather than a hardcoded number.
    def test_deny_at_threshold(self):
        assert model_score_threshold(THRESHOLD) == "deny"

    def test_deny_above_threshold(self):
        assert model_score_threshold(min(1.0, THRESHOLD + 0.01)) == "deny"

    def test_allow_just_below_threshold(self):
        assert model_score_threshold(THRESHOLD - 0.001) == "allow"

    def test_allow_zero(self):
        assert model_score_threshold(0.0) == "allow"


# ---------------------------------------------------------------------------
# Default-built Features are not hard-denied (tests (f) and (f2) of the plan)
# ---------------------------------------------------------------------------

class TestDefaultFeatures:
    def _defaults(self, t, age):
        t.age = age
        # Every Features field left at its default: ip_country='Unknown',
        # did_uid_days_on_file=0, days_since_signup=0.
        return Features(**t.model_dump(by_alias=False))

    def test_valid_age_and_defaults_not_hard_denied(self, sample_transaction):
        f = self._defaults(sample_transaction, 30)
        assert execute_hard_rules(sample_transaction, f) == ("allow", None)


# ---------------------------------------------------------------------------
# rate_limiter_control — SHADOW MODE: returns the branches that would deny, never denies
# ---------------------------------------------------------------------------

class TestRateLimiterShadow:
    def test_device_flood_on_new_pair(self, sample_transaction, sample_features):
        # 10+ distinct users on same device within 90d, (device, user) pair <= 90d old
        sample_features.did_uid_days_on_file = 30
        sample_features.did_uid_90d = 10
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == ["device_many_users_90d"]

    def test_device_flood_on_old_pair(self, sample_transaction, sample_features):
        sample_features.did_uid_days_on_file = 91
        sample_features.did_uid_90d = 10
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == []

    def test_user_device_count(self, sample_transaction, sample_features):
        sample_features.uid_did_90d = 20
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == ["user_many_devices_90d"]

    def test_ip_velocity(self, sample_transaction, sample_features):
        sample_features.ip_uid_7d = 50
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == ["ip_many_users_7d"]

    def test_high_purchase_volume_branch_removed(self, sample_transaction, sample_features):
        # The old b4 ($1M in 90 d) branch is gone.
        sample_features.purchase_cnt_90d = 20
        sample_features.purchase_amt_90d = 1_000_000.0
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == []

    def test_normal(self, sample_transaction, sample_features):
        assert rate_limiter_control(sample_transaction, sample_features, 0.1) == []

    # Score + velocity combos: "elevated" comes from model_meta.json (top 3% of validation).
    def test_elevated_score_and_purchase_velocity(self, sample_transaction, sample_features):
        sample_features.purchase_cnt_24h = 5
        assert rate_limiter_control(sample_transaction, sample_features, ELEVATED_SCORE) == [
            "elevated_score_purchases_24h"]

    def test_purchase_velocity_below_elevated_score(self, sample_transaction, sample_features):
        sample_features.purchase_cnt_24h = 5
        assert rate_limiter_control(sample_transaction, sample_features, ELEVATED_SCORE - 0.001) == []

    def test_elevated_score_and_multiple_devices(self, sample_transaction, sample_features):
        sample_features.uid_did_24h = 3
        assert rate_limiter_control(sample_transaction, sample_features, ELEVATED_SCORE) == [
            "elevated_score_devices_24h"]

    def test_elevated_score_is_below_deny_threshold(self):
        assert 0 < ELEVATED_SCORE < THRESHOLD

    def _all_branches(self, f):
        f.did_uid_days_on_file = 0
        f.did_uid_90d = 10
        f.uid_did_90d = 20
        f.ip_uid_7d = 50
        f.purchase_cnt_24h = 5
        f.uid_did_24h = 3

    def test_shadow_branches_never_deny(self, sample_transaction, sample_features):
        # Every shadow branch fires, the score is below the model threshold: still allow.
        self._all_branches(sample_features)
        score = ELEVATED_SCORE
        assert len(execute_shadow_rules(sample_transaction, sample_features, score)) == 5
        assert execute_soft_rules(sample_transaction, sample_features, score) == ("allow", None)

    def test_soft_deny_comes_from_model_only(self, sample_transaction, sample_features):
        self._all_branches(sample_features)
        assert execute_soft_rules(sample_transaction, sample_features, THRESHOLD) == (
            "deny", "model_score_threshold")
