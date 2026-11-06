"""
Shared fixtures for all tests.

A "fixture" is a piece of reusable test setup. pytest automatically passes
fixtures into test functions that name them as parameters — no manual wiring needed.
"""
from datetime import datetime, timezone

import pytest

from app.data.data_models import Features, Transaction


@pytest.fixture
def sample_transaction():
    """A typical, non-fraudulent transaction. Tests can mutate specific fields."""
    return Transaction(
        user_id="user_001",
        signup_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        purchase_time=datetime(2025, 3, 14, 12, 0, tzinfo=timezone.utc),
        purchase_value=100.0,
        device_id="device_001",
        source="SEO",
        browser="Chrome",
        sex="M",
        age=30,
        ip_address=1234567890,
    )


@pytest.fixture
def sample_features(sample_transaction):
    """Features built from sample_transaction. Defaults to a clean US profile."""
    return Features(
        **sample_transaction.model_dump(by_alias=False),
        ip_country="United States",
        days_since_signup=438,
        days_since_signup_group="361-720d",
    )
