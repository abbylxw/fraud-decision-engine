"""
Integration tests for the HTTP endpoints.

"Integration" here means we test through the full FastAPI request/response cycle,
but we replace ("mock") the database and Redis with fake objects so the tests
don't need real services running.

Key concepts used here:
  - app.dependency_overrides: FastAPI's built-in way to swap dependencies in tests.
  - AsyncMock: like MagicMock, but for async functions (ones that use `await`).
  - @patch: temporarily replace a name in a module with a fake during a test.
  - TestClient: a synchronous HTTP client that drives the async FastAPI app internally.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_db, get_redis
from app.data.data_models import Features


# ---------------------------------------------------------------------------
# Shared mock setup
# ---------------------------------------------------------------------------

def _make_mock_db():
    """AsyncMock that behaves like an AsyncSession (execute, add, commit)."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _make_mock_redis(cached=None):
    """AsyncMock that behaves like a Redis client. cached= simulates an idempotency hit."""
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.getex = AsyncMock(return_value=cached)
    redis.set = AsyncMock(return_value=True)
    return redis


def _apply_overrides(mock_db, mock_redis):
    """Register dependency overrides on the app. Call _clear_overrides() after the test."""
    async def _db():
        yield mock_db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_redis] = lambda: mock_redis
    # get_executor is not overridden — the lifespan creates a real ThreadPoolExecutor


def _clear_overrides():
    app.dependency_overrides.clear()


# A minimal valid transaction payload
_TXN_PAYLOAD = {
    "user_id": "user_001",
    "signup_time": "2024-01-01T00:00:00Z",
    "purchase_time": "2025-03-14T12:00:00Z",
    "purchase_value": 100.0,
    "device_id": "device_001",
    "source": "SEO",
    "browser": "Chrome",
    "sex": "M",
    "age": 30,
    "ip_address": 1234567890,
}

# Features that build_features will return in tests that reach the model
_CLEAN_FEATURES = Features(
    user_id="user_001",
    purchase_time=datetime(2025, 3, 14, 12, 0, tzinfo=timezone.utc),
    purchase_value=100.0,
    ip_country="United States",
    days_since_signup=438,
    days_since_signup_group="361-720d",
)


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------

class TestHealth:
    def test_ok_when_both_healthy(self):
        mock_db = _make_mock_db()
        mock_redis = _make_mock_redis()
        _apply_overrides(mock_db, mock_redis)
        try:
            with TestClient(app) as client:
                resp = client.get("/health")
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db"] == "ok"
        assert data["redis"] == "ok"

    def test_degraded_when_db_down(self):
        mock_db = _make_mock_db()
        mock_db.execute.side_effect = Exception("connection refused")
        mock_redis = _make_mock_redis()
        _apply_overrides(mock_db, mock_redis)
        try:
            with TestClient(app) as client:
                resp = client.get("/health")
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["db"] != "ok"
        assert data["redis"] == "ok"

    def test_degraded_when_redis_down(self):
        mock_db = _make_mock_db()
        mock_redis = _make_mock_redis()
        mock_redis.ping.side_effect = Exception("connection refused")
        _apply_overrides(mock_db, mock_redis)
        try:
            with TestClient(app) as client:
                resp = client.get("/health")
        finally:
            _clear_overrides()

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["db"] == "ok"
        assert data["redis"] != "ok"


# ---------------------------------------------------------------------------
# POST /api/transactions tests
# ---------------------------------------------------------------------------

class TestTransactionEndpoint:
    def test_deny_by_hard_rule(self):
        """A purchase >= $5000 is denied by the transaction_limit hard rule.
        Because hard rules short-circuit, build_features is still called but
        model_score is never called."""
        mock_db = _make_mock_db()
        mock_redis = _make_mock_redis()
        _apply_overrides(mock_db, mock_redis)

        payload = {**_TXN_PAYLOAD, "purchase_value": 9999.0}

        with (
            patch("app.main.build_features", new_callable=AsyncMock, return_value=_CLEAN_FEATURES),
            patch("app.main.update_velocity_counters", new_callable=AsyncMock),
        ):
            try:
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/transactions",
                        json=payload,
                        headers={"Idempotency-Key": "deny-rule-test"},
                    )
            finally:
                _clear_overrides()

        assert resp.status_code == 201
        data = resp.json()
        assert data["fraud_decision"] == "deny"
        assert "transaction_id" in data
        assert "response_time" in data

    def test_allow_legitimate_transaction(self):
        """A clean transaction (US, normal amount, low model score) is approved."""
        mock_db = _make_mock_db()
        mock_redis = _make_mock_redis()
        _apply_overrides(mock_db, mock_redis)

        with (
            patch("app.main.build_features", new_callable=AsyncMock, return_value=_CLEAN_FEATURES),
            patch("app.main.model_score", return_value=0.05),   # well below 0.5 threshold
            patch("app.main.update_velocity_counters", new_callable=AsyncMock),
        ):
            try:
                with TestClient(app) as client:
                    resp = client.post(
                        "/api/transactions",
                        json=_TXN_PAYLOAD,
                        headers={"Idempotency-Key": "allow-test"},
                    )
            finally:
                _clear_overrides()

        assert resp.status_code == 201
        assert resp.json()["fraud_decision"] == "allow"

    def test_idempotency_returns_cached_response(self):
        """Sending the same Idempotency-Key twice returns the first response
        without re-processing the transaction."""
        # Simulate a cached response already stored in Redis
        cached_json = (
            '{"transaction_id": "00000000-0000-0000-0000-000000000001",'
            ' "fraud_decision": "allow",'
            ' "response_time": "2025-03-14T12:00:00Z"}'
        )
        mock_db = _make_mock_db()
        mock_redis = _make_mock_redis(cached=cached_json)
        _apply_overrides(mock_db, mock_redis)

        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/transactions",
                    json=_TXN_PAYLOAD,
                    headers={"Idempotency-Key": "duplicate-key"},
                )
        finally:
            _clear_overrides()

        assert resp.status_code == 201
        data = resp.json()
        # The cached transaction_id is returned — not a newly generated one
        assert data["transaction_id"] == "00000000-0000-0000-0000-000000000001"
        assert data["fraud_decision"] == "allow"
