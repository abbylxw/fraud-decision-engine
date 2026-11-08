"""
Idempotency and fire-and-forget task handling in app/main.py.

These call the endpoint coroutine directly with a real (in-memory) Redis from fakeredis,
so SET NX / GET / DELETE behave exactly like Redis, and asyncio.gather gives true
concurrency (TestClient serialises requests, so it can't reproduce the race).
"""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis
import pytest
from fastapi import HTTPException

from app import main
from app.data.data_models import DecisionResponse, Transaction

_TXN = Transaction(
    user_id="user_001",
    signup_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    purchase_time=datetime(2025, 3, 14, 12, 0, tzinfo=timezone.utc),
    purchase_value=100.0,
    device_id="device_001",
    source="SEO",
    browser="Chrome",
    sex="M",
    ip_address=1234567890,
)


def _response() -> DecisionResponse:
    return DecisionResponse(transaction_id=uuid4(), fraud_decision="allow",
                            response_time=datetime.now(tz=timezone.utc))


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


def _slow_pipeline(calls: list, delay=0.05, fail=False):
    async def run_process(*_args, **_kwargs):
        calls.append(1)
        await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("db down")
        return _response()
    return run_process


async def _post(redis, key="k1"):
    return await main.process_transaction(_TXN, key, db=MagicMock(), redis=redis, executor=None)


class TestIdempotency:
    async def test_concurrent_duplicates_run_pipeline_once(self, redis):
        calls = []
        with patch.object(main, "run_process", _slow_pipeline(calls)):
            results = await asyncio.gather(_post(redis), _post(redis), return_exceptions=True)

        assert len(calls) == 1
        ok = [r for r in results if isinstance(r, DecisionResponse)]
        conflicts = [r for r in results if isinstance(r, HTTPException)]
        assert len(ok) == 1 and len(conflicts) == 1
        assert conflicts[0].status_code == 409
        assert conflicts[0].headers["Retry-After"] == "1"

    async def test_retry_after_completion_replays_stored_response(self, redis):
        calls = []
        with patch.object(main, "run_process", _slow_pipeline(calls, delay=0)):
            first = await _post(redis)
            second = await _post(redis)

        assert len(calls) == 1
        assert second.transaction_id == first.transaction_id
        assert await redis.ttl("idem:k1") > main.IDEMPOTENCY_LOCK_SECONDS

    async def test_failed_request_releases_key_for_retry(self, redis):
        calls = []
        with patch.object(main, "run_process", _slow_pipeline(calls, delay=0, fail=True)):
            with pytest.raises(RuntimeError):
                await _post(redis)
        assert await redis.get("idem:k1") is None

        with patch.object(main, "run_process", _slow_pipeline(calls, delay=0)):
            resp = await _post(redis)
        assert len(calls) == 2
        assert resp.fraud_decision == "allow"

    async def test_different_keys_are_independent(self, redis):
        calls = []
        with patch.object(main, "run_process", _slow_pipeline(calls)):
            a, b = await asyncio.gather(_post(redis, "a"), _post(redis, "b"))
        assert len(calls) == 2
        assert a.transaction_id != b.transaction_id


class TestBackgroundTasks:
    async def test_failure_is_counted_and_reference_released(self):
        async def boom():
            raise RuntimeError("redis down")

        before = main.BACKGROUND_TASK_FAILURES.labels(task="t")._value.get()
        task = main.spawn_background(boom(), name="t")
        assert task in main._background_tasks  # strong reference while running
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)  # let the done-callback run

        assert task not in main._background_tasks
        assert main.BACKGROUND_TASK_FAILURES.labels(task="t")._value.get() == before + 1

    async def test_velocity_not_updated_when_db_commit_fails(self, sample_features):
        db = MagicMock()
        db.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
        update = AsyncMock()
        with (
            patch.object(main, "build_features", AsyncMock(return_value=sample_features)),
            patch.object(main, "model_score", return_value=0.01),
            patch.object(main, "update_velocity_counters", update),
        ):
            # executor=None -> run_in_executor uses the loop's default executor
            with pytest.raises(RuntimeError, match="commit failed"):
                await main.run_process(_TXN, db, redis=MagicMock(), executor=None)
        update.assert_not_called()
