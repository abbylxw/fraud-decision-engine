"""
Tests that velocity functions fall back to PostgreSQL when Redis has no data.

Background: each get_*_counts function first checks redis.exists(key). If the
key is absent (0), the function skips Redis entirely and runs a SQL query instead.
This happens on cold start or after TTL expiry.

These tests verify that fallback path works correctly by:
  - mocking redis.exists() to return 0 (cache miss)
  - mocking db.execute() to return known values
  - asserting the result matches those DB values (not Redis)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.decisions.features.velocity import (
    get_did_uid_counts,
    get_uid_did_counts,
    get_did_txn_counts,
    get_uid_last_txn_time,
    get_ip_uid_counts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_redis(cache_hit: bool = False):
    """Redis mock that either has the key (cache_hit=True) or doesn't."""
    redis = AsyncMock()
    redis.exists = AsyncMock(return_value=1 if cache_hit else 0)
    return redis


def _mock_db(row):
    """
    DB mock whose execute() returns a single row.
    row should be a tuple matching the SQL SELECT column order.
    """
    result = MagicMock()
    result.one.return_value = row
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


def _mock_db_multi(*rows):
    """DB mock that returns different rows on successive execute() calls."""
    results = []
    for row in rows:
        r = MagicMock()
        r.one.return_value = row
        r.all.return_value = [(v,) for v in row]  # used by _purchase_counts median query
        results.append(r)
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=results)
    return db


# ---------------------------------------------------------------------------
# get_did_uid_counts — counts distinct users per device, per time window
# ---------------------------------------------------------------------------

class TestDidUidCountsFallback:
    async def test_cache_miss_reads_from_db(self):
        # 8 windows: 5min, 10min, 1h, 2h, 24h, 7d, 30d, 90d
        db_row = (0, 0, 1, 1, 3, 5, 5, 8)
        redis = _mock_redis(cache_hit=False)
        db = _mock_db(db_row)

        result = await get_did_uid_counts("device_abc", redis, db)

        # Redis was checked, but no data came from it
        redis.exists.assert_called_once_with("did:uid:device_abc")
        db.execute.assert_called_once()

        assert result["5min"] == 0
        assert result["1h"] == 1
        assert result["24h"] == 3
        assert result["90d"] == 8

    async def test_cache_hit_skips_db(self):
        redis = _mock_redis(cache_hit=True)
        # Simulate Redis returning members for each window query
        redis.zrangebyscore = AsyncMock(side_effect=[
            ["u1"],               # 5min
            ["u1"],               # 10min
            ["u1", "u2"],         # 1h
            ["u1", "u2"],         # 2h
            ["u1", "u2", "u3"],   # 24h
            ["u1", "u2", "u3"],   # 7d
            ["u1", "u2", "u3"],   # 30d
            ["u1", "u2", "u3"],   # 90d
        ])
        db = AsyncMock()

        result = await get_did_uid_counts("device_abc", redis, db)

        db.execute.assert_not_called()
        assert result["5min"] == 1
        assert result["1h"] == 2
        assert result["24h"] == 3

    async def test_all_zeros_on_new_device(self):
        # Brand-new device — no history in DB either
        db_row = (0, 0, 0, 0, 0, 0, 0, 0)
        redis = _mock_redis(cache_hit=False)
        db = _mock_db(db_row)

        result = await get_did_uid_counts("new_device", redis, db)

        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# get_uid_did_counts — counts distinct devices per user, per time window
# ---------------------------------------------------------------------------

class TestUidDidCountsFallback:
    async def test_cache_miss_reads_from_db(self):
        db_row = (0, 0, 0, 1, 2, 2, 3, 3)
        redis = _mock_redis(cache_hit=False)
        db = _mock_db(db_row)

        result = await get_uid_did_counts("user_xyz", redis, db)

        redis.exists.assert_called_once_with("uid:did:user_xyz")
        db.execute.assert_called_once()
        assert result["24h"] == 2
        assert result["90d"] == 3

    async def test_cache_hit_skips_db(self):
        redis = _mock_redis(cache_hit=True)
        redis.zrangebyscore = AsyncMock(side_effect=[
            [], [], ["dev1"], ["dev1"], ["dev1", "dev2"],
            ["dev1", "dev2"], ["dev1", "dev2"], ["dev1", "dev2"],
        ])
        db = AsyncMock()

        result = await get_uid_did_counts("user_xyz", redis, db)

        db.execute.assert_not_called()
        assert result["1h"] == 1
        assert result["24h"] == 2


# ---------------------------------------------------------------------------
# get_did_txn_counts — total transaction count per device, per time window
# ---------------------------------------------------------------------------

class TestDidTxnCountsFallback:
    async def test_cache_miss_reads_from_db(self):
        # 7 windows: 5min, 10min, 1h, 2h, 24h, 7d, 30d
        db_row = (0, 0, 2, 4, 10, 50, 100)
        redis = _mock_redis(cache_hit=False)
        db = _mock_db(db_row)

        result = await get_did_txn_counts("device_abc", redis, db)

        redis.exists.assert_called_once_with("did:txn:device_abc")
        db.execute.assert_called_once()
        assert result["1h"] == 2
        assert result["24h"] == 10
        assert result["30d"] == 100

    async def test_cache_hit_skips_db(self):
        redis = _mock_redis(cache_hit=True)
        redis.zrangebyscore = AsyncMock(side_effect=[
            [],                           # 5min
            [],                           # 10min
            ["t1", "t2"],                 # 1h
            ["t1", "t2", "t3"],           # 2h
            ["t1", "t2", "t3"],           # 24h
            ["t1", "t2", "t3", "t4"],     # 7d
            ["t1", "t2", "t3", "t4"],     # 30d
        ])
        db = AsyncMock()

        result = await get_did_txn_counts("device_abc", redis, db)

        db.execute.assert_not_called()
        assert result["5min"] == 0
        assert result["1h"] == 2
        assert result["7d"] == 4


# ---------------------------------------------------------------------------
# get_uid_last_txn_time — seconds since user's last transaction
# ---------------------------------------------------------------------------

class TestUidLastTxnTimeFallback:
    async def test_cache_miss_reads_from_db(self):
        # DB returns 3600 seconds (1 hour ago)
        redis = _mock_redis()
        redis.get = AsyncMock(return_value=None)  # cache miss
        db = _mock_db((3600.0,))

        result = await get_uid_last_txn_time("user_xyz", redis, db)

        redis.get.assert_called_once_with("uid:last_txn:user_xyz")
        db.execute.assert_called_once()
        assert result == pytest.approx(3600.0)

    async def test_cache_miss_db_returns_none_gives_zero(self):
        # User has never transacted — DB returns NULL → 0.0
        redis = _mock_redis()
        redis.get = AsyncMock(return_value=None)
        db = _mock_db((None,))

        result = await get_uid_last_txn_time("new_user", redis, db)

        assert result == 0.0

    async def test_cache_hit_skips_db(self):
        import time
        redis = _mock_redis()
        # Redis stores a timestamp (seconds since epoch)
        stored_time = str(time.time() - 120)  # 2 minutes ago
        redis.get = AsyncMock(return_value=stored_time)
        db = AsyncMock()

        result = await get_uid_last_txn_time("user_xyz", redis, db)

        db.execute.assert_not_called()
        # Should be approximately 120 seconds (allow ±5s for test execution time)
        assert 115 <= result <= 125


# ---------------------------------------------------------------------------
# get_ip_uid_counts — counts distinct users per IP, per time window
# ---------------------------------------------------------------------------

class TestIpUidCountsFallback:
    async def test_cache_miss_reads_from_db(self):
        # 6 windows: 5min, 10min, 1h, 2h, 24h, 7d
        db_row = (0, 0, 1, 1, 5, 20)
        redis = _mock_redis(cache_hit=False)
        db = _mock_db(db_row)

        result = await get_ip_uid_counts(3663237372, redis, db)

        redis.exists.assert_called_once_with("ip:uid:3663237372")
        db.execute.assert_called_once()
        assert result["1h"] == 1
        assert result["24h"] == 5
        assert result["7d"] == 20
