"""
build_features() against Redis: the cache-hit path, and the Postgres fallback.

The real code reads every velocity counter in one Redis pipeline. The only DB fallback
left is purchase history: when Redis has no purchase entries for a user (cold start,
eviction), build_features() asks Postgres instead - and must ask "as of the
transaction's time", not the server's NOW().

Redis is an in-memory fake (fakeredis); Postgres is a mock that records its queries.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import aioredis as fake_aioredis

from app.data.data_models import Transaction
from app.decisions.features import build_features, event_ts, update_velocity_counters

PURCHASE_TIME = datetime(2015, 6, 1, 12, 0, 0)   # naive, like the client sends


def _txn(**overrides) -> Transaction:
    base = dict(
        user_id="user_1", signup_time=datetime(2015, 1, 1), purchase_time=PURCHASE_TIME,
        purchase_value=40.0, device_id="device_1", source="SEO", browser="Chrome",
        sex="M", age=30, ip_address=1234567890,
    )
    return Transaction(**(base | overrides))


def _mock_db(rows=()):
    """Postgres mock for the purchase-history fallback: its one execute() returns the
    user's 90-day rows as (transaction_id, purchase_value, purchase_time, fraud_decision)."""
    result = MagicMock()
    result.all.return_value = list(rows)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


# One user's history before PURCHASE_TIME: (minutes earlier, amount, decision).
HISTORY = [
    (3, 10.0, "allow"),             # inside 5min
    (120, 20.0, "deny"),            # inside 24h, but not a successful purchase
    (2 * 1440, 30.0, "allow"),      # inside 7d
    (20 * 1440, 40.0, "allow"),     # inside 30d
    (60 * 1440, 50.0, "allow"),     # inside 90d only
]


def _history_rows():
    now = PURCHASE_TIME.replace(tzinfo=timezone.utc)
    return [(f"h{i}", amount, now - timedelta(minutes=mins), decision)
            for i, (mins, amount, decision) in enumerate(HISTORY)]


@pytest.fixture
async def redis():
    r = fake_aioredis.FakeRedis(decode_responses=True)
    # Pre-fill the two lookup caches so the tests exercise only the velocity path.
    await r.set("cache:ip:1234567890", "United States")
    for user in ("user_0", "user_1", "user_9"):
        await r.set(f"cache:did:device_1:{user}", "0")
    yield r
    await r.aclose()


async def _record(redis, t: Transaction, txn_id: str, decision="allow"):
    await update_velocity_counters(
        user_id=t.user_id, device_id=t.device_id, ip=t.ip_address,
        ip_country="United States", transaction_id=txn_id,
        purchase_value=t.purchase_value, fraud_decision=decision,
        event_ts=event_ts(t), redis=redis,
    )


class TestPurchaseHistoryFallback:
    async def test_empty_redis_asks_db_as_of_transaction_time(self, redis):
        db = _mock_db()
        await build_features(_txn(), db, redis)

        assert db.execute.await_count == 1   # one query, not one per aggregate
        params = db.execute.await_args_list[0].args[1]
        # The window is anchored on purchase_time (read as UTC), not the server clock.
        assert params["as_of"] == PURCHASE_TIME.replace(tzinfo=timezone.utc)
        assert params["user_id"] == "user_1"

    async def test_as_of_is_cast_to_timestamptz(self, redis):
        # Without the cast, Postgres reads `:as_of - INTERVAL` as interval - interval and
        # every request 500s on a real database (the mocked DB here can't catch that).
        db = _mock_db()
        await build_features(_txn(), db, redis)

        for call in db.execute.await_args_list:
            sql = str(call.args[0])
            assert ":as_of - INTERVAL" not in sql
            assert "CAST(:as_of AS timestamptz) - INTERVAL" in sql

    async def test_db_rows_flow_into_features(self, redis):
        f = await build_features(_txn(), _mock_db(_history_rows()), redis)

        assert f.purchase_cnt_5min == 1
        assert (f.purchase_cnt_24h, f.purchase_amt_24h) == (2, 30.0)
        assert f.purchase_success_cnt_24h == 1       # the denied purchase is not a success
        assert (f.purchase_cnt_30d, f.purchase_amt_30d) == (4, 100.0)
        assert f.purchase_cnt_90d == 5
        assert f.purchase_success_amt_90d == 130.0
        assert f.pur_median_amt_90d == 30.0

    async def test_db_fallback_matches_redis_path(self, redis):
        # The same history must give the same purchase features whether it is read
        # from Postgres (cold start) or from Redis (warm path).
        from_db = await build_features(_txn(), _mock_db(_history_rows()), redis)

        for tid, amount, pt, decision in _history_rows():
            earlier = _txn(purchase_time=pt.replace(tzinfo=None), purchase_value=amount)
            await _record(redis, earlier, tid, decision)
        db = _mock_db()
        from_redis = await build_features(_txn(), db, redis)

        db.execute.assert_not_awaited()
        purchase_fields = [k for k in from_db.model_dump()
                           if k.startswith("purchase_") and k not in ("purchase_value", "purchase_time")]
        purchase_fields.append("pur_median_amt_90d")
        assert len(purchase_fields) > 20
        for k in purchase_fields:
            assert getattr(from_db, k) == pytest.approx(getattr(from_redis, k)), k

    async def test_redis_history_skips_db(self, redis):
        earlier = _txn(purchase_time=datetime(2015, 6, 1, 11, 0, 0), purchase_value=25.0)
        await _record(redis, earlier, "t1")
        db = _mock_db()

        f = await build_features(_txn(), db, redis)

        db.execute.assert_not_awaited()
        assert f.purchase_cnt_1h == 1        # exactly 1h earlier: inside the window
        assert f.purchase_amt_24h == 25.0
        assert f.time_since_last_txn_uid == 3600.0


class TestVelocityCounters:
    async def test_new_device_has_zero_counts(self, redis):
        f = await build_features(_txn(), _mock_db(), redis)
        assert (f.did_uid_90d, f.did_txn_cnt_30d, f.ip_uid_7d) == (0, 0, 0)

    async def test_counts_use_event_time_not_server_clock(self, redis):
        # Two transactions 10 days apart in EVENT time, recorded milliseconds apart in
        # real time. The 24h window must not see the older one; the 30d window must.
        old = _txn(user_id="user_0", purchase_time=datetime(2015, 5, 22, 12, 0, 0))
        await _record(redis, old, "t0")
        f = await build_features(_txn(), _mock_db(), redis)

        assert f.did_uid_24h == 0
        assert f.did_uid_30d == 1
        assert f.did_txn_cnt_24h == 0
        assert f.did_txn_cnt_30d == 1

    async def test_future_events_are_not_counted(self, redis):
        # A transaction recorded with a LATER purchase_time (out-of-order arrival)
        # must not leak into an earlier transaction's windows.
        later = _txn(user_id="user_9", purchase_time=datetime(2015, 6, 1, 12, 30, 0))
        await _record(redis, later, "t9")
        f = await build_features(_txn(), _mock_db(), redis)
        assert f.did_uid_1h == 0
