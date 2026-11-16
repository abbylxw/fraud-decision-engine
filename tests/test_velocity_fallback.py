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
from app.decisions import execute_hard_rules
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
    # A known device with no transactions yet, so the all-time-count fallback stays out of
    # the other tests (TestDeviceAllTimeCount removes this key to exercise it).
    await r.set("did:txn_all:device_1", "0")
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
    async def test_new_device_counts_only_the_buyer(self, redis):
        # Set features include the buyer (1); count features are strictly prior (0).
        f = await build_features(_txn(), _mock_db(), redis)
        assert (f.did_uid_90d, f.uid_did_90d, f.ip_uid_7d, f.did_txn_cnt_30d) == (1, 1, 1, 0)

    async def test_country_windows_are_the_three_the_model_uses(self, redis):
        # 5min/10min/1h were dropped (no model input or rule read them); 24h/7d/30d are inputs.
        f = await build_features(_txn(), _mock_db(), redis)
        assert (f.country_txn_count_24h, f.country_txn_count_7d, f.country_txn_count_30d) == (0, 0, 0)
        assert not hasattr(f, "country_txn_count_5min")

    async def test_counts_use_event_time_not_server_clock(self, redis):
        # Two transactions 10 days apart in EVENT time, recorded milliseconds apart in
        # real time. The 24h window must not see the older one; the 30d window must.
        old = _txn(user_id="user_0", purchase_time=datetime(2015, 5, 22, 12, 0, 0))
        await _record(redis, old, "t0")
        f = await build_features(_txn(), _mock_db(), redis)

        assert f.did_uid_24h == 1   # only the buyer
        assert f.did_uid_30d == 2   # user_0 + the buyer
        assert f.did_txn_cnt_24h == 0
        assert f.did_txn_cnt_30d == 1

    async def test_future_events_are_not_counted(self, redis):
        # A transaction recorded with a LATER purchase_time (out-of-order arrival)
        # must not leak into an earlier transaction's windows.
        later = _txn(user_id="user_9", purchase_time=datetime(2015, 6, 1, 12, 30, 0))
        await _record(redis, later, "t9")
        f = await build_features(_txn(), _mock_db(), redis)
        assert f.did_uid_1h == 1   # only the buyer


def _history_db(history: dict):
    """Postgres mock whose execute() answers the device-history query with
    (user_id, transaction count) rows."""
    def answer(sql, params):
        result = MagicMock()
        # The purchase-history fallback shares this mock; it gets "no rows".
        result.all.return_value = list(history.items()) if "device_id" in params else []
        return result

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=answer)
    return db


def _device_history_calls(db):
    return [c for c in db.execute.await_args_list if "device_id" in c.args[1]]


class TestDeviceAllTimeHistory:
    """did_txn_cnt_all: transactions ever seen on the device, strictly before this one.
    did_other_uid_all: OTHER accounts ever seen on it (the model input)."""

    async def test_history_has_no_window(self, redis):
        # 200 days earlier: far outside every windowed counter, still remembered here.
        old = _txn(user_id="user_0", purchase_time=PURCHASE_TIME - timedelta(days=200))
        await _record(redis, old, "t0")
        f = await build_features(_txn(), _mock_db(), redis)
        assert f.did_txn_cnt_30d == 0
        assert f.did_uid_90d == 1          # only the buyer: the 90-day set forgot user_0
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (1, 1)

    async def test_loyal_customer_on_own_device_has_no_other_accounts(self, redis):
        # The reason did_other_uid_all replaced did_txn_cnt_all as the model input: the same
        # account coming back is history, but it is not ANOTHER account.
        for i, days in enumerate((60, 30)):
            await _record(redis, _txn(purchase_time=PURCHASE_TIME - timedelta(days=days)), f"t{i}")
        f = await build_features(_txn(), _mock_db(), redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (2, 0)

    async def test_other_accounts_are_distinct(self, redis):
        # user_0 twice and user_9 once before user_1: three transactions, two other accounts.
        for i, user in enumerate(("user_0", "user_0", "user_9")):
            await _record(redis, _txn(user_id=user,
                                      purchase_time=PURCHASE_TIME - timedelta(days=100 + i)), f"t{i}")
        f = await build_features(_txn(), _mock_db(), redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (3, 2)

    async def test_redis_history_skips_db(self, redis):
        await _record(redis, _txn(user_id="user_0",
                                  purchase_time=PURCHASE_TIME - timedelta(days=5)), "t0")
        db = _history_db({"someone": 99})
        f = await build_features(_txn(), db, redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (1, 1)
        assert _device_history_calls(db) == []

    async def test_new_device_asks_db_as_of_transaction_time(self, redis):
        await redis.delete("did:txn_all:device_1")
        db = _history_db({})
        f = await build_features(_txn(), db, redis)

        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (0, 0)
        (call,) = _device_history_calls(db)
        assert call.args[1] == {"device_id": "device_1",
                                "as_of": PURCHASE_TIME.replace(tzinfo=timezone.utc)}
        assert "CAST(:as_of AS timestamptz)" in str(call.args[0])

    async def test_db_answer_excludes_the_buyer(self, redis):
        await redis.delete("did:txn_all:device_1")
        f = await build_features(_txn(), _history_db({"user_1": 4, "user_0": 2, "user_9": 1}), redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (7, 2)

    async def test_db_answer_seeds_redis_so_history_continues(self, redis):
        # Redis lost the keys; Postgres knows 7 transactions by 3 accounts (the buyer is
        # one). After this transaction is recorded the next buyer must see 8 transactions
        # and 3 other accounts, without another Postgres query.
        await redis.delete("did:txn_all:device_1")
        t = _txn()
        await build_features(t, _history_db({"user_1": 4, "user_0": 2, "user_9": 1}), redis)
        await _record(redis, t, "t1")

        await redis.set("cache:did:device_1:user_7", "0")
        db = _history_db({})
        f = await build_features(_txn(user_id="user_7"), db, redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (8, 3)
        assert _device_history_calls(db) == []

    async def test_counter_without_account_set_falls_back(self, redis):
        # Keys written by the one-day-old version that only had the counter: the counter
        # says 5 but there is no account set. Postgres fills in the accounts.
        await redis.set("did:txn_all:device_1", "5")
        f = await build_features(_txn(), _history_db({"user_0": 3, "user_9": 2}), redis)
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (5, 2)
        assert await redis.scard("did:uids_all:device_1") == 2

    async def test_history_is_arrival_order_not_event_time(self, redis):
        # Known trade-off (docs/design_tradeoffs_and_todos.md, 2026-11-15): a counter and a
        # plain set cannot tell that an already-recorded transaction has a LATER
        # purchase_time, so an out-of-order arrival is counted. The windowed counters do
        # not have this gap.
        later = _txn(user_id="user_9", purchase_time=PURCHASE_TIME + timedelta(minutes=30))
        await _record(redis, later, "t9")
        f = await build_features(_txn(), _mock_db(), redis)
        assert f.did_txn_cnt_1h == 0
        assert (f.did_txn_cnt_all, f.did_other_uid_all) == (1, 1)


class TestDeviceManyUsers1hRedisPath:
    """device_many_users_1h on real build_features() output (fakeredis), not hand-set values."""

    async def test_second_user_on_device_denied_on_first_attempt(self, redis):
        b = _txn(user_id="user_0", purchase_time=PURCHASE_TIME - timedelta(minutes=10))
        await _record(redis, b, "tb")
        a = _txn(user_id="user_1")
        f = await build_features(a, _mock_db(), redis)
        assert f.did_uid_1h == 2
        assert execute_hard_rules(a, f) == ("deny", "device_many_users_1h")

    async def test_single_user_on_fresh_device_allowed(self, redis):
        a = _txn(user_id="user_1")
        f = await build_features(a, _mock_db(), redis)
        assert f.did_uid_1h == 1
        assert execute_hard_rules(a, f) == ("allow", None)
