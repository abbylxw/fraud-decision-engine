"""
Velocity counter storage (Redis sorted sets) and the Postgres fallback for purchase history.

Every counter is scored by the transaction's EVENT time (purchase_time, via
request.event_ts), never the server clock, so serving counts windows exactly the way
training/build_features.py does. tests/test_feature_parity.py enforces this.

Reads happen in build_features() (features/__init__.py) as one Redis pipeline.
"""
import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("fraud_engine")

NINETY_DAYS = 90 * 86400


# ---------------------------------------------------------------------------
# DB fallback — called by build_features() when Redis has no purchase history
# for the user (cold start, eviction). One query fetches the user's last 90 days
# of rows (an index lookup on idx_transactions_user_time, see db_init/006); the
# windows, sums and median are then computed by the same parser the Redis path
# uses, so the two paths cannot disagree. All windows count back from :as_of
# (the transaction's own time). asyncpg sends :as_of untyped, and Postgres would
# read `:as_of - INTERVAL` as interval - interval, so it is cast explicitly.
# ---------------------------------------------------------------------------

async def _purchase_counts_from_db(user_id: str, db: AsyncSession, as_of: float) -> dict:
    # Imported here: features/__init__.py imports this module at load time.
    from . import _parse_purchase_entries

    as_of_dt = datetime.fromtimestamp(as_of, tz=timezone.utc)
    rows = (await db.execute(text("""
        SELECT transaction_id, purchase_value, purchase_time, fraud_decision
        FROM transactions
        WHERE user_id = :user_id
          AND purchase_time >= CAST(:as_of AS timestamptz) - INTERVAL '90 days'
          AND purchase_time <= :as_of
    """), {"user_id": user_id, "as_of": as_of_dt})).all()

    # Same (member, score) shape as the Redis purchase:all / purchase:success sets.
    all_entries = [(f"{tid}:{float(value)}", pt.timestamp()) for tid, value, pt, _ in rows]
    suc_entries = [(f"{tid}:{float(value)}", pt.timestamp())
                   for tid, value, pt, decision in rows if decision != 'deny']
    windows = {
        "5min": as_of - 300, "10min": as_of - 600, "1h": as_of - 3600,
        "24h": as_of - 86400, "7d": as_of - 604800, "30d": as_of - 2592000,
        "90d": as_of - NINETY_DAYS,
    }
    return _parse_purchase_entries(all_entries, suc_entries, windows)


async def _device_history_from_db(device_id: str, db: AsyncSession, as_of: float) -> dict[str, int]:
    """Every account that has transacted on this device up to the transaction's own time,
    with its number of transactions: {user_id: count}.

    Fallback for the two all-time device features (did_txn_cnt_all, did_other_uid_all) when
    Redis has no history for the device: either the device is new (the usual case, empty
    answer) or Redis lost the keys. One index lookup on idx_transactions_device_time
    (db_init/008) answers both features."""
    as_of_dt = datetime.fromtimestamp(as_of, tz=timezone.utc)
    res = await db.execute(text("""
        SELECT user_id, COUNT(*)
        FROM transactions
        WHERE device_id = :device_id
          AND purchase_time <= CAST(:as_of AS timestamptz)
        GROUP BY user_id
    """), {"device_id": device_id, "as_of": as_of_dt})
    return {str(user_id): int(n) for user_id, n in res.all()}


# ---------------------------------------------------------------------------
# Counter updates — called fire-and-forget after the decision is returned
# ---------------------------------------------------------------------------

async def update_velocity_counters(user_id: str, device_id: str, ip: int,
                                   ip_country: str,
                                   transaction_id: UUID | str,
                                   purchase_value: float, fraud_decision: str,
                                   event_ts: float,
                                   redis: Redis) -> None:
    """Record this transaction in every counter family, scored by its event time.

    event_ts is the transaction's purchase_time (request.event_ts), so a replay of
    historical data fills the windows exactly as the training replay does.
    """
    ninety_days_ago = event_ts - NINETY_DAYS
    coros = [
        # "Distinct X" sets: member = the X, score = when it was last seen.
        redis.zadd(f"uid:did:{user_id}",      {device_id: event_ts}),
        redis.zadd(f"did:uid:{device_id}",    {user_id: event_ts}),
        redis.zadd(f"ip:uid:{ip}",            {user_id: event_ts}),
        # "Every transaction" sets: member must be unique per transaction, otherwise two
        # purchases in the same second collapse into one entry and undercount.
        redis.zadd(f"purchase:all:{user_id}", {f"{transaction_id}:{purchase_value}": event_ts}),
        redis.zadd(f"did:txn:{device_id}",    {f"{transaction_id}": event_ts}),
        redis.zadd(f"country:txn:{ip_country}", {f"{transaction_id}": event_ts}),
        redis.set(f"uid:last_txn:{user_id}", str(event_ts)),
        # All-time transactions on the device (model feature did_txn_cnt_all). A plain
        # counter with no TTL and no 90-day purge: the signal is precisely the device
        # coming back after the windowed counters have forgotten it.
        redis.incr(f"did:txn_all:{device_id}"),
        # Every account ever seen on the device (model feature did_other_uid_all = this
        # set minus the buyer). Also never expires.
        redis.sadd(f"did:uids_all:{device_id}", user_id),
    ]
    # "Successful" = not denied. A "review" counts as successful here: its real outcome is
    # unknown at decision time (no label feedback yet; docs/production_readiness.md).
    if fraud_decision != 'deny':
        coros.append(
            redis.zadd(f"purchase:success:{user_id}", {f"{transaction_id}:{purchase_value}": event_ts})
        )
    await asyncio.gather(*coros)

    # Purge entries strictly older than 90 days. The "(" makes the bound exclusive:
    # an entry exactly 90 days old is still inside a 90d window (training counts
    # ts >= cutoff), so it must survive for a later transaction at the same second.
    cutoff = f"({ninety_days_ago}"
    await asyncio.gather(
        redis.zremrangebyscore(f"uid:did:{user_id}",         "-inf", cutoff),
        redis.zremrangebyscore(f"did:uid:{device_id}",       "-inf", cutoff),
        redis.zremrangebyscore(f"ip:uid:{ip}",               "-inf", cutoff),
        redis.zremrangebyscore(f"purchase:all:{user_id}",    "-inf", cutoff),
        redis.zremrangebyscore(f"purchase:success:{user_id}","-inf", cutoff),
        redis.zremrangebyscore(f"did:txn:{device_id}",       "-inf", cutoff),
        redis.zremrangebyscore(f"country:txn:{ip_country}",  "-inf", cutoff),
    )
