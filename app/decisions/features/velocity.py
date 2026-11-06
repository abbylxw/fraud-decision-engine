"""
Velocity counter storage (Redis sorted sets) and the Postgres fallback for purchase history.

Every counter is scored by the transaction's EVENT time (purchase_time, via
request.event_ts), never the server clock, so serving counts windows exactly the way
training/build_features.py does. tests/test_feature_parity.py enforces this.

Reads happen in build_features() (features/__init__.py) as one Redis pipeline.
"""
import asyncio
import logging
import statistics
from datetime import datetime, timezone
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("fraud_engine")

NINETY_DAYS = 90 * 86400


# ---------------------------------------------------------------------------
# DB fallback — called by build_features() when Redis has no purchase history
# for the user (cold start, eviction). One SQL query with COUNT(CASE WHEN ...)
# per time window, all relative to :as_of (the transaction's own time).
# ---------------------------------------------------------------------------

async def _purchase_counts_from_db(user_id: str, db: AsyncSession, as_of: float) -> dict:
    as_of_dt = datetime.fromtimestamp(as_of, tz=timezone.utc)
    row = (await db.execute(text("""
        SELECT
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '5 minutes'  THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '5 minutes'  THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '10 minutes' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '10 minutes' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '1 hour'    THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '1 hour'    THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '24 hours'  THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '24 hours'  THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '7 days'    THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '7 days'    THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '30 days'   THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '30 days'   THEN purchase_value ELSE 0 END),
          COUNT(1),
          SUM(purchase_value),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '5 minutes'  AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '5 minutes'  AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '10 minutes' AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '10 minutes' AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '1 hour'    AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '1 hour'    AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '24 hours'  AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '24 hours'  AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '7 days'    AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '7 days'    AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= :as_of - INTERVAL '30 days'   AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= :as_of - INTERVAL '30 days'   AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN fraud_decision != 'deny' THEN purchase_value ELSE 0 END)
        FROM transactions
        WHERE user_id = :user_id
          AND purchase_time >= :as_of - INTERVAL '90 days'
          AND purchase_time <= :as_of
    """), {"user_id": user_id, "as_of": as_of_dt})).one()

    windows = ["5min", "10min", "1h", "24h", "7d", "30d", "90d"]
    out = {}
    for i, w in enumerate(windows):
        out[f"cnt_{w}"]         = int(row[i * 2]     or 0)
        out[f"amt_{w}"]         = float(row[i * 2 + 1] or 0.0)
    for i, w in enumerate(windows):
        out[f"success_cnt_{w}"] = int(row[14 + i * 2]     or 0)
        out[f"success_amt_{w}"] = float(row[14 + i * 2 + 1] or 0.0)

    # Median: fetch raw 90d amounts for Python-side calculation
    amounts = [float(r[0]) for r in (await db.execute(text("""
        SELECT purchase_value FROM transactions
        WHERE user_id = :user_id
          AND purchase_time >= :as_of - INTERVAL '90 days'
          AND purchase_time <= :as_of
    """), {"user_id": user_id, "as_of": as_of_dt})).all()]
    out["median_amt_90d"] = statistics.median(amounts) if amounts else 0.0

    return out


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
    ]
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
