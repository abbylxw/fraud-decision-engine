import asyncio
import statistics
import time
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# DB fallback helpers — called on Redis cache miss (key absent entirely)
# Each runs a single SQL query with COUNT(CASE WHEN ...) per time window.
# ---------------------------------------------------------------------------

async def _did_uid_from_db(device_id: str, db: AsyncSession) -> dict:
    row = (await db.execute(text("""
        SELECT
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'     THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '2 hours'    THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'   THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'     THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'    THEN user_id END),
          COUNT(DISTINCT user_id)
        FROM transactions
        WHERE device_id = :device_id AND purchase_time >= NOW() - INTERVAL '90 days'
    """), {"device_id": device_id})).one()
    keys = ["5min", "10min", "1h", "2h", "24h", "7d", "30d", "90d"]
    return {k: int(v or 0) for k, v in zip(keys, row)}


async def _uid_did_from_db(user_id: str, db: AsyncSession) -> dict:
    row = (await db.execute(text("""
        SELECT
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'     THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '2 hours'    THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'   THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'     THEN device_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'    THEN device_id END),
          COUNT(DISTINCT device_id)
        FROM transactions
        WHERE user_id = :user_id AND purchase_time >= NOW() - INTERVAL '90 days'
    """), {"user_id": user_id})).one()
    keys = ["5min", "10min", "1h", "2h", "24h", "7d", "30d", "90d"]
    return {k: int(v or 0) for k, v in zip(keys, row)}


async def _ip_uid_from_db(ip: float, db: AsyncSession) -> dict:
    row = (await db.execute(text("""
        SELECT
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'     THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '2 hours'    THEN user_id END),
          COUNT(DISTINCT CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'   THEN user_id END),
          COUNT(DISTINCT user_id)
        FROM transactions
        WHERE ip_address = :ip AND purchase_time >= NOW() - INTERVAL '7 days'
    """), {"ip": ip})).one()
    keys = ["5min", "10min", "1h", "2h", "24h", "7d"]
    return {k: int(v or 0) for k, v in zip(keys, row)}


async def _purchase_counts_from_db(user_id: str, db: AsyncSession) -> dict:
    # One query for all counts, amounts, and success-only variants across all windows
    row = (await db.execute(text("""
        SELECT
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'    THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'    THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'  THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'  THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'    THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'    THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'   THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'   THEN purchase_value ELSE 0 END),
          COUNT(1),
          SUM(purchase_value),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'    AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'    AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'  AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'  AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'    AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'    AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'   AND fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN purchase_time >= NOW() - INTERVAL '30 days'   AND fraud_decision != 'deny' THEN purchase_value ELSE 0 END),
          COUNT(CASE WHEN fraud_decision != 'deny' THEN 1 END),
          SUM(CASE WHEN fraud_decision != 'deny' THEN purchase_value ELSE 0 END)
        FROM transactions
        WHERE user_id = :user_id AND purchase_time >= NOW() - INTERVAL '90 days'
    """), {"user_id": user_id})).one()

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
        WHERE user_id = :user_id AND purchase_time >= NOW() - INTERVAL '90 days'
    """), {"user_id": user_id})).all()]
    out["median_amt_90d"] = statistics.median(amounts) if amounts else 0.0

    return out


async def _uid_last_txn_from_db(user_id: str, db: AsyncSession) -> float:
    row = (await db.execute(text("""
        SELECT EXTRACT(EPOCH FROM (NOW() - MAX(purchase_time)))
        FROM transactions WHERE user_id = :user_id
    """), {"user_id": user_id})).one()
    val = row[0]
    return max(0.0, float(val)) if val is not None else 0.0


async def _did_txn_from_db(device_id: str, db: AsyncSession) -> dict:
    row = (await db.execute(text("""
        SELECT
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'     THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '2 hours'    THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'   THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'     THEN 1 END),
          COUNT(1)
        FROM transactions
        WHERE device_id = :device_id AND purchase_time >= NOW() - INTERVAL '30 days'
    """), {"device_id": device_id})).one()
    keys = ["5min", "10min", "1h", "2h", "24h", "7d", "30d"]
    return {k: int(v or 0) for k, v in zip(keys, row)}


async def _country_txn_from_db(ip_country: str, db: AsyncSession) -> dict:
    row = (await db.execute(text("""
        SELECT
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '5 minutes'  THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '10 minutes' THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '1 hour'     THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '24 hours'   THEN 1 END),
          COUNT(CASE WHEN purchase_time >= NOW() - INTERVAL '7 days'     THEN 1 END),
          COUNT(1)
        FROM transactions
        WHERE ip_country = :ip_country AND purchase_time >= NOW() - INTERVAL '30 days'
    """), {"ip_country": ip_country})).one()
    keys = ["5min", "10min", "1h", "24h", "7d", "30d"]
    return {k: int(v or 0) for k, v in zip(keys, row)}


# ---------------------------------------------------------------------------
# Public get_* functions — try Redis first, fall back to DB on cache miss
# ---------------------------------------------------------------------------

async def get_did_uid_counts(device_id: str, redis: Redis, db: AsyncSession) -> dict:
    # Returns counts of distinct user IDs seen on this device_id
    # Keys: '5min', '10min', '1h', '2h', '24h', '7d', '30d', '90d'
    if not await redis.exists(f"did:uid:{device_id}"):
        return await _did_uid_from_db(device_id, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - (5  * 60),
        "10min": current_time - (10 * 60),
        "1h":    current_time - (1  * 3600),
        "2h":    current_time - (2  * 3600),
        "24h":   current_time - (24 * 3600),
        "7d":    current_time - (7  * 86400),
        "30d":   current_time - (30 * 86400),
        "90d":   current_time - (90 * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"did:uid:{device_id}", since, current_time)
                                     for since in windows.values()])
    return {key: len(members) for key, members in zip(windows, results)}


async def get_uid_did_counts(user_id: str, redis: Redis, db: AsyncSession) -> dict:
    # Returns counts of distinct device IDs seen for this user_id
    # Keys: '5min', '10min', '1h', '2h', '24h', '7d', '30d', '90d'
    if not await redis.exists(f"uid:did:{user_id}"):
        return await _uid_did_from_db(user_id, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - (5  * 60),
        "10min": current_time - (10 * 60),
        "1h":    current_time - (1  * 3600),
        "2h":    current_time - (2  * 3600),
        "24h":   current_time - (24 * 3600),
        "7d":    current_time - (7  * 86400),
        "30d":   current_time - (30 * 86400),
        "90d":   current_time - (90 * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"uid:did:{user_id}", since, current_time)
                                     for since in windows.values()])
    return {key: len(members) for key, members in zip(windows, results)}


async def get_ip_uid_counts(ip: float, redis: Redis, db: AsyncSession) -> dict:
    # Returns counts of distinct user IDs seen from this IP
    # Keys: '5min', '10min', '1h', '2h', '24h', '7d'
    if not await redis.exists(f"ip:uid:{ip}"):
        return await _ip_uid_from_db(ip, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - (5  * 60),
        "10min": current_time - (10 * 60),
        "1h":    current_time - (1  * 3600),
        "2h":    current_time - (2  * 3600),
        "24h":   current_time - (24 * 3600),
        "7d":    current_time - (7  * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"ip:uid:{ip}", since, current_time)
                                     for since in windows.values()])
    return {key: len(members) for key, members in zip(windows, results)}


async def get_purchase_counts(user_id: str, redis: Redis, db: AsyncSession) -> dict:
    if not await redis.exists(f"purchase:all:{user_id}"):
        return await _purchase_counts_from_db(user_id, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - 5  * 60,
        "10min": current_time - 10 * 60,
        "1h":    current_time - 1  * 3600,
        "24h":   current_time - 24 * 3600,
        "7d":    current_time - 7  * 86400,
        "30d":   current_time - 30 * 86400,
        "90d":   current_time - 90 * 86400,
    }
    all_results, success_results = await asyncio.gather(
        asyncio.gather(*[
            redis.zrangebyscore(f"purchase:all:{user_id}", since, current_time)
            for since in windows.values()
        ]),
        asyncio.gather(*[
            redis.zrangebyscore(f"purchase:success:{user_id}", since, current_time)
            for since in windows.values()
        ]),
    )
    out = {}
    for key, members in zip(windows, all_results):
        out[f"cnt_{key}"] = len(members)
        out[f"amt_{key}"] = sum(float(m.rsplit(":", 1)[1]) for m in members)
    for key, members in zip(windows, success_results):
        out[f"success_cnt_{key}"] = len(members)
        out[f"success_amt_{key}"] = sum(float(m.rsplit(":", 1)[1]) for m in members)

    # Median purchase amount in the 90d window — reuses already-fetched all_results[-1]
    amounts_90d = sorted(float(m.rsplit(":", 1)[1]) for m in all_results[-1])
    if amounts_90d:
        n, mid = len(amounts_90d), len(amounts_90d) // 2
        out["median_amt_90d"] = (
            amounts_90d[mid] if n % 2 else (amounts_90d[mid - 1] + amounts_90d[mid]) / 2
        )
    else:
        out["median_amt_90d"] = 0.0

    return out


async def get_uid_last_txn_time(user_id: str, redis: Redis, db: AsyncSession) -> float:
    """Seconds since this user's last transaction. 0.0 if no prior transaction."""
    val = await redis.get(f"uid:last_txn:{user_id}")
    if val is None:
        return await _uid_last_txn_from_db(user_id, db)
    return max(0.0, time.time() - float(val))


async def get_did_txn_counts(device_id: str, redis: Redis, db: AsyncSession) -> dict:
    """Count of all transactions on this device per time window (no user dedup).
    Keys: '5min', '10min', '1h', '2h', '24h', '7d', '30d'"""
    if not await redis.exists(f"did:txn:{device_id}"):
        return await _did_txn_from_db(device_id, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - (5  * 60),
        "10min": current_time - (10 * 60),
        "1h":    current_time - (1  * 3600),
        "2h":    current_time - (2  * 3600),
        "24h":   current_time - (24 * 3600),
        "7d":    current_time - (7  * 86400),
        "30d":   current_time - (30 * 86400),
    }
    results = await asyncio.gather(*[
        redis.zrangebyscore(f"did:txn:{device_id}", since, current_time)
        for since in windows.values()
    ])
    return {key: len(members) for key, members in zip(windows, results)}


async def get_country_txn_counts(ip_country: str, redis: Redis, db: AsyncSession) -> dict:
    """Count of all transactions from this country per time window.
    Keys: '5min', '10min', '1h', '24h', '7d', '30d'"""
    if not await redis.exists(f"country:txn:{ip_country}"):
        return await _country_txn_from_db(ip_country, db)
    current_time = time.time()
    windows = {
        "5min":  current_time - (5  * 60),
        "10min": current_time - (10 * 60),
        "1h":    current_time - (1  * 3600),
        "24h":   current_time - (24 * 3600),
        "7d":    current_time - (7  * 86400),
        "30d":   current_time - (30 * 86400),
    }
    results = await asyncio.gather(*[
        redis.zrangebyscore(f"country:txn:{ip_country}", since, current_time)
        for since in windows.values()
    ])
    return {key: len(members) for key, members in zip(windows, results)}


async def update_velocity_counters(user_id: str, device_id: str, ip: float,
                                   ip_country: str,
                                   transaction_id: UUID,
                                   purchase_value: float, fraud_decision: str,
                                   redis: Redis) -> None:
    # Called after fraud decision; ZADD entries to sorted sets for all counter families
    current_time = time.time()
    ninety_days_ago = current_time - 90 * 86400
    coros = [
        redis.zadd(f"uid:did:{user_id}",      {device_id: current_time}),
        redis.zadd(f"did:uid:{device_id}",    {user_id: current_time}),
        redis.zadd(f"ip:uid:{ip}",            {user_id: current_time}),
        redis.zadd(f"purchase:all:{user_id}", {f"{transaction_id}:{purchase_value}": current_time}),
        redis.zadd(f"did:txn:{device_id}", {f"{user_id}:{current_time:.6f}": current_time}),
        redis.zadd(f"country:txn:{ip_country}", {f"{transaction_id}": current_time}),
        redis.set(f"uid:last_txn:{user_id}", str(current_time)),
    ]
    if fraud_decision != 'deny':
        coros.append(
            redis.zadd(f"purchase:success:{user_id}", {f"{transaction_id}:{purchase_value}": current_time})
        )
    await asyncio.gather(*coros)

    # Purge entries older than 90 days from all sets
    await asyncio.gather(
        redis.zremrangebyscore(f"uid:did:{user_id}",         "-inf", ninety_days_ago),
        redis.zremrangebyscore(f"did:uid:{device_id}",       "-inf", ninety_days_ago),
        redis.zremrangebyscore(f"ip:uid:{ip}",               "-inf", ninety_days_ago),
        redis.zremrangebyscore(f"purchase:all:{user_id}",    "-inf", ninety_days_ago),
        redis.zremrangebyscore(f"purchase:success:{user_id}","-inf", ninety_days_ago),
        redis.zremrangebyscore(f"did:txn:{device_id}",       "-inf", ninety_days_ago),
        redis.zremrangebyscore(f"country:txn:{ip_country}",  "-inf", ninety_days_ago),
    )
