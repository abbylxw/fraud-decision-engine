from redis.asyncio import Redis
import time
import asyncio
from uuid import UUID


async def get_did_uid_counts(device_id: str, redis: Redis) -> dict:
    # Returns counts of distinct user IDs seen on this device_id
    # Keys: '2h', '24h', '7d', '30d', '90d'
    current_time = time.time()
    windows = {
        "2h": current_time - (2 * 3600),
        '24h': current_time - (24 * 3600),
        '7d': current_time - (7 * 86400),
        '30d': current_time - (30 * 86400),
        '90d': current_time - (90 * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"did:uid:{device_id}", since, current_time)
                                     for since in windows.values()])
    out = {}
    for key, members in zip(windows, results):
        out[key] = len(members)
    return out

async def get_uid_did_counts(user_id: str, redis: Redis) -> dict:
    # Returns counts of distinct device IDs seen for this user_id
    # Keys: '2h', '24h', '7d', '30d', '90d'
    current_time = time.time()
    windows = {
        "2h": current_time - (2 * 3600),
        '24h': current_time - (24 * 3600),
        '7d': current_time - (7 * 86400),
        '30d': current_time - (30 * 86400),
        '90d': current_time - (90 * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"uid:did:{user_id}", since, current_time)
                                     for since in windows.values()])
    out = {}
    for key, members in zip(windows, results):
        out[key] = len(members)
    return out


async def get_ip_uid_counts(ip: float, redis: Redis) -> dict:
    # Returns counts of distinct user IDs seen from this IP
    # Keys: '2h', '24h', '7d'
    current_time = time.time()
    windows = {
        "2h": current_time - (2 * 3600),
        '24h': current_time - (24 * 3600),
        '7d': current_time - (7 * 86400),
    }
    results = await asyncio.gather(*[redis.zrangebyscore(f"ip:uid:{ip}", since, current_time)
                                     for since in windows.values()])
    out = {}
    for key, members in zip(windows, results):
        out[key] = len(members)
    return out



async def get_purchase_counts(user_id: str, redis: Redis) -> dict:
    current_time = time.time()
    windows = {
        "24h": current_time - 24 * 3600,
        "7d":  current_time - 7  * 86400,
        "30d": current_time - 30 * 86400,
        "90d": current_time - 90 * 86400,
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
    return out



    
async def update_velocity_counters(user_id: str, device_id: str, ip: float,
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
    )