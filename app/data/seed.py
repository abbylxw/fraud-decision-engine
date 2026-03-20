"""
Seed script for realistic load testing.

Pre-populates:
  - PostgreSQL device_data table: randomised device age per (device_id, user_id) pair,
    capped by account age so a device can't predate the account
  - Redis velocity sorted sets: one historical entry per CSV row within the last 30 days

Run once before client_mimic.py:
    python app/data/seed.py
"""
import asyncio
import csv
import random
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg
from redis.asyncio import Redis

from app.config import settings

DATA_PATH = Path(__file__).parent / "Fraud_Data.csv"
HISTORY_WINDOW = 30 * 86400  # seed Redis entries within the last 30 days


def load_rows():
    with open(DATA_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _random_device_age(signup_time_str: str) -> int:
    """
    Returns a random device age in days between 0 and account_age.

    Uses a triangular distribution with mode=0 so most devices are newer,
    but some are as old as the account itself.
    Account age is computed from signup_time in the CSV.
    """
    if not signup_time_str:
        return 0
    signup = datetime.fromisoformat(signup_time_str.replace(" ", "T")).date()
    account_age = max((date.today() - signup).days, 0)
    if account_age == 0:
        return 0
    return int(random.triangular(0, account_age, 0))


async def seed_db(rows: list[dict]) -> None:
    # Replace Docker-internal hostname 'db' with 'localhost' for host-side runs
    db_url = (
        settings.database_url
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("@db/", "@localhost/")
    )
    conn = await asyncpg.connect(db_url)

    # Collect unique (device_id, user_id) pairs; keep first occurrence's signup_time
    pairs: dict[tuple, str] = {}
    for r in rows:
        key = (r["device_id"], r["user_id"])
        if key not in pairs:
            pairs[key] = r["signup_time"]

    records = [
        (d, u, date.today() - timedelta(days=_random_device_age(signup)))
        for (d, u), signup in pairs.items()
    ]

    await conn.executemany(
        "INSERT INTO device_data(device_id, user_id, did_uid_date) "
        "VALUES($1, $2, $3) ON CONFLICT DO NOTHING",
        records,
    )
    await conn.close()

    ages = sorted((date.today() - r[2]).days for r in records)
    print(
        f"DB seeded: {len(records)} device+user pairs | "
        f"age range: {ages[0]}–{ages[-1]}d | median: {ages[len(ages) // 2]}d"
    )


async def seed_redis(rows: list[dict]) -> None:
    # Use localhost when running from the host (settings.redis_host is 'redis' inside Docker)
    redis_host = "localhost" if settings.redis_host != "localhost" else settings.redis_host
    redis = Redis(host=redis_host, port=settings.redis_port, decode_responses=True)
    now = time.time()
    pipe = redis.pipeline()

    for row in rows:
        uid = row["user_id"]
        did = row["device_id"]
        ip = row.get("ip_address", "")
        val = row.get("purchase_value", "0")
        ts = now - random.uniform(0, HISTORY_WINDOW)

        pipe.zadd(f"did:uid:{did}", {uid: ts})
        pipe.zadd(f"uid:did:{uid}", {did: ts})
        if ip:
            pipe.zadd(f"ip:uid:{int(float(ip))}", {uid: ts})
        pipe.zadd(f"purchase:all:{uid}", {f"{uuid.uuid4()}:{val}": ts})

    await pipe.execute()
    await redis.aclose()
    print(f"Redis seeded: {len(rows)} entries across velocity sorted sets")


async def main() -> None:
    print("Loading CSV...")
    rows = load_rows()
    print(f"Loaded {len(rows)} rows. Seeding...")
    await seed_db(rows)
    await seed_redis(rows)
    print("Done. Flush idempotency cache then run client_mimic.py:")
    print("  docker compose exec redis redis-cli FLUSHDB")
    print("  python app/data/client_mimic.py")


asyncio.run(main())
