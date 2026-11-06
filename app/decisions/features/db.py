from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_ip_country(ip_int: int, db: AsyncSession) -> str:
    stmt = text("""
        SELECT country
        FROM ip_country
        WHERE ip_range @> CAST(:ip AS bigint)
        LIMIT 1
    """)
    res = await db.execute(stmt, {"ip": ip_int})
    return res.scalar_one_or_none() or 'Unknown'


async def get_did_days_on_file(device_id: str, user_id: str, db: AsyncSession,
                               as_of_date: date) -> int:
    """Days since this (device, user) pair was first seen, as of the transaction's date
    (not the server's CURRENT_DATE - see event_ts in request.py)."""
    stmt = text("""
        WITH ins AS (
            INSERT INTO device_data (device_id, user_id, did_uid_date)
            VALUES (:device_id, :user_id, CAST(:as_of_date AS date))
            ON CONFLICT (device_id, user_id) DO NOTHING
            RETURNING 1
        )
        SELECT COALESCE(
            (SELECT 0 FROM ins LIMIT 1),
            (SELECT GREATEST(CAST(:as_of_date AS date) - did_uid_date, 0)::int
             FROM device_data
             WHERE device_id = :device_id AND user_id = :user_id
             LIMIT 1),
            0
        ) AS did_uid_days_on_file;
    """)
    res = await db.execute(stmt, {"device_id": device_id, "user_id": user_id,
                                  "as_of_date": as_of_date})
    return res.scalar_one()

