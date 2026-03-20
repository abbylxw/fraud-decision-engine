import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.data.data_models import Transaction, Features
from .request import (
    get_days_since_signup, get_days_since_signup_group, get_age_account_gap,
    get_hours_to_purchase, get_hour_of_day, get_day_of_week, get_month_of_year,
)
from .db import get_ip_country, get_did_days_on_file
from .velocity import (
    get_did_uid_counts, get_uid_did_counts, get_ip_uid_counts,
    get_purchase_counts, get_did_txn_counts, get_uid_last_txn_time,
    get_country_txn_counts, update_velocity_counters,
)


async def build_features(t: Transaction, db: AsyncSession, redis: Redis) -> Features:
    # Check Redis cache for the two DB-backed lookups (parallel)
    _ip_key  = f"cache:ip:{int(t.ip_address)}"
    _did_key = f"cache:did:{t.device_id}:{t.user_id}"
    ip_cached, did_cached = await asyncio.gather(
        redis.get(_ip_key),
        redis.get(_did_key),
    )

    # Cache hit: skip DB entirely. Cache miss: query DB and populate cache.
    if ip_cached is not None:
        ip_country = ip_cached
    else:
        ip_country = await get_ip_country(t.ip_address, db)
        await redis.set(_ip_key, ip_country, ex=86400)   # IP→country is static; 24h TTL

    if did_cached is not None:
        did_days_on_file = int(did_cached)
    else:
        did_days_on_file = await get_did_days_on_file(t.device_id, t.user_id, db)
        await redis.set(_did_key, str(did_days_on_file), ex=3600)  # device age; 1h TTL

    # Velocity + country lookups are Redis-first; gather fires all in parallel
    (
        did_uid, uid_did, ip_uid, purchase, did_txn, last_txn_time, country_txn,
    ) = await asyncio.gather(
        get_did_uid_counts(t.device_id, redis, db),
        get_uid_did_counts(t.user_id, redis, db),
        get_ip_uid_counts(t.ip_address, redis, db),
        get_purchase_counts(t.user_id, redis, db),
        get_did_txn_counts(t.device_id, redis, db),
        get_uid_last_txn_time(t.user_id, redis, db),
        get_country_txn_counts(ip_country, redis, db),
    )

    days_since_signup = get_days_since_signup(t)

    # Derived features — computed from already-fetched Redis values + transaction fields
    _user_avg_30d = purchase['amt_30d'] / max(purchase['cnt_30d'], 1)
    amount_vs_user_avg_30d = t.purchase_value / max(_user_avg_30d, 1.0)

    high_amount_vs_hist = int(
        purchase['median_amt_90d'] > 0 and
        t.purchase_value > 2 * purchase['median_amt_90d']
    )

    return Features(
        **t.model_dump(by_alias=True),
        days_since_signup         = days_since_signup,
        days_since_signup_group   = get_days_since_signup_group(days_since_signup),
        ip_country                = ip_country,
        did_uid_days_on_file      = did_days_on_file,
        did_uid_5min              = did_uid['5min'],
        did_uid_10min             = did_uid['10min'],
        did_uid_1h                = did_uid['1h'],
        did_uid_2h                = did_uid['2h'],
        did_uid_24h               = did_uid['24h'],
        did_uid_7d                = did_uid['7d'],
        did_uid_30d               = did_uid['30d'],
        did_uid_90d               = did_uid['90d'],
        uid_did_5min              = uid_did['5min'],
        uid_did_10min             = uid_did['10min'],
        uid_did_1h                = uid_did['1h'],
        uid_did_2h                = uid_did['2h'],
        uid_did_24h               = uid_did['24h'],
        uid_did_7d                = uid_did['7d'],
        uid_did_30d               = uid_did['30d'],
        uid_did_90d               = uid_did['90d'],
        ip_uid_5min               = ip_uid['5min'],
        ip_uid_10min              = ip_uid['10min'],
        ip_uid_1h                 = ip_uid['1h'],
        ip_uid_2h                 = ip_uid['2h'],
        ip_uid_24h                = ip_uid['24h'],
        ip_uid_7d                 = ip_uid['7d'],
        did_txn_cnt_5min          = did_txn['5min'],
        did_txn_cnt_10min         = did_txn['10min'],
        did_txn_cnt_1h            = did_txn['1h'],
        did_txn_cnt_2h            = did_txn['2h'],
        did_txn_cnt_24h           = did_txn['24h'],
        did_txn_cnt_7d            = did_txn['7d'],
        did_txn_cnt_30d           = did_txn['30d'],
        purchase_cnt_5min         = purchase['cnt_5min'],
        purchase_amt_5min         = purchase['amt_5min'],
        purchase_cnt_10min        = purchase['cnt_10min'],
        purchase_amt_10min        = purchase['amt_10min'],
        purchase_cnt_1h           = purchase['cnt_1h'],
        purchase_amt_1h           = purchase['amt_1h'],
        purchase_cnt_24h          = purchase['cnt_24h'],
        purchase_amt_24h          = purchase['amt_24h'],
        purchase_cnt_7d           = purchase['cnt_7d'],
        purchase_amt_7d           = purchase['amt_7d'],
        purchase_cnt_30d          = purchase['cnt_30d'],
        purchase_amt_30d          = purchase['amt_30d'],
        purchase_cnt_90d          = purchase['cnt_90d'],
        purchase_amt_90d          = purchase['amt_90d'],
        purchase_success_cnt_5min  = purchase['success_cnt_5min'],
        purchase_success_amt_5min  = purchase['success_amt_5min'],
        purchase_success_cnt_10min = purchase['success_cnt_10min'],
        purchase_success_amt_10min = purchase['success_amt_10min'],
        purchase_success_cnt_1h    = purchase['success_cnt_1h'],
        purchase_success_amt_1h    = purchase['success_amt_1h'],
        purchase_success_cnt_24h   = purchase['success_cnt_24h'],
        purchase_success_amt_24h   = purchase['success_amt_24h'],
        purchase_success_cnt_7d    = purchase['success_cnt_7d'],
        purchase_success_amt_7d    = purchase['success_amt_7d'],
        purchase_success_cnt_30d   = purchase['success_cnt_30d'],
        purchase_success_amt_30d   = purchase['success_amt_30d'],
        purchase_success_cnt_90d   = purchase['success_cnt_90d'],
        purchase_success_amt_90d   = purchase['success_amt_90d'],
        pur_median_amt_90d         = purchase['median_amt_90d'],
        amount_vs_user_avg_30d     = amount_vs_user_avg_30d,
        age_account_gap            = get_age_account_gap(t, days_since_signup),
        hours_to_purchase          = get_hours_to_purchase(t),
        hour_of_day                = get_hour_of_day(t),
        day_of_week                = get_day_of_week(t),
        month_of_year              = get_month_of_year(t),
        high_amount_vs_hist        = high_amount_vs_hist,
        time_since_last_txn_uid    = last_txn_time,
        country_txn_count_5min     = country_txn['5min'],
        country_txn_count_10min    = country_txn['10min'],
        country_txn_count_1h       = country_txn['1h'],
        country_txn_count_24h      = country_txn['24h'],
        country_txn_count_7d       = country_txn['7d'],
        country_txn_count_30d      = country_txn['30d'],
    )
