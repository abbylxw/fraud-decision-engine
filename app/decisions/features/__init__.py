import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
from app.data.data_models import Transaction, Features
from .request import get_days_since_signup, get_age_group
from .db import get_ip_country, get_did_days_on_file
from .velocity import get_did_uid_counts, get_uid_did_counts, get_ip_uid_counts, get_purchase_counts


async def build_features(t: Transaction, db: AsyncSession, redis: Redis) -> Features:
    ip_country, did_days_on_file, did_uid, uid_did, ip_uid, purchase = await asyncio.gather(
        get_ip_country(t.ip_address, db),
        get_did_days_on_file(t.device_id, t.user_id, db),
        get_did_uid_counts(t.device_id, redis),
        get_uid_did_counts(t.user_id, redis),
        get_ip_uid_counts(t.ip_address, redis),
        get_purchase_counts(t.user_id, redis),
    )
    return Features(
        **t.model_dump(by_alias=True),
        days_since_signup         = get_days_since_signup(t),
        age_group                 = get_age_group(t),
        ip_country                = ip_country,
        did_uid_days_on_file      = did_days_on_file,
        did_uid_2h                = did_uid['2h'],
        did_uid_24h               = did_uid['24h'],
        did_uid_7d                = did_uid['7d'],
        did_uid_30d               = did_uid['30d'],
        did_uid_90d               = did_uid['90d'],
        uid_did_2h                = uid_did['2h'],
        uid_did_24h               = uid_did['24h'],
        uid_did_7d                = uid_did['7d'],
        uid_did_30d               = uid_did['30d'],
        uid_did_90d               = uid_did['90d'],
        ip_uid_2h                 = ip_uid['2h'],
        ip_uid_24h                = ip_uid['24h'],
        ip_uid_7d                 = ip_uid['7d'],
        purchase_cnt_24h          = purchase['cnt_24h'],
        purchase_amt_24h          = purchase['amt_24h'],
        purchase_cnt_7d           = purchase['cnt_7d'],
        purchase_amt_7d           = purchase['amt_7d'],
        purchase_cnt_30d          = purchase['cnt_30d'],
        purchase_amt_30d          = purchase['amt_30d'],
        purchase_cnt_90d          = purchase['cnt_90d'],
        purchase_amt_90d          = purchase['amt_90d'],
        purchase_success_cnt_24h  = purchase['success_cnt_24h'],
        purchase_success_amt_24h  = purchase['success_amt_24h'],
        purchase_success_cnt_7d   = purchase['success_cnt_7d'],
        purchase_success_amt_7d   = purchase['success_amt_7d'],
        purchase_success_cnt_30d  = purchase['success_cnt_30d'],
        purchase_success_amt_30d  = purchase['success_amt_30d'],
        purchase_success_cnt_90d  = purchase['success_cnt_90d'],
        purchase_success_amt_90d  = purchase['success_amt_90d'],
    )
