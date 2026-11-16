import logging
from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.data_models import Features, Transaction

from .db import get_did_days_on_file, get_ip_country
from .request import (
    event_ts,
    get_age_account_gap,
    get_day_of_week,
    get_days_since_signup,
    get_days_since_signup_group,
    get_hour_of_day,
    get_hours_to_purchase,
    get_month_of_year,
)
from .velocity import (
    _device_history_from_db,
    _purchase_counts_from_db,
    update_velocity_counters,
)

__all__ = ["build_features", "update_velocity_counters", "event_ts"]

logger = logging.getLogger("fraud_engine")


def _parse_purchase_entries(all_entries, suc_entries, windows):
    """Compute purchase count/amount dict from 90d ZRANGE WITHSCORES results.

    all_entries / suc_entries: [(member_str, timestamp_float), ...]
      where member_str = "{uuid}:{purchase_value}".
    windows: {"5min": since_ts, ...} — entry is in a window if ts >= since_ts.
    """
    out = {}
    for w in windows:
        out[f"cnt_{w}"] = 0
        out[f"amt_{w}"] = 0.0
        out[f"success_cnt_{w}"] = 0
        out[f"success_amt_{w}"] = 0.0

    amounts_90d = []
    for member, ts in all_entries:
        amount = float(member.rsplit(":", 1)[1])
        amounts_90d.append(amount)
        for w_name, since in windows.items():
            if ts >= since:
                out[f"cnt_{w_name}"] += 1
                out[f"amt_{w_name}"] += amount

    for member, ts in suc_entries:
        amount = float(member.rsplit(":", 1)[1])
        for w_name, since in windows.items():
            if ts >= since:
                out[f"success_cnt_{w_name}"] += 1
                out[f"success_amt_{w_name}"] += amount

    if amounts_90d:
        amounts_90d.sort()
        n, mid = len(amounts_90d), len(amounts_90d) // 2
        out["median_amt_90d"] = amounts_90d[mid] if n % 2 else (amounts_90d[mid - 1] + amounts_90d[mid]) / 2
    else:
        out["median_amt_90d"] = 0.0

    return out


async def build_features(t: Transaction, db: AsyncSession, redis: Redis) -> Features:
    """Build the live feature row.

    Convention: the "distinct X" SET features (did_uid_*, uid_did_*, ip_uid_*) include the
    buyer (the current purchase's member); COUNT features (did_txn_cnt_*, purchase_*,
    country_txn_count_*, amount_vs_user_avg_30d) are strictly prior. Counters are written
    after the response, so this request is never in Redis yet.
    """
    # The clock for every window is the transaction's own time, not the server's.
    # Training counts back from purchase_time; see tests/test_feature_parity.py.
    ct = event_ts(t)
    ip_int = int(t.ip_address)

    # --- Phase 1: cache check for ip_country and did_days_on_file ---
    # Two GETs sent in one pipeline — avoids two separate round-trips.
    _ip_key  = f"cache:ip:{ip_int}"
    _did_key = f"cache:did:{t.device_id}:{t.user_id}"
    cache_pipe = redis.pipeline(transaction=False)
    cache_pipe.getex(_ip_key, ex=86400)   # refresh 24h TTL on every hit
    cache_pipe.getex(_did_key, ex=3600)   # refresh 1h TTL on every hit
    ip_cached, did_cached = await cache_pipe.execute()

    if ip_cached is not None:
        ip_country = ip_cached
    else:
        logger.warning("cache.miss", extra={"key": _ip_key, "fallback": "db"})
        ip_country = await get_ip_country(t.ip_address, db)
        await redis.set(_ip_key, ip_country, ex=86400)

    if did_cached is not None:
        did_days_on_file = int(did_cached)
    else:
        logger.warning("cache.miss", extra={"key": _did_key, "fallback": "db"})
        did_days_on_file = await get_did_days_on_file(
            t.device_id, t.user_id, db,
            as_of_date=datetime.fromtimestamp(ct, tz=timezone.utc).date())
        await redis.set(_did_key, str(did_days_on_file), ex=3600)

    # --- Phase 2: ALL 38 velocity commands in ONE pipeline ---
    # One TCP write → one TCP read → one event-loop yield for all commands.
    # Replaces asyncio.gather of 38 individual coroutines (which each acquired
    # their own connection and created separate event-loop wake-ups).
    ninety_days_ago = ct - 90 * 86400
    _5m, _10m = ct - 300, ct - 600
    _1h, _2h = ct - 3600, ct - 7200
    _24h, _7d = ct - 86400, ct - 604800
    _30d = ct - 2592000

    pipe = redis.pipeline(transaction=False)

    # [0-7]   did:uid — distinct users seen on this device (8 windows)
    for since in (_5m, _10m, _1h, _2h, _24h, _7d, _30d, ninety_days_ago):
        pipe.zcount(f"did:uid:{t.device_id}", since, ct)
    # [8-15]  uid:did — distinct devices seen for this user (8 windows)
    for since in (_5m, _10m, _1h, _2h, _24h, _7d, _30d, ninety_days_ago):
        pipe.zcount(f"uid:did:{t.user_id}", since, ct)
    # [16-21] ip:uid — distinct users seen from this IP (6 windows)
    for since in (_5m, _10m, _1h, _2h, _24h, _7d):
        pipe.zcount(f"ip:uid:{ip_int}", since, ct)
    # [22]    purchase:all 90d — (member="{uuid}:{amount}", score=timestamp)
    pipe.zrange(f"purchase:all:{t.user_id}", ninety_days_ago, ct, byscore=True, withscores=True)
    # [23]    purchase:success 90d
    pipe.zrange(f"purchase:success:{t.user_id}", ninety_days_ago, ct, byscore=True, withscores=True)
    # [24]    uid:last_txn — timestamp of user's last transaction
    pipe.get(f"uid:last_txn:{t.user_id}")
    # [25-31] did:txn — total transactions on this device (7 windows)
    for since in (_5m, _10m, _1h, _2h, _24h, _7d, _30d):
        pipe.zcount(f"did:txn:{t.device_id}", since, ct)
    # [32-34] country:txn — total transactions from this country (3 windows; the 5min/10min/1h
    #         windows were dropped 2026-11-09: no model input or rule used them)
    for since in (_24h, _7d, _30d):
        pipe.zcount(f"country:txn:{ip_country}", since, ct)
    # [35-37] last-seen time of THIS purchase's member in each set (or None). Used to add the
    #         buyer to the set features: sets include the current purchase, counts do not.
    pipe.zscore(f"did:uid:{t.device_id}", t.user_id)
    pipe.zscore(f"uid:did:{t.user_id}", t.device_id)
    pipe.zscore(f"ip:uid:{ip_int}", t.user_id)
    # [38]    did:txn_all — all-time transaction counter for this device (no window)
    pipe.get(f"did:txn_all:{t.device_id}")
    # [39-40] did:uids_all — every account ever seen on this device, and whether the buyer
    #         is already one of them
    pipe.scard(f"did:uids_all:{t.device_id}")
    pipe.sismember(f"did:uids_all:{t.device_id}", t.user_id)

    r = await pipe.execute()

    # --- Parse results by index ---
    _w8  = ("5min", "10min", "1h", "2h", "24h", "7d", "30d", "90d")
    _w6  = ("5min", "10min", "1h", "2h", "24h", "7d")
    _w7  = ("5min", "10min", "1h", "2h", "24h", "7d", "30d")
    _w3c = ("24h", "7d", "30d")

    did_uid      = dict(zip(_w8,  r[0:8]))
    uid_did      = dict(zip(_w8,  r[8:16]))
    ip_uid       = dict(zip(_w6,  r[16:22]))
    all_entries  = r[22]
    suc_entries  = r[23]
    last_txn_val = r[24]
    did_txn      = dict(zip(_w7,  r[25:32]))
    country_txn  = dict(zip(_w3c, r[32:35]))

    # Set features include the buyer: +1 when the current member is not already in the window
    # (same as training/build_features.py::_distinct_since_incl).
    def _incl(counts, member_ts, sinces):
        return {w: n + (0 if member_ts is not None and since <= float(member_ts) <= ct else 1)
                for (w, n), since in zip(counts.items(), sinces)}

    _s8 = (_5m, _10m, _1h, _2h, _24h, _7d, _30d, ninety_days_ago)
    did_uid = _incl(did_uid, r[35], _s8)
    uid_did = _incl(uid_did, r[36], _s8)
    ip_uid  = _incl(ip_uid,  r[37], _s8[:6])

    if all_entries:
        pur_windows = {
            "5min": _5m, "10min": _10m, "1h": _1h, "24h": _24h,
            "7d": _7d, "30d": _30d, "90d": ninety_days_ago,
        }
        purchase = _parse_purchase_entries(all_entries, suc_entries, pur_windows)
    else:
        purchase = await _purchase_counts_from_db(t.user_id, db, as_of=ct)

    last_txn_time = max(0.0, ct - float(last_txn_val)) if last_txn_val else 0.0

    # All-time device history: transactions ever (did_txn_cnt_all) and OTHER accounts ever
    # (did_other_uid_all, the buyer excluded: a loyal customer on their own device is 0).
    # Missing Redis history means a new device or lost keys; Postgres knows which. Its
    # answer is written back so the post-response update continues from the right state
    # and the next request for this device does not hit the database again. NX on the
    # counter: never overwrite one another request just created. A counter of 0 with no
    # accounts is a consistent "known new device", not lost history.
    if r[38] is not None and (r[39] > 0 or int(r[38]) == 0):
        did_txn_cnt_all = int(r[38])
        did_other_uid_all = int(r[39]) - int(bool(r[40]))
    else:
        history = await _device_history_from_db(t.device_id, db, as_of=ct)
        did_txn_cnt_all = int(r[38]) if r[38] is not None else sum(history.values())
        did_other_uid_all = len(set(history) - {t.user_id})
        await redis.set(f"did:txn_all:{t.device_id}", did_txn_cnt_all, nx=True)
        if history:
            await redis.sadd(f"did:uids_all:{t.device_id}", *history)

    # --- Derived features ---
    days_since_signup = get_days_since_signup(t)
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
        did_txn_cnt_all           = did_txn_cnt_all,
        did_other_uid_all         = did_other_uid_all,
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
        country_txn_count_24h      = country_txn['24h'],
        country_txn_count_7d       = country_txn['7d'],
        country_txn_count_30d      = country_txn['30d'],
    )
