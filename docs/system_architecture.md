# System Architecture

## Overview

Real-time fraud detection API for e-commerce transactions. Each incoming transaction
is scored by rule-based and ML-based checks, written to PostgreSQL, and returned
with a `fraud_decision` of `allow` or `deny` — all within a single async HTTP response cycle.

---

## High-Level Architecture

```
                          ┌─────────────────────────────────────┐
                          │            FastAPI (port 8000)       │
  Client ──POST /api/──►  │                                      │
  transactions            │  1. Idempotency check (Redis)        │
                          │  2. Build features                   │
                          │  3. Hard rules (short-circuit)       │
                          │  4. ML model score                   │
                          │  5. Soft rules + threshold           │
                          │  6. Write to DB                      │
                          │  7. Update velocity counters         │
                          └────────┬──────────────┬─────────────┘
                                   │              │
                          ┌────────▼───┐   ┌──────▼──────┐
                          │  Redis 7   │   │ PostgreSQL16 │
                          │            │   │              │
                          │ Sorted sets│   │  ip_country  │
                          │ Idempotency│   │  device_data │
                          │ keys       │   │  transactions│
                          └────────────┘   └─────────────┘
```

---

## Request Flow (Step by Step)

```
POST /api/transactions
        │
        ▼
1. Idempotency check
   redis.getex("idem:{key}", ex=600)
   ├── HIT  → return cached DecisionResponse immediately
   └── MISS → continue
        │
        ▼
2. build_features(transaction, db, redis)          [sequential awaits]
   ├── get_ip_country(ip_address, db)              → PostgreSQL GIST query
   ├── get_did_days_on_file(device_id, user_id, db)→ PostgreSQL upsert
   ├── get_did_uid_counts(device_id, redis, db)    → ZRANGEBYSCORE, 8 windows
   ├── get_uid_did_counts(user_id, redis, db)      → ZRANGEBYSCORE, 8 windows
   ├── get_ip_uid_counts(ip_address, redis, db)    → ZRANGEBYSCORE, 6 windows
   ├── get_purchase_counts(user_id, redis, db)     → ZRANGEBYSCORE, 7 windows × 2 sets
   ├── get_did_txn_counts(device_id, redis, db)    → ZRANGEBYSCORE, 7 windows
   ├── get_uid_last_txn_time(user_id, redis, db)   → last transaction timestamp
   └── get_country_txn_counts(ip_country, redis, db) → ZRANGEBYSCORE, 6 windows
        │
        ▼
3. Hard rules — short-circuit on any deny (no ML needed)
   execute_hard_rules(transaction, features)
   ├── transaction_limit()  → deny if purchase_value >= 5000
   ├── age_limit()          → deny if age invalid, or age >= 80 with new account
   └── ip_country_block()   → deny if ip_country != 'US'
        │
        ├── deny → skip to step 6
        └── allow → continue
        │
        ▼
4. ML model inference (non-blocking)
   loop.run_in_executor(executor, model_score, features)
   → RandomForest predict_proba → float (0.0–1.0)
        │
        ▼
5. Soft rules — all receive the model score
   execute_soft_rules(transaction, features, prob)
   ├── rate_limiter_control()      → deny on velocity anomalies
   └── model_score_threshold()     → deny if prob >= THRESHOLD (0.5)
        │
        ▼
6. Write to PostgreSQL
   INSERT INTO transactions (all DBWrite fields)
        │
        ▼
7. Cache idempotency key (atomic)
   redis.set("idem:{key}", response_json, nx=True, ex=600)
        │
        ▼
8. asyncio.create_task(update_velocity_counters(...))   [fire-and-forget]
   ├── ZADD uid:did:{user_id}          device_id  → timestamp
   ├── ZADD did:uid:{device_id}        user_id    → timestamp
   ├── ZADD ip:uid:{ip}                user_id    → timestamp
   ├── ZADD did:txn:{device_id}        txn_id     → timestamp
   ├── ZADD country:uid:{ip_country}   user_id    → timestamp
   ├── ZADD purchase:all:{user_id}     txn:amount → timestamp
   ├── ZADD purchase:success:{user_id} (if not deny)
   └── ZREMRANGEBYSCORE all sets      -inf → 90_days_ago
        │
        ▼
   Return DecisionResponse to client
```

---

## Component Map

```
app/
├── main.py                    Entry point, FastAPI app, lifespan, dependency injection
├── config.py                  Pydantic-settings: REDIS_HOST, DATABASE_URL, etc.
│
├── data/
│   ├── data_models.py         Pydantic model hierarchy (see below)
│   ├── orm_models.py          SQLAlchemy ORM — TransactionRecord (fully implemented)
│   ├── ip_country_db.py       [STUB] IP→country lookup (superseded by features/db.py)
│   ├── client_mimic.py        Load test client — 50 rps for 120s
│   └── db_init/
│       ├── 001_create_ip_country.sql   Creates ip_country table + GIST index
│       ├── 002_load_ip_country.sql     COPY from IpAddress_to_Country.csv
│       └── 003_create_device_data.sql  Creates device_data table
│
└── decisions/
    ├── __init__.py            execute_hard_rules() / execute_soft_rules()
    ├── rules.py               Rule functions: transaction_limit, age_limit,
    │                          ip_country_block, rate_limiter_control, model_score_threshold
    ├── models.py              ML inference — loads model.joblib, model_features.json,
    │                          ip_country_categories.json at import; exports score() + predict()
    └── features/
        ├── __init__.py        build_features() — orchestrates all feature functions
        ├── request.py         get_days_since_signup(), get_days_since_signup_group(),
        │                      get_age_account_gap(), get_hours_to_purchase(),
        │                      get_hour_of_day(), get_day_of_week(), get_month_of_year()
        ├── db.py              get_ip_country(), get_did_days_on_file()
        └── velocity.py        get_did_uid_counts(), get_uid_did_counts(),
                               get_ip_uid_counts(), get_purchase_counts(),
                               get_did_txn_counts(), get_uid_last_txn_time(),
                               get_country_txn_counts(), update_velocity_counters()
```

---

## Pydantic Model Hierarchy

```
Transaction                         ← raw client input
  ├── user_id, purchase_time, purchase_value
  ├── device_id, ip_address, age, sex, browser, source
  ├── signup_time (optional)
  └── fraud_class (alias: 'class')

  ├── DecisionResponse              ← returned to client (lean)
  │     ├── transaction_id (UUID)
  │     ├── fraud_decision (str)
  │     └── response_time (datetime)
  │
  ├── Features(Transaction)         ← + all computed features
  │     ├── days_since_signup, days_since_signup_group, ip_country
  │     ├── did_uid_days_on_file
  │     ├── did_uid_{5min,10min,1h,2h,24h,7d,30d,90d}    (UIDs per device)
  │     ├── uid_did_{5min,10min,1h,2h,24h,7d,30d,90d}    (DIDs per user)
  │     ├── ip_uid_{5min,10min,1h,2h,24h,7d}             (UIDs per IP)
  │     ├── purchase_{cnt,amt}_{5min,10min,1h,24h,7d,30d,90d}
  │     ├── purchase_success_{cnt,amt}_{5min,10min,1h,24h,7d,30d,90d}
  │     ├── did_txn_cnt_{5min,10min,1h,2h,24h,7d,30d}   (transactions per device)
  │     ├── country_txn_count_{5min,10min,1h,24h,7d,30d} (transactions per country)
  │     ├── pur_median_amt_90d, amount_vs_user_avg_30d   (derived from purchase history)
  │     ├── age_account_gap, high_amount_vs_hist         (derived)
  │     ├── time_since_last_txn_uid                      (seconds since last user txn)
  │     └── hours_to_purchase, hour_of_day, day_of_week, month_of_year  (temporal)
  │
  └── DBWrite(Features, DecisionResponse)  ← written to PostgreSQL
        └── + created_at
```

---

## PostgreSQL Schema

### `ip_country`
| Column     | Type      | Notes                                  |
|------------|-----------|----------------------------------------|
| id         | bigserial | Primary key                            |
| ip_from    | bigint    | Range start                            |
| ip_to      | bigint    | Range end                              |
| country    | text      |                                        |
| ip_range   | int8range | Generated from ip_from/ip_to; STORED   |

Index: `GIST (ip_range)` — enables `WHERE ip_range @> :ip` containment queries.
Loaded from `IpAddress_to_Country.csv` (~5M rows) on first container start.

### `device_data`
| Column       | Type | Notes                           |
|--------------|------|---------------------------------|
| device_id    | text | Composite primary key           |
| user_id      | text | Composite primary key           |
| did_uid_date | date | First date this pair was seen   |

Used by `get_did_days_on_file()` via upsert: inserts on first visit,
returns days elapsed on subsequent visits.

### `transactions`
All fields from `DBWrite`: full transaction input + all computed features
+ `transaction_id` (UUID PK) + `fraud_decision` + `response_time` + `created_at`.
ORM class: `TransactionRecord` in `app/data/orm_models.py`.

---

## Redis Data Structures

All velocity data uses **sorted sets** (`ZADD`/`ZRANGEBYSCORE`).
Score = Unix timestamp. Members = entity IDs or encoded strings.
Entries older than 90 days are pruned on every write (`ZREMRANGEBYSCORE`).

### Velocity sorted sets

| Key pattern                  | Member         | Score      | Query for                        |
|------------------------------|----------------|------------|----------------------------------|
| `uid:did:{user_id}`          | device_id      | timestamp  | distinct devices per user        |
| `did:uid:{device_id}`        | user_id        | timestamp  | distinct users per device        |
| `ip:uid:{ip}`                | user_id        | timestamp  | distinct users per IP            |
| `did:txn:{device_id}`        | txn_id         | timestamp  | transaction count per device     |
| `country:uid:{ip_country}`   | user_id        | timestamp  | transaction count per country    |
| `purchase:all:{user_id}`     | `txn_id:amt`   | timestamp  | all purchases — count + sum      |
| `purchase:success:{user_id}` | `txn_id:amt`   | timestamp  | approved purchases only          |

**Key convention:** `{what-you-filter-by}:{what-you-count}:{entity-id}`

### Idempotency keys

| Key pattern         | Value              | TTL       |
|---------------------|--------------------|-----------|
| `idem:{key}`        | DecisionResponse JSON | 10 min |

Set atomically with `SET NX EX 600`.

---

## Feature Sources Summary

| Feature group              | Source              | Function                      |
|----------------------------|---------------------|-------------------------------|
| `days_since_signup`        | Transaction fields  | `get_days_since_signup()`     |
| `days_since_signup_group`  | Transaction fields  | `get_days_since_signup_group()`|
| `age_account_gap`          | Transaction fields  | `get_age_account_gap()`       |
| `hours_to_purchase`        | Transaction fields  | `get_hours_to_purchase()`     |
| `hour_of_day`              | Transaction fields  | `get_hour_of_day()`           |
| `day_of_week`              | Transaction fields  | `get_day_of_week()`           |
| `month_of_year`            | Transaction fields  | `get_month_of_year()`         |
| `ip_country`               | PostgreSQL          | `get_ip_country()`            |
| `did_uid_days_on_file`     | PostgreSQL          | `get_did_days_on_file()`      |
| `did_uid_*`                | Redis sorted set    | `get_did_uid_counts()`        |
| `uid_did_*`                | Redis sorted set    | `get_uid_did_counts()`        |
| `ip_uid_*`                 | Redis sorted set    | `get_ip_uid_counts()`         |
| `purchase_*`               | Redis sorted set    | `get_purchase_counts()`       |
| `did_txn_cnt_*`            | Redis sorted set    | `get_did_txn_counts()`        |
| `country_txn_count_*`      | Redis sorted set    | `get_country_txn_counts()`    |
| `time_since_last_txn_uid`  | Redis sorted set    | `get_uid_last_txn_time()`     |
| `amount_vs_user_avg_30d`   | Derived (in-memory) | `build_features()` inline     |
| `pur_median_amt_90d`       | Derived (in-memory) | `build_features()` inline     |
| `high_amount_vs_hist`      | Derived (in-memory) | `build_features()` inline     |

---

## Infrastructure

| Service    | Image              | Port  | Purpose                          |
|------------|--------------------|-------|----------------------------------|
| api        | (built from ./app) | 8000  | FastAPI application               |
| db         | postgres:16        | 5432  | Feature store + transactions      |
| redis      | redis:7-alpine     | 6379  | Velocity counters + idempotency   |

**ThreadPoolExecutor** (in-process, not a Docker service): managed via `lifespan`,
stored in `app.state.executor`. Used to run the ML model (`model_score`) without
blocking the async event loop — ML inference is CPU-bound and not awaitable.

Services defined in `docker-compose.yaml`. The `api` service waits for
`db` and `redis` health checks before starting.

---

## Known Limitations

See `design_tradeoffs_and_todos.md` for full detail. Key open items:

- Velocity counters are fire-and-forget — stale features if the server crashes mid-request
- `build_features` runs IO calls sequentially — could be parallelized with `asyncio.gather`
  for lower latency under load
