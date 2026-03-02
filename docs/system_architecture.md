# System Architecture

## Overview

Real-time fraud detection API for e-commerce transactions. Each incoming transaction
is scored by rule-based and ML-based checks, written to PostgreSQL, and returned
with a `fraud_decision` of `allow` or `deny` — all within a single synchronous
HTTP response cycle.

---

## High-Level Architecture

```
                          ┌─────────────────────────────────────┐
                          │            FastAPI (port 8000)       │
  Client ──POST /api/──►  │                                      │
  transactions            │  1. Idempotency check (Redis)        │
                          │  2. Build features                   │
                          │  3. Rule-based decision              │
                          │  4. ML model decision                │
                          │  5. Write to DB                      │
                          │  6. Update velocity counters         │
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
   redis.get("idem:{key}")
   ├── HIT  → return cached Response immediately
   └── MISS → continue
        │
        ▼
2. build_features(transaction, db, redis)          [asyncio.gather — all parallel]
   ├── get_ip_country(ip_address, db)              → PostgreSQL GIST query
   ├── get_did_days_on_file(device_id, user_id, db)→ PostgreSQL upsert
   ├── get_did_uid_counts(device_id, redis)        → 5 ZRANGEBYSCORE calls
   ├── get_uid_did_counts(user_id, redis)          → 5 ZRANGEBYSCORE calls
   ├── get_ip_uid_counts(ip_address, redis)        → 3 ZRANGEBYSCORE calls
   └── get_purchase_counts(user_id, redis)         → 8 ZRANGEBYSCORE calls
        │
        ▼
3. Rule-based decision
   transaction_limit(transaction)  → deny if purchase_value >= 5000
        │
        ▼
4. ML model decision
   predict(features)               → allow / deny / review
        │
        ▼
5. Write to PostgreSQL
   INSERT INTO transactions (DBWrite fields)
        │
        ▼
6. Cache idempotency key
   redis.setex("idem:{key}", 86400, response_json)
        │
        ▼
7. asyncio.create_task(update_velocity_counters(...))   [fire-and-forget]
   ├── ZADD uid:did:{user_id}        device_id  → timestamp
   ├── ZADD did:uid:{device_id}      user_id    → timestamp
   ├── ZADD ip:uid:{ip}              user_id    → timestamp
   ├── ZADD purchase:all:{user_id}   txn:amount → timestamp
   ├── ZADD purchase:success:{user_id} (if not deny)
   └── ZREMRANGEBYSCORE all sets    -inf → 90_days_ago
        │
        ▼
   Return Response to client
```

---

## Component Map

```
app/
├── main.py                    Entry point, FastAPI app, dependency injection
├── config.py                  Pydantic-settings: REDIS_HOST, DATABASE_URL, etc.
│
├── data/
│   ├── data_models.py         Pydantic model hierarchy (see below)
│   ├── orm_models.py          [TODO] SQLAlchemy ORM — TransactionRecord
│   ├── ip_country_db.py       [STUB] IP→country lookup (to be implemented)
│   ├── client_mimic.py        Load test client — 50 rps for 120s
│   └── db_init/
│       ├── 001_create_ip_country.sql   Creates ip_country table + GIST index
│       ├── 002_load_ip_country.sql     COPY from IpAddress_to_Country.csv
│       └── 003_create_device_data.sql  Creates device_data table
│
└── decisions/
    ├── rules.py               Rule-based checks (purchase_value threshold)
    ├── models.py              [STUB] ML model inference
    └── features/
        ├── __init__.py        build_features() — orchestrates all feature functions
        ├── request.py         get_days_since_signup(), get_age_group()
        ├── db.py              get_ip_country(), get_did_days_on_file()
        └── velocity.py        get_did_uid_counts(), get_uid_did_counts(),
                               get_ip_uid_counts(), get_purchase_counts(),
                               update_velocity_counters()
```

---

## Pydantic Model Hierarchy

```
Transaction                         ← raw client input
  ├── user_id, purchase_time, purchase_value
  ├── device_id, ip_address, age, sex, browser, source
  ├── signup_time (optional)
  └── fraud_class (alias: 'class')

  ├── Response(Transaction)         ← returned to client  [TODO: make lean]
  │     └── + transaction_id, fraud_decision
  │
  ├── Features(Transaction)         ← + 34 computed features
  │     ├── days_since_signup, ip_country, age_group
  │     ├── did_uid_days_on_file
  │     ├── did_uid_{2h,24h,7d,30d,90d}       (UIDs per device)
  │     ├── uid_did_{2h,24h,7d,30d,90d}       (DIDs per user)
  │     ├── ip_uid_{2h,24h,7d}                (UIDs per IP)
  │     ├── purchase_{cnt,amt}_{24h,7d,30d,90d}
  │     └── purchase_success_{cnt,amt}_{24h,7d,30d,90d}
  │
  └── DBWrite(Transaction, Features)  ← written to PostgreSQL
        └── + transaction_id, fraud_decision, created_at
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

### `transactions` *(planned — requires orm_models.py)*
All fields from `DBWrite`: full transaction input + all 34 features
+ `transaction_id` (PK) + `fraud_decision` + `created_at`.

---

## Redis Data Structures

All velocity data uses **sorted sets** (`ZADD`/`ZRANGEBYSCORE`).
Score = Unix timestamp. Members = entity IDs or encoded strings.
Entries older than 90 days are pruned on every write (`ZREMRANGEBYSCORE`).

### Velocity sorted sets

| Key pattern              | Member       | Score      | Query for                        |
|--------------------------|--------------|------------|----------------------------------|
| `uid:did:{user_id}`      | device_id    | timestamp  | distinct devices per user        |
| `did:uid:{device_id}`    | user_id      | timestamp  | distinct users per device        |
| `ip:uid:{ip}`            | user_id      | timestamp  | distinct users per IP            |
| `purchase:all:{user_id}` | `txn_id:amt` | timestamp  | all purchases — count + sum      |
| `purchase:success:{user_id}` | `txn_id:amt` | timestamp | approved purchases only    |

**Key convention:** `{what-you-filter-by}:{what-you-count}:{entity-id}`

### Idempotency keys

| Key pattern         | Value         | TTL    |
|---------------------|---------------|--------|
| `idem:{key}`        | Response JSON | 24 hrs |

---

## Feature Sources Summary

| Feature group         | Source          | Function                    |
|-----------------------|-----------------|-----------------------------|
| `days_since_signup`   | Transaction fields | `get_days_since_signup()`  |
| `age_group`           | Transaction fields | `get_age_group()`          |
| `ip_country`          | PostgreSQL        | `get_ip_country()`         |
| `did_uid_days_on_file`| PostgreSQL        | `get_did_days_on_file()`   |
| `did_uid_*`           | Redis sorted set  | `get_did_uid_counts()`     |
| `uid_did_*`           | Redis sorted set  | `get_uid_did_counts()`     |
| `ip_uid_*`            | Redis sorted set  | `get_ip_uid_counts()`      |
| `purchase_*`          | Redis sorted set  | `get_purchase_counts()`    |

---

## Infrastructure

| Service    | Image              | Port  | Purpose                     |
|------------|--------------------|-------|-----------------------------|
| api        | (built from ./app) | 8000  | FastAPI application          |
| db         | postgres:16        | 5432  | Feature store + transactions |
| redis      | redis:7-alpine     | 6379  | Velocity counters + idempotency |

Services defined in `docker-compose.yaml`. The `api` service waits for
`db` and `redis` health checks before starting.

---

## Known Limitations

See `design_tradeoffs_and_todos.md` for full detail. Key open items:

- `main.py` uses sync `Session` instead of `AsyncSession` — blocks event loop
- `get_redis()` opens a new TCP connection per request — should use `lifespan`
- Idempotency check has a TOCTOU race — fix with atomic `SET NX EX`
- `db.add(response)` passes Pydantic model to SQLAlchemy — needs ORM instance
- `Response` echoes all input fields back to client — should be lean
- Velocity counters are fire-and-forget — stale on server crash
