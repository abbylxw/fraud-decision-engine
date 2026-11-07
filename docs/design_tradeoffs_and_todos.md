# Design Trade-offs & To-Do Items

## Major Trade-offs / Problem Statements

---

### 1. Sync vs Async Database Driver

**Problem:** FastAPI runs on a single-threaded async event loop. Synchronous DB calls
(psycopg2 / SQLAlchemy `Session`) block the event loop — no other requests can be
handled while waiting for the DB.

**Trade-off:** Async adds complexity (different driver, `await` everywhere, harder
debugging) but is necessary for high concurrency. Sync is simpler but serializes
all requests at the DB call.

**Solution keywords:** `asyncpg`, `AsyncSession`, `async_sessionmaker`,
`create_async_engine`, `yield` dependency, connection pool sizing

**Status: ✅ SOLVED** — `create_async_engine` + `AsyncSession` in `main.py`

---

### 2. Redis Connection Per Request

**Problem:** Creating a new `Redis(host=..., port=...)` object inside `get_redis()`
on every request opens a new TCP connection each time — expensive handshake overhead,
port exhaustion under load.

**Trade-off:** Shared client is efficient but requires careful lifecycle management
(startup / shutdown).

**Solution keywords:** `lifespan`, `app.state`, connection pool, `aclose()` on shutdown,
`asynccontextmanager`

**Status: ✅ SOLVED** — `lifespan` creates one shared client on `app.state.redis`

---

### 3. Pydantic Model vs ORM Model for DB Writes

**Problem:** `db.add(response)` fails because SQLAlchemy's `add()` requires a mapped
ORM instance, not a Pydantic model. They serve different purposes: Pydantic validates
data shapes; SQLAlchemy maps Python objects to table rows.

**Trade-off:** Maintaining two parallel class hierarchies (Pydantic + ORM) adds
boilerplate but separates API contract from persistence schema cleanly.

**Solution keywords:** `DeclarativeBase`, `Mapped`, `mapped_column`, `__tablename__`,
`model_dump()` → ORM constructor, `DBWrite` as source of truth for columns

**Status: ✅ SOLVED** — `TransactionRecord` ORM in `orm_models.py`; `DBWrite.model_dump()` feeds it

---

### 4. CPU-bound Model Inference Blocking Event Loop

**Problem:** `model.predict()` is CPU-bound (no I/O waits). Running it directly in
an `async def` function blocks the event loop — all other incoming requests queue
behind it.

**Trade-off:** Thread pool adds concurrency but Python GIL limits true parallelism for
pure-Python models. Process pool bypasses GIL but has higher IPC overhead.
Most ML libraries (XGBoost, scikit-learn, PyTorch) release the GIL so threads work.

**Solution keywords:** `ThreadPoolExecutor`, `ProcessPoolExecutor`, `run_in_executor`,
`get_running_loop`, GIL, `max_workers = n_cpu_cores`, `Depends(get_executor)`

**Status: ✅ SOLVED** — `ThreadPoolExecutor` stored on `app.state.executor` in `lifespan`;
`get_executor` dependency injected into endpoint; model called via
`await loop.run_in_executor(executor, model_score, features)`

---

### 5. Idempotency Race Condition (TOCTOU)

**Problem:** Current pattern — GET key → if missing → process → SET key — has a
time-of-check-time-of-use gap. Two concurrent requests with the same idempotency key
can both pass the `if cached` check and both process the transaction.

**Trade-off:** Atomic claim with `SET NX` prevents double processing but requires a
polling/retry loop for the losing request, adding latency and complexity.

**Solution keywords:** `SET NX EX` (atomic), placeholder sentinel value, polling loop,
`asyncio.sleep`, `HTTPException 503` on timeout, TOCTOU

**Status: ✅ SOLVED (2026-11-05)** — the earlier "fix" still did GET → process → `SET NX`, so two
concurrent duplicates both ran the pipeline. Now the key is claimed with
`SET idem:<key> __pending__ NX EX 30` before processing; a loser sees `__pending__` and gets
`409` + `Retry-After: 1` (no polling, so no held connections), or the stored response if the
original finished (`EX 60`). A failed request deletes the key so the client can retry.
Remaining gap: if the worker dies after the DB commit but before the response is stored, a
retry after the 30 s lock expires is processed again; closing that needs a unique
`idempotency_key` column in Postgres. Tests: `tests/test_idempotency.py`.

---

### 6. Redis Feature Cold Start / Cache Miss

**Problem:** On cache miss, silently defaulting to `0` produces incorrect feature
values for returning users (expired TTL) — the model sees wrong inputs without any
error signal.

**Trade-off:** Read-through fallback (query DB on miss) is correct but adds DB latency
to the hot path for cold requests. Warming on signup prevents first-transaction misses
but doesn't cover TTL expiry.

**Solution keywords:** read-through cache, `GETEX` (atomic get + refresh TTL),
`zremrangebyscore`, warm-up on signup, TTL jitter, cache miss logging,
thundering herd

**Status: ✅ SOLVED** — Every `get_*` function in `velocity.py` checks `redis.exists(key)`
first; on miss calls a private `_*_from_db` helper that reconstructs counts from
the `transactions` table using a single `COUNT(CASE WHEN ...)` SQL query per function.
Remaining gaps: `GETEX` TTL refresh on hits; cache miss warning logs.

---

### 7. Windowed Aggregate Accuracy

**Problem:** Simple `INCR` keys for velocity counters (purchase_cnt_24h etc.) reset
hard at key expiry — a user's 24h count drops to 0 the moment the TTL expires, even
if they had 50 transactions 2 hours ago.

**Trade-off:** Sorted sets (ZADD with timestamp scores) are self-maintaining and always
accurate but use more memory and require pruning logic. Simple counters are cheap but
wrong at window boundaries.

**Solution keywords:** `ZADD`, `ZCOUNT`, `ZRANGEBYSCORE`, `ZREMRANGEBYSCORE`,
sliding window, score = unix timestamp, TTL on the key itself, `ZCARD`

**Status: ✅ SOLVED** — All counters use Redis sorted sets with unix timestamp scores;
`ZREMRANGEBYSCORE` prunes entries older than 90 days on every write

---

### 8. Velocity Counter Update Timing

**Problem:** Velocity counters must be updated after every transaction so the next
request sees current counts. If updates happen synchronously before returning the
response, they add latency. If fire-and-forget, a crash between response and update
leaves counters stale.

**Trade-off:** Synchronous update is accurate but slower. Fire-and-forget is fast but
risks under-counting. Message queue (Kafka) is durable and decoupled but adds infra.

**Solution keywords:** `asyncio.create_task` (fire-and-forget), outbox pattern,
Kafka consumer, at-least-once, idempotent ZADD, post-response hook

**Status: ✅ SOLVED** — `update_velocity_counters` runs as a tracked background task after
the DB commit (a failed commit no longer updates counters). `spawn_background()` keeps a strong
reference (asyncio only holds weak ones), logs + counts failures
(`fraud_background_task_failures_total`), and shutdown waits up to 5 s for in-flight tasks.
ZADD is idempotent so duplicate writes are safe. A crash still loses in-flight updates
(outbox/Kafka would fix that).

---

### 9. Response Model Carries Too Much Data

**Problem:** `Response` currently extends `Transaction`, so it returns all raw input
fields back to the client (signup_time, ip_address, age, etc.). The client sent these
— echoing them back wastes bandwidth and leaks internal data.

**Trade-off:** Lean response (just transaction_id + fraud_decision) is clean but
requires separating the API contract from the persistence model explicitly.

**Solution keywords:** lean response model, API contract vs persistence schema,
`model_dump(include={...})`, separate `DBWrite` from `Response`

**Status: ✅ SOLVED** — `DecisionResponse` is a standalone `BaseModel` with only
`transaction_id`, `fraud_decision`, `response_time`; `DBWrite` handles persistence

---

### 10. Rules Blind to Model Score

**Problem:** Rules ran before the model, so they could only use raw transaction fields
and velocity counts — not the model's fraud probability. This made it impossible to
write rules like "deny if score > 0.7 AND velocity is elevated".

**Trade-off:** Running the model before all rules means it always runs even for
eventually-denied transactions. The two-phase design (hard rules → model → soft rules)
avoids this: hard rules short-circuit clear-denials cheaply; model only runs for
borderline cases; soft rules receive the score.

**Solution keywords:** two-phase pipeline, hard vs soft rules, `score() -> float`,
`execute_hard_rules`, `execute_soft_rules`, separation of concerns

**Status: ✅ SOLVED** — `models.py` exposes `score(features) -> float` alongside `predict()`;
`decisions/__init__.py` defines `execute_hard_rules` (amount/age/country) and
`execute_soft_rules(score)` (velocity + model threshold); `main.py` runs them in order

---

### 11. Training-Serving Encoding Skew

**Problem:** The model only sees integers, and strings are turned into integers in two
places: pandas in the training notebook, and plain dicts in `app/decisions/models.py`.
Training encoded `sex` as `M=0, F=1`; serving used `F=0, M=1`. Every live request fed
the model the opposite value — no error, just silently worse decisions.

**Trade-off:** Two independent copies of the encoding are easy to write but drift apart.
One shared definition means training code imports from the serving package.

**Solution keywords:** training-serving skew, single source of truth, contract test

**Status: ✅ SOLVED (2026-11-04)** — `_SEX_CODES` fixed to `{"M": 0, "F": 1}`;
`training/build_features.py` imports the serving encodings instead of redefining them;
`tests/test_encoding_contract.py` fails if they ever disagree with how `model.joblib`
was trained.

---

### 12. Velocity Clock: Event Time vs Server Time

**Problem:** Every velocity feature ("distinct users on this device in the last 90 days")
needs a "now". Training counts back from each transaction's `purchase_time`. Serving
counted back from the server clock (`time.time()`, SQL `NOW()`, `CURRENT_DATE`). When
`client_mimic.py` replays a year of 2015 data in minutes, every window held "everything
replayed recently", so `did_uid_90d` (the model's top feature) meant something different
live than in training. A second, hidden trap: the client sends naive datetimes, and
Python's `.timestamp()` reads those as the server's *local* time while pandas reads them
as UTC, which shifts every hour-based window by the timezone offset.

**Trade-off:**
- **Event time** (chosen): serving matches training exactly, and replays are repeatable.
  The risk is that a client controls `purchase_time`, so an attacker could send an old
  timestamp to land outside the windows that would flag them.
- **Server time:** safe from spoofed timestamps, but then training must be rebuilt on
  ingestion time, and a historical replay can never reproduce training features.

In real production the two are nearly the same (a purchase is scored seconds after it
happens). The production mitigation is to reject, or clamp to the server clock, any
`purchase_time` more than a few minutes away from server time. That isn't enabled here,
because the load test deliberately replays 2015 data.

**Solution keywords:** event time vs processing time, training-serving skew, parity test,
naive datetime = UTC, exclusive purge bound

**Status: ✅ SOLVED (2026-11-05)**
- `request.event_ts(t)` is the single clock (naive datetimes are read as UTC).
- `build_features`, `update_velocity_counters` and both SQL fallbacks use it instead of
  the server clock.
- `tests/test_feature_parity.py` replays the same transactions through the real serving
  path (fake Redis) and through `training/build_features.py`, and requires every numeric
  feature to match. It runs on synthetic data in CI, and on 5,000 real rows locally.
- Written before the fix, it failed on ~50 features.
- It also caught two smaller bugs:
  - `did:txn` entries keyed by `user:time` collapsed two purchases in the same second;
    entries now use the transaction id.
  - The 90-day purge used an inclusive bound, which deleted an entry exactly 90 days
    old that a same-second transaction still needed; the bound is now exclusive.

---

## To-Do Items (Current Codebase)

### Minimum to get a test request through (do these first)

- [x] **`main.py`** — `engine` defined via `create_async_engine(settings.database_url)` at module level
- [x] **`features/velocity.py`, `features/db.py`, `features/request.py`** — all stubs replaced
      with real implementations; no longer returns `Ellipsis`
- [x] **`main.py`** — `db.add()` now receives a `TransactionRecord` ORM instance built from `DBWrite.model_dump()`

---

### `app/main.py`

- [x] Replace `Session` / `SessionLocal` with `AsyncSession` + `create_async_engine`
- [x] Move Redis into `lifespan` via `asynccontextmanager`; stored on `app.state.redis`; closed on shutdown
- [x] Add `get_executor` dependency backed by `ThreadPoolExecutor` in `lifespan`; stored on `app.state.executor`
- [x] Fix idempotency: reserve the key with `SET NX` before processing; 409 for in-flight duplicates (2026-11-05)
- [x] Wire full rule + model pipeline: `execute_hard_rules` → `model_score` in executor → `execute_soft_rules`
- [x] Call `update_velocity_counters()` via `asyncio.create_task` after response built
- [x] Fix `db.add()` — now passes `TransactionRecord(**db_record.model_dump())` ORM instance

### `app/data/data_models.py`

- [x] Made `DecisionResponse` lean — now a standalone `BaseModel` with only
      `transaction_id`, `fraud_decision`, `response_time`
- [x] Add default values to all `Features` fields; add missing 7d/90d velocity and
      purchase fields to match `build_features` output

### `app/data/orm_models.py` _(new file)_

- [x] `TransactionRecord(Base)` defined with all `DBWrite` fields as `Mapped` columns
- [x] `__tablename__ = "transactions"`, primary key on `transaction_id` (`SAUUID(as_uuid=True)`)

### `app/data/db_init/` _(new files)_

- [x] `003_create_device_data.sql` — `device_data` table with `PRIMARY KEY (device_id, user_id)`;
      mounted in `docker-compose.yaml`
- [x] `004_create_transactions.sql` — `transactions` table mirroring all `DBWrite` fields

### `app/decisions/features/` _(package)_

- [x] `request.py` — `get_days_since_signup`, `get_days_since_signup_group`, temporal features implemented
- [x] `db.py` — `get_ip_country` (queries `ip_country` GIST index) and `get_did_days_on_file`
      (upsert into `device_data`) implemented
- [x] `velocity.py` — all `get_*_counts` functions implemented using sorted sets
      (`ZRANGEBYSCORE` with per-entity keys, `asyncio.gather` for parallel window queries)
- [x] `velocity.py` — `update_velocity_counters()` implemented with `asyncio.gather` + `ZADD`;
      conditional write to `purchase:success:{user_id}` on non-deny decisions;
      write-time pruning via `ZREMRANGEBYSCORE` (90-day cutoff)
- [x] `velocity.py` — read-through DB fallback added to every `get_*` function;
      `redis.exists(key)` used for miss detection; single `COUNT(CASE WHEN ...)` SQL
      query per function reconstructs all time-window counts in one round-trip
- [x] Add `GETEX` TTL refresh on every Redis cache hit (keep hot keys alive longer)
- [x] Add cache miss warning log to every DB fallback path in `velocity.py`

### `app/decisions/rules.py`

- [x] All rules use Pydantic attribute access (`transaction.purchase_value`, not dict access)
- [x] Rules expanded: `transaction_limit`, `age_limit`, `ip_country_block`, `rate_limiter_control`,
      `model_score_threshold` — split into hard and soft rule sets
- [x] Expand `rate_limiter_control` to use `score` parameter for combined velocity+model conditions
- [ ] Add more rules (country block list, device-age heuristics, etc.)

### `app/decisions/models.py`

- [x] `score(features) -> float` — returns raw fraud probability; used by `main.py` + `model_score_threshold`
- [x] `predict(features) -> str` — delegates to `score()` + `THRESHOLD` comparison
- [x] Model and feature list loaded at import time (not per request)

### `app/decisions/__init__.py`

- [x] `execute_hard_rules(t, f)` — chains transaction_limit, age_limit, ip_country_block;
      returns `(decision, rule_name | None)` so the triggering rule can be logged
- [x] `execute_soft_rules(t, f, score)` — chains rate_limiter_control + model_score_threshold;
      returns `(decision, rule_name | None)`; all rules receive the model score

---

---

## Production Hardening (2025-03-14)

Items added to make the project deployment-ready and portfolio-presentable.

### New Files

- [x] **`README.md`** — project overview, architecture diagram, quick start, example request/response,
      engineering decisions section
- [x] **`pyproject.toml`** — ruff linting config (`select = ["E","F","W","I"]`, `line-length = 100`);
      pytest config (`asyncio_mode = "auto"`, `testpaths = ["tests"]`)
- [x] **`.github/workflows/ci.yml`** — GitHub Actions CI: lint → test → docker build on every push

### `app/main.py`

- [x] Add `GET /health` endpoint — checks `db.execute(SELECT 1)` and `redis.ping()`;
      returns `{"status": "ok"|"degraded", "db": "...", "redis": "..."}`;
      standard readiness probe used by Kubernetes and load balancers
- [x] Add structured logging via `_JSONFormatter` (stdlib `logging`; no new dependency);
      emits one JSON object per line; 5 log events:
  - `request.received` — user_id, purchase_value, device_id
  - `hard_rule.deny` — user_id, rule name, ip_country, did_uid_days_on_file, purchase_value, age
  - `model.scored` — user_id, prob (4dp), decision
  - `decision.written` — transaction_id, fraud_decision
  - `idempotency.hit` — idempotency_key
- [x] Fix `redis.close()` → `redis.aclose()` (deprecated since redis-py 5.0.1)

### `tests/` _(new directory)_

69 tests across 4 files; all pass without live DB or Redis.

- [x] `tests/conftest.py` — `sample_transaction` and `sample_features` shared fixtures
- [x] `tests/test_rules.py` — 24 tests covering all 5 rule functions; pure unit tests
      (no mocking, just call rule functions with crafted inputs)
- [x] `tests/test_features.py` — 20 tests for pure request-based feature functions
      (`days_since_signup`, `days_since_signup_group`, `hours_to_purchase`, temporal)
- [x] `tests/test_api.py` — 6 integration tests for `/health` and `POST /api/transactions`;
      DB and Redis replaced with `AsyncMock` via FastAPI `dependency_overrides`;
      covers: healthy, DB-down, Redis-down, hard-rule deny, allow, idempotency hit
- [x] `tests/test_velocity_fallback.py` — 11 tests for Redis→DB fallback in velocity functions;
      verifies that on `redis.exists()` → 0, the function calls `db.execute()` and returns
      DB values; also verifies cache hits skip the DB entirely

### `requirements.txt` _(replaced by `pyproject.toml` `[dependency-groups]`, 2026-11-06)_

- [x] Add `pytest`, `pytest-asyncio`, `ruff` (now the `dev` group)

---

## Trade-off: asyncio.gather() in build_features() — Why It Was Removed

**Decision:** `build_features()` uses sequential `await` calls, not `asyncio.gather()`.

**Why `asyncio.gather` was tried and removed (see Fix 2 below):**

`asyncio.gather()` runs coroutines concurrently. All feature functions share the same
`AsyncSession`, which wraps a single asyncpg connection. A single connection handles one
query at a time — concurrent `await db.execute()` calls on the same session raise:
`This session is provisioning a new connection; concurrent operations are not permitted`.

**Why not use separate sessions per call?** Each session uses one connection from the pool.
8 concurrent feature functions = 8 connections per request. Under 50 RPS load that's
400 simultaneous connections — well above a typical pool size.

**Why the performance impact is minimal in practice:**
- Redis is in-memory; each call completes in <1ms. Sequential is fast.
- `asyncio.gather` IS used inside each velocity function for the multiple `ZRANGEBYSCORE`
  calls to Redis — that's where it actually helps (fan-out to one server per function).
- On a DB fallback (cache miss), sequential is also correct because the session is shared.

**Conclusion:** Sequential `build_features()` + parallel Redis calls inside each function
is the correct and intentional design. Do not add `gather` at the `build_features()` level
without refactoring to use per-call sessions.

---

## Trade-off: structlog vs stdlib logging

**Decision:** Used Python's stdlib `logging` with a custom `_JSONFormatter`.

**Trade-off:** `structlog` provides richer context binding (attach `transaction_id` once,
have it appear on every subsequent log line) but adds a dependency. Stdlib logging with
`extra={}` achieves structured output with zero new dependencies. For this project's
log volume and team size, stdlib is sufficient.

**Revisit if:** log correlation across async tasks becomes important (structlog's
`contextvars` integration handles this cleanly).

---

## Bug Fixes Applied (First Test Run Session)

These bugs were found and fixed while getting the first real HTTP request through end-to-end.

---

### Fix 1: Wrong database URL scheme for async driver

**File:** `app/config.py`, `.env`

**Problem:** `postgresql://` tells SQLAlchemy to use `psycopg2` (sync driver). But the code
uses `create_async_engine`, which requires an async driver.

**Fix:** Change to `postgresql+asyncpg://`.

**Background to learn:** SQLAlchemy URL prefixes select the *driver* (library used to talk to
the DB). `postgresql://` = psycopg2 (sync). `postgresql+asyncpg://` = asyncpg (async).
The `+asyncpg` part is called a "dialect+driver" specifier.

---

### Fix 2: AsyncSession cannot be used concurrently

**File:** `app/decisions/features/__init__.py`

**Problem:** `asyncio.gather()` runs coroutines concurrently. All 8 feature functions shared
the same `AsyncSession`. On a Redis cache miss, they all fall back to the DB simultaneously.
SQLAlchemy's `AsyncSession` does not support concurrent access — it raises:
`This session is provisioning a new connection; concurrent operations are not permitted`.

**Fix:** Replace `asyncio.gather(...)` with sequential `await` calls.

**Background to learn:** `asyncio.gather()` starts all coroutines at once (concurrent).
`AsyncSession` is not thread-safe or concurrency-safe — it assumes one operation at a time.
When Redis is warm (after the first request), the DB is never touched, so the performance
difference is negligible.

---

### Fix 3: Explicit db.begin() when transaction already open

**File:** `app/decisions/features/db.py`

**Problem:** `get_ip_country` called `db.execute()` directly, which implicitly starts a
transaction (SQLAlchemy "autobegin"). Then `get_did_days_on_file` called `async with db.begin():`
explicitly — but a transaction was already open, causing:
`A transaction is already begun on this Session`.

**Fix:** Remove `async with db.begin():` — the session's transaction is committed in `main.py`
via `await db.commit()`.

**Background to learn:** SQLAlchemy uses "autobegin" — a transaction starts automatically on
first use of the session. You only need to call `db.begin()` if you need manual transaction
control. Calling it twice raises an error.

---

### Fix 4: PostgreSQL range type with asyncpg parameter binding

**File:** `app/decisions/features/db.py`

**Problem:** The query `WHERE ip_range @> :ip` passes a Python `int` to a PostgreSQL `int8range`
column. asyncpg doesn't know to treat the int as a scalar point value in a range — it expects a
range object. Also, writing `:ip::bigint` in SQLAlchemy `text()` causes a syntax error because
`::` conflicts with the `:name` parameter syntax (`:ip:` is parsed as the parameter name).

**Fix:** Use `CAST(:ip AS bigint)` — standard SQL syntax that SQLAlchemy's parser handles correctly.

**Background to learn:**
- PostgreSQL `@>` is the "contains" operator for range types (`int8range`, `tsrange`, etc.)
- SQLAlchemy `text()` uses `:name` as parameter placeholders — `::` (PostgreSQL cast syntax)
  conflicts with this. Use standard SQL `CAST(x AS type)` instead.
- asyncpg maps Python types to PostgreSQL types automatically, but range containment with a
  scalar needs an explicit cast hint.

---

### Fix 5: ORM datetime columns must match timestamptz DB columns

**File:** `app/data/orm_models.py`

**Problem:** The PostgreSQL schema uses `timestamptz` (timezone-aware timestamps) for
`response_time`, `signup_time`, `purchase_time`, `created_at`. SQLAlchemy's `Mapped[datetime]`
defaults to `TIMESTAMP WITHOUT TIME ZONE`. asyncpg rejects timezone-aware Python `datetime`
objects when inserting into a tz-naive column:
`can't subtract offset-naive and offset-aware datetimes`.

**Fix:** Add `mapped_column(DateTime(timezone=True))` to all four datetime columns.

**Background to learn:** Python `datetime` objects can be "naive" (no timezone) or "aware"
(has `tzinfo`). PostgreSQL has two timestamp types: `TIMESTAMP` (naive) and `TIMESTAMPTZ`
(aware). asyncpg enforces this distinction strictly. SQLAlchemy's `DateTime(timezone=True)`
maps to `TIMESTAMPTZ`.

---

### Fix 6: ip_address ORM column is int32, DB schema is bigint

**File:** `app/data/orm_models.py`

**Problem:** `ip_address` is stored as a large integer in the dataset (e.g. `3627734734`),
which exceeds int32 max (~2.1 billion). The DB schema correctly uses `bigint` (64-bit), but
`Mapped[Optional[int]]` in SQLAlchemy defaults to `INTEGER` (32-bit), causing:
`value out of int32 range`.

**Fix:** Add `mapped_column(BigInteger, nullable=True)`.

**Background to learn:** SQLAlchemy's `Mapped[int]` maps to PostgreSQL `INTEGER` (4 bytes,
max 2,147,483,647). For larger integers, use `BigInteger` (8 bytes, same as PostgreSQL `bigint`).
IP addresses encoded as 32-bit unsigned integers exceed the signed int32 range.

---

---

## Bug Fixes Applied (End-to-End Test Run — 2026-03-16)

These bugs were found and fixed while running `client_mimic.py` against a live Docker stack.

---

### Fix 7: ip_country_block rule denied 'Unknown' country unconditionally

**File:** `app/decisions/rules.py`

**Problem:** `ip_country_block` returned `'deny'` for any country that wasn't `'United States'`,
including `'Unknown'`. The synthetic IP addresses in `Fraud_Data.csv` don't exist in the real
`IpAddress_to_Country.csv` ranges, so every lookup returned `'Unknown'` → every transaction denied.

**Fix:** Updated the rule to use `'Unknown'` + device age as a combined signal:
- `'United States'` → allow
- `'Unknown'` + device older than 90 days → allow (established device offsets unresolvable IP)
- `'Unknown'` + new device → deny (two unknowns together = high risk)
- Any confirmed non-US country → deny

**Background to learn:** Blocking on data-quality gaps (unknown country = deny) punishes legitimate
users when your lookup database has coverage gaps. A better policy is to use missing data as a weak
signal and require corroborating evidence before denying.

---

### Fix 8: execute_hard_rules lost the triggering rule name

**File:** `app/decisions/__init__.py`, `app/main.py`

**Problem:** `execute_hard_rules` used a single `if (rule1 == 'deny' or rule2 == 'deny' ...)`
expression — short-circuit evaluation means only the first matching rule fires, and the log
only said `"hard_rule.deny"` with no indication of which rule triggered it.

**Fix:** Replaced the short-circuit `or` chain with a loop over `(rule_fn, args)` tuples.
Both `execute_hard_rules` and `execute_soft_rules` now return `(decision, rule_name | None)`.
The log now includes `"rule": "ip_country_block"` (or whichever rule fired).

**Why this matters:** Without the rule name in logs, debugging "why is this transaction denied?"
requires adding temporary print statements. Named rule logging is essential for production audit trails.

---

### Fix 9: Redis idempotency cache returned stale deny decisions after rule changes

**File:** None (operational procedure)

**Problem:** After fixing rules, `client_mimic.py` still showed all-deny. The script generates
idempotency keys as `{user_id}-{counter}` where counter resets to 0 on each run. Redis cached
the old deny decisions (10-minute TTL) from the previous run, so the API returned cached results
without re-evaluating the new rules.

**Fix:** `docker compose exec redis redis-cli FLUSHDB` between test runs.

**Background to learn:** Idempotency caches are intentionally sticky — that's their purpose.
During development, always flush the cache after changing rule logic, or use unique key prefixes
per test session (e.g. include a timestamp in the key).

---

### Fix 10: model score() crashed on categorical string features

**File:** `app/decisions/models.py`

**Problem:** `model_features.json` includes `sex`, `source`, `browser`, and `age_group`.
The `score()` function handled `ip_country_code` and `days_since_signup_group` via special cases,
but let the rest fall through to `getattr(features, name)` — which returned raw strings like `'M'`,
`'SEO'`, `'Chrome'`. `np.array(..., dtype=np.float32)` cannot convert strings:
`ValueError: could not convert string to float: 'M'`

**Fix:** Added explicit encoding maps mirroring the `pd.Categorical` codes used during training:
- `sex`: `{'F': 0, 'M': 1}`
- `source`: `{'Ads': 0, 'Direct': 1, 'SEO': 2}`
- `browser`: `{'Chrome': 0, 'FireFox': 1, 'IE': 2, 'Opera': 3, 'Safari': 4}`
- `age_group`: `pd.cut` equivalent using `age_bins.json` + `bisect.bisect_left`

**Background to learn:** sklearn models store no encoding logic — they just expect numbers.
The encoding that happened during training (e.g. `pd.Categorical(...).codes`) must be reproduced
exactly at inference time. Saving the category lists to JSON (as this project does for
`ip_country_categories.json`) is the correct pattern; `sex`/`source`/`browser` just needed
the same treatment.

---

### Fix 11: client_mimic.py crashed on non-JSON server error responses

**File:** `app/data/client_mimic.py`

**Problem:** When the API returned a 500 error (empty body), `r.json()` raised
`JSONDecodeError: Expecting value: line 1 column 1`. The script crashed entirely instead of
printing the error.

**Fix:** Wrapped `r.json()` in a try/except; on failure, returns `{"_raw": r.text}` so the
raw error is visible in the output.

---

---

## Performance Optimisations (2026-03-19)

Identified and fixed during load testing with `client_mimic.py` + Grafana step-latency dashboard.
Starting point: feature extraction p95 ~700–800ms. After all fixes: ~66ms.

---

### Opt 1: asyncio.gather for Redis-first velocity functions

**File:** `app/decisions/features/__init__.py`

**Problem:** `build_features()` awaited all 8 feature lookups sequentially. Total time = sum of all
round-trips. At 5 RPS, the 7 Redis-first velocity functions alone added ~400ms in serial waiting.

**Fix:** Wrapped the 7 Redis-first velocity functions in `asyncio.gather()` so they fire in parallel.
Total time becomes `max(all)` instead of `sum(all)`.

**Trade-off:** The two guaranteed-DB calls (`get_ip_country`, `get_did_days_on_file`) are kept
sequential because `AsyncSession` does not support concurrent access — two coroutines calling
`db.execute()` on the same session simultaneously raise `IllegalStateChangeError`. Only Redis-first
functions are safe to gather because they almost never fall back to DB after the cache is warm.

**Rule:** Redis calls → `asyncio.gather`. DB calls → sequential.

---

### Opt 2: Redis cache for DB-backed lookups

**File:** `app/decisions/features/__init__.py`

**Problem:** `get_ip_country` (5M-row range query) and `get_did_days_on_file` (INSERT + SELECT on
`device_data`) hit the DB on every request. At 5 RPS these two sequential DB calls accounted for
~500ms of the feature extraction time.

**Fix:** Before calling the DB functions, check Redis for cached results (both checks run in parallel
via `asyncio.gather`). On cache hit, skip the DB entirely. On miss, call DB and cache the result.

| Key pattern | TTL | Rationale |
|-------------|-----|-----------|
| `cache:ip:{ip_int}` | 24h | IP→country mapping is static |
| `cache:did:{device_id}:{user_id}` | 1h | Device age changes by 1 day/day; 1h staleness is acceptable |

**Trade-off:** `did_days_on_file` can be stale by up to 1 hour. A device that crossed the 90-day
threshold within the last hour might still be treated as "new" and denied by `ip_country_block`.
Acceptable given the risk model — the alternative (DB on every request) is worse for latency.

---

### Opt 3: ZCOUNT instead of ZRANGEBYSCORE for count-only velocity functions

**File:** `app/decisions/features/velocity.py`

**Problem:** `get_did_uid_counts`, `get_uid_did_counts`, `get_ip_uid_counts`, `get_did_txn_counts`,
`get_country_txn_counts` all called `ZRANGEBYSCORE` and then `len(result)`. `ZRANGEBYSCORE` returns
the full list of member names (user IDs, device IDs) — potentially thousands of strings — just to
count them. This caused ~600ms of network transfer overhead per request.

**Fix:** Replace `ZRANGEBYSCORE` with `ZCOUNT` for all count-only functions. `ZCOUNT key min max`
returns the same count as `len(ZRANGEBYSCORE key min max)` but sends only an integer back. The time
window semantics are identical — the score is a unix timestamp in both commands.

**Trade-off:** None. `ZCOUNT` is strictly better when you only need the count. `get_purchase_counts`
was not changed because it needs the member values (purchase amounts) to compute sums and medians.

**Impact:** Feature extraction p95 dropped from ~800ms → ~66ms after this change.

---

### Opt 4: Eliminated sklearn feature-name warning overhead

**File:** `app/decisions/models.py`

**Problem:** `score()` built a plain `np.array` and passed it to `model.predict_proba()`. The model
was trained on a named pandas DataFrame, so sklearn emitted a `UserWarning` on every single call:
`X does not have valid feature names`. Python's warning machinery involves string formatting and
stack inspection on every call — measurable overhead at high RPS.

**Fix:** Changed line 78 from `np.array([...])` to `pd.DataFrame([row], columns=_features)`.
The model receives named columns, recognises them, skips validation. Warning eliminated.
Inference time: 13.9ms → 12.78ms per call.

**Trade-off:** Adds pandas as a runtime dependency of `models.py`. pandas was already a test/training
dependency; making it a runtime dep is acceptable given the marginal overhead.

---

### Opt 5: Single pipeline for all velocity Redis commands

**File:** `app/decisions/features/__init__.py`

**Problem:** `build_features()` used `asyncio.gather` to fire 7 velocity functions concurrently.
Each function internally issued its own `asyncio.gather` over 6–8 `ZCOUNT` calls. At 20 RPS with
~300ms latency, there were ~8 concurrent requests each holding 38 outstanding Redis coroutines
simultaneously — 304 coroutines all competing for the event loop. Python's event loop processes
one callback at a time; 304 pending callbacks added ~400ms of scheduling overhead per request.

**Fix:** Replaced the nested `asyncio.gather` tree with a single `redis.pipeline(transaction=False)`
containing all 38 Redis commands. Commands are queued synchronously (no `await`), sent in one TCP
write, and responses received in one TCP read. The event loop yields only once (`await pipe.execute()`).

**Trade-off:** `pipeline.execute()` blocks the event loop for the duration of one Redis RTT (~1ms).
At 20 RPS this means ~8ms/s of blocking — negligible. At 1000+ RPS the blocking would compete
with other requests' pipeline calls. Acceptable trade-off for the expected load range.

**Impact:** Feature extraction p95 dropped from ~800ms → ~300ms.

---

### Opt 6: Dropped redundant redis.exists() check from velocity functions

**File:** `app/decisions/features/velocity.py`

**Problem:** Every `get_*_counts` velocity function called `await redis.exists(key)` before the
`ZCOUNT` batch. This added a sequential round-trip before every function's main work — doubling
the Redis RTTs needed per function.

**Fix:** Removed the `exists` check from all count-only velocity functions. `ZCOUNT` on a
non-existent key returns `0`, which is the correct value for a new user/device with no history.
The DB fallback was dropped from these functions because returning 0 is semantically identical
to what the DB would return for a genuinely new entity.

**Retained:** The DB fallback in `get_purchase_counts` (needs real amounts, not just 0) and in
the `ip_country` / `did_days_on_file` lookups (where 0 would be wrong).

**Trade-off:** If Redis is cold (e.g. after a restart without reseeding) and a returning user
transacts, their velocity counts show 0 instead of historical values. Acceptable for this
project; production systems would warm the cache on startup.

---

### Opt 7: Single ZRANGE WITHSCORES fetch replaces 14 ZRANGEBYSCORE calls for purchase counts

**File:** `app/decisions/features/velocity.py`

**Problem:** `get_purchase_counts` called `ZRANGEBYSCORE` 14 times (7 windows × 2 sets: all and
success) to get purchase amounts per window. Each call transferred all matching member strings.
With 7 concurrent windows, Redis had to serialize and send the full list 14 times.

**Fix:** Replaced with 2 `ZRANGE … BYSCORE WITHSCORES` calls — one for the 90-day all-purchases
set and one for success-only. Both return `(member, score)` tuples where `member = "{uuid}:{amount}"`.
Sub-window filtering (`ts >= since`) happens in Python using the returned timestamps.

This turns 14 Redis calls into 2. Combined with the single pipeline (Opt 5), these 2 calls are
part of the same pipeline execution.

**Trade-off:** Python-side filtering is O(entries × windows). With ~12 entries/user and 7 windows
= 84 comparisons — negligible. Would only matter at thousands of purchases per user per 90 days.

---

### Infrastructure: Missing Docker mount for 004_create_transactions.sql

**File:** `docker-compose.yaml`

**Problem:** `003_create_device_data.sql` was mounted but `004_create_transactions.sql` was not.
PostgreSQL only runs init scripts on first container start — if the data volume already exists,
new scripts are skipped silently. The `transactions` table was therefore never created.

**Fix:** Added the missing volume mount. For existing containers (volume already initialized),
run manually:
```bash
docker compose exec db psql -U postgres -d fraud -c "$(cat app/data/db_init/004_create_transactions.sql)"
```
