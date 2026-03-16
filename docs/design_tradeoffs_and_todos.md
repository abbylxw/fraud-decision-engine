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

**Status: ✅ SOLVED** — `redis.set(..., nx=True, ex=600)` + `redis.getex()` for TTL refresh on reads

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

**Status: ✅ SOLVED** — `asyncio.create_task(update_velocity_counters(...))` fires after
response is built; ZADD is idempotent so duplicate writes are safe

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
- [x] Fix idempotency: replaced GET→check→SET with atomic `redis.set(..., nx=True, ex=600)`
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
- [ ] Add `GETEX` TTL refresh on every Redis cache hit (keep hot keys alive longer)
- [ ] Add cache miss warning log to every DB fallback path in `velocity.py`

### `app/decisions/rules.py`

- [x] All rules use Pydantic attribute access (`transaction.purchase_value`, not dict access)
- [x] Rules expanded: `transaction_limit`, `age_limit`, `ip_country_block`, `rate_limiter_control`,
      `model_score_threshold` — split into hard and soft rule sets
- [ ] Expand `rate_limiter_control` to use `score` parameter for combined velocity+model conditions
- [ ] Add more rules (country block list, device-age heuristics, etc.)

### `app/decisions/models.py`

- [x] `score(features) -> float` — returns raw fraud probability; used by `main.py` + `model_score_threshold`
- [x] `predict(features) -> str` — delegates to `score()` + `THRESHOLD` comparison
- [x] Model and feature list loaded at import time (not per request)

### `app/decisions/__init__.py`

- [x] `execute_hard_rules(t, f)` — chains transaction_limit, age_limit, ip_country_block
- [x] `execute_soft_rules(t, f, score)` — chains rate_limiter_control + model_score_threshold;
      all rules receive the model score for richer conditions

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
  - `hard_rule.deny` — user_id, ip_country (when hard rules short-circuit)
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

### `requirements.txt`

- [x] Add `pytest`, `pytest-asyncio`, `ruff`

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
