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
- [x] Rule review (2026-11): `transaction_limit` and the Unknown-IP / age >= 80 clauses removed,
      `rate_limiter_control` moved to shadow mode — see "Rule Logic Review" below
- [ ] Add more rules (country block list, device-age heuristics, etc.)

### `app/decisions/models.py`

- [x] `score(features) -> float` — returns raw fraud probability; used by `main.py` + `model_score_threshold`
- [x] `predict(features) -> str` — delegates to `score()` + `THRESHOLD` comparison
- [x] Model and feature list loaded at import time (not per request)

### `app/decisions/__init__.py`

- [x] `execute_hard_rules(t, f)` — chains signup_burst, age_limit, ip_country_block;
      returns `(decision, rule_name | None)` so the triggering rule can be logged
- [x] `execute_soft_rules(t, f, score)` — model_score_threshold;
      returns `(decision, rule_name | None)`
- [x] `execute_shadow_rules(t, f, score)` — rate_limiter_control in shadow mode; returns the
      branch names that would deny (logged and counted, never applied)

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
the old deny decisions (10-minute TTL at the time) from the previous run, so the API returned cached
results without re-evaluating the new rules. The TTL is now 60 s (`IDEMPOTENCY_TTL_SECONDS` in
`app/main.py`, cut in ab0f917), so a cached decision outlives a rule change by at most a minute.

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

## Rule Logic Review (2026-11)

Evidence: `results/agents/rule-logic-review/` (local, gitignored). Cost = missed fraud $ + $10
per false decline, on the Fraud_Data test period (30,223 rows).

**Changes**
- `ip_country_block`: the "Unknown IP + device pair ≤ 90 days" clause is removed. It denied
  14.6% of test traffic at a fraud rate no better than average (4.0% vs 4.7%), costing
  ~$36k [34.5k, 37.5k] in false declines. The OFAC branch stays.
- `transaction_limit` (≥ $5,000) removed: unreachable on Fraud_Data (max $129), and on Sparkov
  the 153 rows above $5,000 had 0 fraud.
- `age_limit`: only the invalid-age data check remains. The "age ≥ 80 + account ≤ 90 days" clause
  had no supporting data and, with a missing signup time (−999), denied by age alone.
- `rate_limiter_control` moved to **shadow mode**: its velocity cut-offs are expert-set and
  can't be validated on one-transaction-per-user data. Branches that would deny are logged
  (`shadow_rule.would_deny`, with `transaction_id`) and counted (`fraud_rule_shadow_total{rule}`);
  shadow errors are logged, never returned as a 500. The dead "$1M in 90 days" branch was deleted.
  Shadow hits are not stored in the audit row, but every branch input is, so precision can be
  measured offline against chargeback labels.
- Request schema: `device_id`, `ip_address`, `signup_time`, `source`, `browser`, `sex` and `age`
  are required (HTTP 422 if missing; an empty `device_id` or negative `ip_address` is also 422).
  Leaving out `signup_time` switched off `signup_burst` (0% of bot rows denied vs 100%), and
  leaving out `source`/`browser`/`sex` let ~60% of model-caught fraud through. Leaving out `age`
  dropped the model's fraud catches on the test period from 106 to 59; no encoding of "missing"
  fixes that, because the model never saw a missing age in training. (An earlier small probe
  showed only 2 of 104; the full-test measurement replaces it.)
- Training/serving parity for a missing age: the replay encoded a NaN age as `age_group` 0 while
  live gave −1, and `age_account_gap` was NaN in the replay but −1 − days/365 live. Now
  `age_group` is −1 on both paths, and `age_account_gap` is the sentinel −999 on both whenever age
  or signup time is missing (a formula value would look like a real gap). Checked in
  `tests/test_encoding_contract.py`. Training data is unchanged (Fraud_Data has no missing ages or
  signup times).

**Open risks (not fixed here)**
- [ ] **Heavy device sharing is not caught.** The model scores a device shared with exactly one
      other user as high risk but ≥ 5 users as low risk (0% denied); with `device_many_users_90d`
      in shadow mode nothing denies it. Options: monotonic constraint, a review outcome, or a
      high-cut-off deny.
- [ ] **No amount guard.** The model scores purchases above ~$120 lower (deny rate 1.3% → 0.46%),
      so larger baskets are an evasion route. Options: monotonic constraint on amount, or clip
      amount at serve time.
- [ ] **Unresolvable IPs skip the OFAC check** (IPv6, or a made-up `ip_address`). Needs the
      sanctions owner's sign-off on the list and the policy for Unknown.
- [ ] **`signup_time` is client-supplied.** A client can send a fake old date. Long term it should
      come from the server's own account store.
- [ ] **Evicted Redis user:** `time_since_last_txn_uid` reads 0 live but the real gap in the replay
      (the Postgres fallback has no last-transaction query, and its counts stop at 90 days). Not a
      served-model feature today; fix before any model uses it.
- [ ] **The model threshold assumes $10 per false decline.** At $40 the cost-optimal threshold
      moves from 0.843 to ~0.897.

---

## Model and Rule Review (2026-11)

Reproduce: `python -m training.evaluate` (model) and `python -m training.rule_backtest` (rules,
segment thresholds). Same 70/10/20 time split as `train.py`; cost = missed fraud $ + $10 per
false decline; test = 30,223 rows, 1,389 fraud, deny-nothing cost $51,395.

**Model: not retrained.** The served model at its threshold denies 365 test rows and catches 106
fraud; it saves ~$1.5k [0.7k, 2.4k] vs denying nothing, about 3%. Real but small. A 16-config
XGBoost re-check (depth 2–6, min_child_weight 10/50, with and without class weighting) scored
0.104–0.112 validation PR-AUC vs 0.108 today: within noise. The limit is the data (one purchase
per user, so most behaviour features are constant), not the tuning.

**Threshold depends on the false-decline cost.** Threshold chosen on validation, test savings:
$5 → $1.9k [0.7k, 3.1k] (4.5% denied); $10 → $1.5k [0.7k, 2.4k] (0.9%); $25 → about $0
[−0.7k, 0.3k]. At $25 or more per wrongly declined customer the model is not worth running on
this data, so the $10 assumption is the most important open business input.

**Rules on the test period**
| Rule | Fires | Precision | Cost change on top of model |
|---|---|---|---|
| `signup_burst` | 0 (7,600 on all rows, all fraud, all January) | 100% | works as designed |
| `age_limit` | 0 | — | data check only |
| `ip_country_block`, with Russia | 409 | 5.9% [4.0, 8.6] | +$3,133 |
| `ip_country_block`, without Russia (now) | 77 | 0% [0, 4.8] | +$760 (cost of compliance) |
| 5 shadow branches | 0 | — | never fire on one-purchase-per-user data |

**Change: Russia removed from `ip_country_block`.** The rule is for comprehensive (country-wide)
OFAC sanctions. Russia's sanctions are targeted (sectoral, listed parties), so a country-wide IP
block was broader than the policy. It denied 332 test rows at a 7.2% fraud rate (base 4.6%),
about twice what the model saves. Russia stays a model input (`ipc_grp_Europe`).

**Segment thresholds (rule condition + model score) don't help here.** "Deny if in segment S and
score ≥ t" with t below the global threshold, using a model fit on the train period only (the
served model was refit on train + validation, so its validation scores are optimistic). t is
chosen on validation; test is scored once:
- device shared (`did_uid_90d ≥ 1`, 23% fraud): t 0.833 vs 0.843, test +$85 [−358, 536]: noise.
- OFAC countries: a cut-off that helped on validation (4 fraud) cost +$290 [185, 410] on test,
  an example of a tiny segment overfitting.
- Russia, Unknown IP, new account, amount ≥ $100, shared IP: no cut-off helps on validation.

The model already sees country, device sharing and account age, so a segment threshold
counts the same signal twice. Score + rule combinations are worth it for signals the model
can't see (true velocity bursts, chargeback history), which is what the `elevated_score_*`
shadow branches wait for: they need repeat-user data.

**Open**
- [ ] **Syria:** its sanctions program was reported as lifted in 2025, but sources disagree. It
      stays blocked until the sanctions owner confirms the list against OFAC's own publications.
- [ ] **Sub-country sanctions** (Crimea, so-called DNR/LNR) can't be enforced with a
      country-level IP table.

**Dropped (2026-11-09):** `country_txn_count_{5min,10min,1h}`. A usage check found no model
input (`model_features.json`), hard rule, soft rule or shadow rule read them; `24h`/`7d`/`30d`
stay (model inputs; `24h` and `30d` are top features). Removing them saves 3 Redis `ZCOUNT`s
per request and 3 audit columns (`db_init/007`). The served model is unchanged, so no retrain.

## Retry After Decline: set features include the buyer, `device_many_users_1h` (2026-11-09)

Evidence: `results/agents/retry-after-decline/` (local, gitignored), `summary.md` §1–6 and
`fd-device-users-k1/result.json`. Cost in units: missed fraud = 3, false decline = 1.

**Semantics change: "distinct X" set features include the buyer.** `did_uid_*` (8 windows),
`uid_did_*` (8) and `ip_uid_*` (6) now mean "distinct members in the window *including* the
current purchase's member" (old count + 1 if the member was not already in the window). Count
features stay strictly prior: `did_txn_cnt_*`, `purchase_*`, `purchase_success_*`,
`country_txn_count_*`, and `amount_vs_user_avg_30d` (excludes the current amount);
`did_uid_days_on_file` is unchanged. Live: three extra `ZSCORE`s in the same pipeline (the
buyer's last-seen time in `did:uid`, `uid:did`, `ip:uid`); offline: `_distinct_since_incl` in
`training/build_features.py`. Why: "users on this device" then reads naturally (the first user
on a device sees 1, the second sees 2), so the rule is a plain `did_uid_1h >= 2` and denies the
second distinct user on their *first* attempt, while the same user retrying stays at 1.
Retrain: the model must be retrained on the new features (on Fraud_Data every user is new, so
the sets shift by exactly +1 and metrics should be ~unchanged). Retrain result
(`python -m training.train --ship`, 2026-11-09, `results/train_20261109T211048/`): ship gate
passed; new model test PR-AUC 0.092 [0.081, 0.107], threshold 0.8428, 365 test rows denied,
106 fraud caught, 259 false declines — identical to the previous model's recorded numbers, as
predicted. The *previous* model scored on the new features drops to PR-AUC 0.048 and a 44.6%
deny rate: that is what shipping the feature change without a retrain would have done, and why
the two must land in one commit. `results/cache/fraud_data.parquet` (old semantics) was deleted;
`training/compare_datasets.py` rebuilds it.
Side effect: `rate_limiter_control` (shadow) cut-offs on `did_uid_90d` / `uid_did_*` /
`ip_uid_7d` now trip one user earlier; not re-tuned (they were expert-set and never fire here).

**New hard rule: `device_many_users_1h`.** Deny when `did_uid_1h >= 2` (`DEVICE_USERS_1H_LIMIT`
in `rules.py`), event time. It runs right after `signup_burst`, so bot rows keep the
`signup_burst` audit label. Audit row: `decision_rule='device_many_users_1h'`, `model_score` NULL.
Fire rate per rule: Prometheus `fraud_decision_rule_total{decision_rule, decision}`.

**Why.** The January 2015 bot was 760 devices × 10 purchases × 10 different user_ids, the 10 rows
on a device within a median 9 s. The model does not catch bot rows (median score ≈ 0.52, it never
saw them on purpose), so `signup_burst` was the only defence, and a bot that waits 61 s after
signup walks past it. This rule does not look at signup time. Backtest (k1): catches 6,840 of
7,600 bot rows (9 of 10 per device; the first user on each device is not catchable this way);
legit rows hit 1 / 1 / 0 in train / validation / test. A 10 min window gives the same catch with
0 / 0 / 0 legit hits; 1 h was kept for margin against a slower bot.

**Limits and fail-open cases.**
- No bots in validation/test, so this is insurance against the observed attack and its 61-s
  variant, not a measured held-out gain. A bot using a new device per purchase evades it.
- Velocity counters are written in the background *after* the response, so two users hitting
  the same device within the same few milliseconds can both see 1 and both be allowed.
- Empty Redis (flush, eviction, cold start): the device looks new, everyone sees 1 -> allow.
- Out-of-order events: a purchase with an earlier `purchase_time` than one already recorded does
  not see the later one (event-time windows), so it may be allowed. Related: Redis keeps one
  last-seen time per member, so a late event for A overwrites A's earlier presence in the window
  and a request from B in between under-counts (offline would say 2). TODO: `ZADD ... GT` so a
  late *older* event cannot move a score backwards (critic v2, minor 2).
- TODO (critic v2): Grafana panel + alert on `fraud_decision_rule_total{decision_rule=
  "device_many_users_1h"}`; tests for the exact window-boundary score and a buyer score later
  than `purchase_time` (`results/agents/retry-after-decline/scratch/edge.py` has the cases); the
  ship gate compared against the previous model *scored on the changed features* (PR-AUC 0.048),
  so it checked nothing this time — when feature semantics change, gate against the previous
  model's recorded test metrics instead; `model_meta.json` `evaluation.previous_model` is to be
  read with that in mind.
- Load test: `client_mimic.py` replays shared-device rows, so some denies that used to be labelled
  `signup_burst`/model now show `device_many_users_1h` — a label shift, not a behaviour bug.
- Fraud_Data has few legit shared devices: watch the deny count on real traffic (households,
  shared/kiosk devices).

**Rejected**
- *Sticky `signup_burst` (user-keyed lockout after a deny):* catches nothing on the real attack
  (each bot user_id bought once) or on the wait-61-s bot (flag never set). It only helps against
  our own load client's retry pattern. On Sparkov a sticky deny after a model deny costs units at
  every window (1 h +9, 6 h +133, 24 h +478).
- *Promoting shadow branches B1 ("had a deny in the last X") / B2 (`purchase_cnt_{1h,24h}`
  cut-offs):* every B1 cell that fires costs units and raises repeat declines of good cards; 0 of
  40 B2 cells beats the model. Kept in shadow.
- *24 h window:* hits 7 legit + 2 fraud rows in train (+1 unit [−10, 10]); not worth it.
- *`did_uid_24h >= 2 AND signup gap < 10 min`* (a script's automatic pick): the gap condition
  brings back signup time, which a bot can adapt to.

**Lesson: tune the baseline before judging a rule.** One backtest first showed the sticky flag
saving 77–487 units. Its coarse threshold grid had picked 0.9969, leaving fraud undenied for the
rule to "catch". With a fine grid (threshold 0.35, tuned on validation only) the saving vanished.
A rule only looks good on top of a badly tuned model.

**Open**
- [ ] **Retry leak after a model deny.** 36–37% of retries after a model deny are allowed (the
      smaller amount lowers the score) and ~20% of those are fraud, ~4× the base rate: "lower the
      amount until it passes". Options: a "declined in the last X, now a smaller amount" check, or
      a floor on how far the score may drop on retry.

---

## Why public Fraud_Data results look 8× better than ours (2026-11-09)

Reference checked: a Medium post / GitHub repo reporting Random Forest precision 0.98, recall
0.54, ROC-AUC 0.77 on Fraud_Data (and 99.96% "accuracy" on creditcard.csv, where 99.83% is the
no-fraud base rate). Their recipe: random `train_test_split`, label-encoded `device_id`, calendar
parts of both timestamps (incl. month, minute, second) as features, SMOTE, default 0.5 threshold,
accuracy/ROC-AUC reporting, Flask endpoint that takes a pre-computed feature vector.

Reproduced with `training/random_split_baseline.py`: precision 0.991, recall 0.539, ROC-AUC 0.768.
Split by row type: recall on bot rows 1.000, on non-bot fraud 0.002; non-bot ROC-AUC 0.498.
Same recipe with a time split: recall 0.000, ROC-AUC 0.506. So the reported recall is the bot's
share of fraud (53.8%), learned from device ids and `month == 1` that leak across a random split;
the model has no skill on anything else and none at all on future months.

Takeaways recorded for interviews / README: (1) split by time; (2) a repeated attack belongs to a
rule, not to a model that memorises its ids; (3) never label-encode a high-cardinality id and then
SMOTE it; (4) report PR-AUC against the base rate, not accuracy. Our model's test PR-AUC 0.092
(2× chance) is what remains once the easy 54% is removed from the model's job.

## Model candidates: tuned LightGBM and CatBoost added, XGBoost kept (2026-11-10)

**Change:** `training/train.py` and `training/evaluate.py` compare four tree candidates
(random forest, XGBoost, LightGBM, CatBoost) instead of two or three. New
`training/tune.py` runs a validation-only grid per algorithm (66 configs, ~2 min) and writes
`results/tune_<ts>/sweep.json` (copy in `docs/evaluation/tune_sweep.json`); the winners are pasted into both scripts by hand with the
sweep folder in a comment. New `training/candidates.py` holds the split/feature-selection
block that train.py and tune.py share and the per-library early-stopping glue. The
explanation of the four algorithms and the full results table are in
`docs/model_candidates.md`.

**Result:** all four tie on the test period (PR-AUC 0.085 to 0.096, every 95% CI about
±0.013 wide and overlapping). LightGBM won validation (0.115 vs XGBoost 0.112) and was middle
of the pack on test. The ceiling is the data, not the learner.

**Selection rule (`train.select_candidate`):** XGBoost is the incumbent because it is the only
booster in the `api` dependency group. A challenger ships only if it wins validation *and* its
test PR-AUC exceeds the incumbent's 95% CI upper bound. Trade-off accepted: this reads test
scores for every candidate, so test now influences model choice slightly; the previous script
scored test once. Switching the serving stack on a validation-only win of 0.003 was judged
the worse error. The rule and reason are recorded in `model_meta.json["selection_rule"]`.

**Library traps found (kept here so they are not re-learned):**
- LightGBM's `early_stopping` callback watches every metric in the model's list, including
  the default log-loss, which with `scale_pos_weight` = 21 worsens from tree 1; training
  stopped at 1 tree. Fix: `metric="average_precision"` + `first_metric_only=True`.
- CatBoost's `PRAUC` eval metric gave 0.108 after one tree where scikit-learn's average
  precision gives 0.076 (different interpolation under heavy score ties), so its overfitting
  detector stopped at tree 1; `use_weights=false` made no difference. Fix: train the full
  budget and pick the tree count with scikit-learn's metric via `staged_predict_proba` every
  10 trees, then `shrink()`.

**Dependencies:** `lightgbm==4.6.0`, `catboost==1.2.8` in the `train` group only. CI (`dev`
group) and the Docker image (`api` group) are unchanged; no test imports the training scripts.

**TODOs:**
- The sweep is a grid, not a budgeted search; with this little signal a wider search would
  only re-sample noise, but on a richer dataset replace it with `optuna` on validation PR-AUC.
- CatBoost's categorical handling is unused because all served features are numeric. The
  real experiment is feeding it raw `device_id` / `ip_country` strings, which needs a
  CatBoost-specific feature path on both training and serving sides.
- `--ship` of the retrained XGBoost (sweep settings, same test score) is a separate decision.

## Rolling-origin evaluation of the four candidates (2026-11-15)

**Why:** one validation month and one test period ranked the boosters differently
(LightGBM first on validation, CatBoost first on test). `training/rolling_eval.py` repeats
"train on the past, score the next period" seven times (scored periods May to Nov+Dec,
expanding training window, tree count picked on the month before the scored one,
hyper-parameters fixed). Results: `docs/evaluation/rolling_origin.json` and `.png`; full
tables in `docs/model_candidates.md` section 6.

**Result:** mean PR-AUC XGBoost 0.090, LightGBM 0.089, CatBoost 0.089, random forest 0.084.
Paired bootstrap of the mean difference vs XGBoost: LightGBM −0.001 [−0.005, +0.002],
CatBoost −0.001 [−0.004, +0.002], random forest −0.006 [−0.009, −0.002]. The boosters tie
over time; the random forest is slightly but measurably behind. XGBoost stays. Nothing was
retrained or shipped.

**Trap found:** XGBoost's `eval_metric="aucpr"` over-credits one-tree models (0.088 vs
scikit-learn's 0.077 in fold 4; tied scores, different interpolation), so
`early_stopping_rounds=100` stopped at tree 1 in two folds and under 15 trees in three more,
under-fitting by up to 0.02 PR-AUC (seven-fold mean 0.082 instead of 0.090). The first run therefore showed the other two boosters
"significantly" ahead. `rolling_eval.py` trains the full budget and picks the tree count with
scikit-learn's average precision every 10 trees (same fix as CatBoost).

**Follow-up, same day:** the fix was moved into the shared `candidates.fit_candidate`
(XGBoost trains the full budget; the best tree count is written to the booster's
`best_iteration`, which `predict_proba` honours, so no caller changes). `early_stopping_rounds`
and `eval_metric="aucpr"` were removed from the XGBoost constructors in `train.py`,
`evaluate.py` and `tune.py`. Re-run sweep: same winning XGBoost config (depth 3, lr 0.03,
min_child_weight 100; 370 trees vs 368), and the "reject configs under 50 trees" rule is no
longer needed. Retrain without ship: test PR-AUC 0.0917 vs served 0.0920, so nothing shipped.
Cost: each XGBoost fit always trains 1000 trees and scores validation 100 times (about 1 s
more per fit at this size). `training/compare_datasets.py` (Sparkov) still uses XGBoost's own
early stopping; scores there have many distinct values (PR-AUC 0.97), so the trap does not
apply, but it is the one script not on the shared helper.

**TODOs:**
- Folds reuse hyper-parameters tuned on Aug 5 to 30; a nested version (re-tune inside each
  fold) would remove that, at 7× the sweep cost and little expected change.

## All-time device count `did_txn_cnt_all`: built and tested, not shipped (2026-11-15)

Full study: `docs/feature_ablation.md`. Summary of the decisions.

**Problem:** four tuned algorithms tied at PR-AUC 0.09; every device counter forgot a device
after 90 days. Devices seen at any earlier time are 4.9% of traffic with a 22.9% fraud rate
(3.6% on new devices).

**Decision 1, which features:** ablation on the rolling-origin folds
(`training/feature_ablation.py`). `did_txn_cnt_all` alone: +0.033 mean PR-AUC
[+0.028, +0.038]. Recency (seconds since first / last transaction) and distinct users ever
give the same gain alone and +0.001 on top. Served: the count only. Alternative rejected:
serving all four "to be safe" (three more values to keep in parity for no measured gain).

**Decision 2, how to serve it:** a plain Redis counter `did:txn_all:{device_id}` (`INCR`, no
TTL), read in the existing pipeline; Postgres `COUNT(*)` fallback on a miss behind a new
`(device_id, purchase_time)` index, re-seeding Redis with `SET NX`. Alternatives rejected:
(a) stop purging the existing `did:txn` sorted set at 90 days: event-time correct, but one
member per transaction forever instead of one integer per device; (b) Postgres only: an
extra query on every request rather than only on first sight of a device.

**Trade-offs accepted** (details and "what would change it" in the study, section 5):
arrival-order rather than event-time counting (pinned by
`test_counter_is_arrival_order_not_event_time`); unbounded Redis keys, one per device; one
extra indexed query per first-seen device, latency not yet load-tested; denied transactions
count.

**Result:** retrained XGBoost (42 features, depth 3, lr 0.05, min_child_weight 100, 200
trees): test PR-AUC 0.141 [0.125, 0.160] vs served 0.092 [0.081, 0.107]; paired difference
+0.049 [+0.040, +0.059]; ROC-AUC 0.641 vs 0.557. Ship gate passed.

**Decision 3, not shipping yet.** At the current costs (false decline $10, average fraud
$37, break-even precision for a decline 21% = $10 / ($10 + $37)) the new model's validation-chosen threshold declines 6.2% of
traffic instead of 1.2%, catching 435 frauds instead of 106 with 1,438 false declines
instead of 259, for the same total cost ($49.7k both). The newly found fraud sits in a band
of about 23% precision, barely above that break-even: clearly profitable for a $5 review or
step-up (break-even 13.5%; total cost $42.1k vs $49.4k), next to worthless for a decline. Shipping it behind allow/deny would quintuple declines for nothing.
Served model unchanged: `xgboost-20261109211048-5ba3793`. The serving code for the feature
is merged and safe: the API computes and stores the value, the served model ignores it.

**Also changed:** hyper-parameters in `train.make_candidates` / `evaluate.make_models`
re-tuned with the new feature. `training/evaluate.py` scores the *served* feature list, so
`docs/evaluation/metrics.json` and the README table were deliberately not refreshed: they
describe what is deployed.

**TODOs:**
- `review` decision band (score between the review and deny thresholds) so the model can be
  shipped where it earns money; then retrain with `--ship` after explicit approval.
- Load-test the first-seen-device Postgres lookup (`client_mimic.py`, p50/p99 before/after).
- Existing databases need `db_init/008` applied by hand (column + index).
- `training/evaluate.py` should take the candidate feature list as an option so a not-yet-
  shipped feature set can be charted.
- A backfill script to rebuild `did:txn_all:*` from Postgres after a Redis flush (the
  fallback does it lazily, one device at a time).

## Third decision `review` between allow and deny (2026-11-15)

**Problem:** the better model's extra catches sit at about 23% precision. A decline breaks
even at 21% ($10 / ($10 + $37 average fraud)), so each of those declines earns under a
dollar and behind allow/deny the model was worth nothing extra (previous section). A $5
review breaks even at 13.5% ($5 / $37). Fraud teams handle this band with a cheaper
action than a decline.

**Decision:** `fraud_decision` is now `allow`, `review` or `deny`. `review` means "do not
auto-approve: step up (3-D Secure, SMS code) or queue for a person". The engine reports it;
the caller acts on it. Scores in `[review_threshold, threshold)` are reviewed, scores at or
above `threshold` are denied; both live in `model_meta.json`. Hard rules never review. The
audit rule name is `model_score_review`.

**Cost model (`training/train.py::policy_cost`):** allowed fraud loses its purchase value;
a wrongly declined order costs `--false-decline-cost` ($10); every reviewed order costs
`--review-cost` ($5); a reviewed fraud is stopped with probability `--review-catch-rate`
(1.0). `pick_thresholds` finds the cheapest pair on validation in one pass (the cost splits
into a review part and a deny part), checked against brute force in
`tests/test_thresholds.py`.

**Alternatives rejected:**
- *A fixed band, e.g. "top 5% of scores".* Simple, but not tied to cost: it would review the
  same share whether a review costs $1 or $20.
- *Reusing `ELEVATED_SCORE` (top 3% of validation scores).* That constant feeds the shadow
  rate-limiter combinations; tying the customer-facing decision to it would couple two
  things that are tuned for different reasons.
- *A second model for the review decision.* No labels exist for review outcomes.
- *Lowering the deny threshold instead.* That is the cost-neutral 6% decline policy.

**Backward compatibility:** `REVIEW_THRESHOLD` defaults to `THRESHOLD` when
`model_meta.json` has no `review_threshold`, so the currently served model behaves exactly
as before (empty band). Clients that only know `allow`/`deny` must be updated before a model
with a band ships: treating an unknown `review` as `allow` silently drops the protection.

**Trade-offs accepted:**
- Review cost and catch rate are assumptions. Sensitivity table in
  `docs/feature_ablation.md` §7: the gain holds for a review ≤ about $5 that stops ≥ 80%;
  beyond that the band disappears.
- A reviewed purchase counts as "successful" in the `purchase_success_*` velocity counters
  (its outcome is unknown at decision time). Fix needs label feedback.
- With ties in cost the search prefers the policy acting on more orders.
- The ship gate's cost comparison is a point estimate without a confidence interval.

**Result of the first run (the shipped policy differs slightly, see the next section):** new model with review ≥ 0.784 and deny ≥ 0.904
on test: 6.6% reviewed (458 frauds at 23% precision), 6 orders denied, total cost $44,403 vs
$49,714 as served (−10.7%) and $51,395 doing nothing. Both ship gates pass.

**TODOs:**
- Tie-break in `pick_thresholds` towards the least intrusive policy.
- Bootstrap interval on the cost difference in the ship gate.
- `training/evaluate.py` and `training/rule_backtest.py` still reason about a single deny
  threshold; extend them to the three-way policy after the ship.
- Review outcomes as labels (`POST /api/labels`) to measure the real catch rate.

## Shipped: 42-feature model with a review band; feature corrected the same day (2026-11-15)

Served model: `xgboost-20261115162923-3b6394a`, review ≥ 0.801, deny threshold 1.01 (never).
Evidence and tables: `docs/feature_ablation.md` sections 7 and 8.

**1. Least-intrusive tie-break (`training/policy.py`).** The cost curve is flat near its
minimum, and the exact minimum sometimes sat on a policy touching 6% of orders for a few
dollars. Rule: among threshold pairs within 1% of the lowest validation cost, take the one
that touches the fewest orders, then the fewest declines. Effect on the shipped policy:
6.6% reviewed + 6 declines ($44,403 on test) became 6.2% reviewed + 0 declines ($44,728).
Trade-off accepted: up to 1% of validation cost given up for a calmer, more stable policy.
The 1% is a judgment, not a measurement.

**2. Cost functions moved to `training/policy.py`** (numpy only) so their tests run in CI,
which does not install LightGBM/CatBoost. The ship gate now also reports a paired bootstrap
interval for the cost difference (−$4,986, [−$6,348, −$3,778] for this ship); the gate rule
itself still uses the point estimate.

**3. First ship, then a flaw.** `xgboost-20261115160020-9aee13e` shipped with
`did_txn_cnt_all`. Running the demo and a hand-made "same customer, own device, three
purchases" sequence against the live stack showed purchases 2 and 3 going to review (score
0.81). Cause: on Fraud_Data every user buys once, so "the device has history" and "another
account used the device" are identical columns; the model learned the second from a feature
that measures the first.

**4. Fix: `did_other_uid_all`** (other accounts ever seen on the device). Identical to
`did_txn_cnt_all` on all 151,112 Fraud_Data rows, so the retrain reproduced the model
exactly (PR-AUC 0.141, cost $44,728) and was re-shipped under the pre-agreed rule "only if
within noise of the approved model". Serving: Redis set `did:uids_all:{device}`; the two
all-time features share one Postgres fallback query; `db_init/009`. Alternatives rejected:
(a) keep the counter and add a rule "same user → ignore": two features doing one job;
(b) reuse the 90-day `did:uid` sorted set without purging: event-time correct, but changes
the meaning of every windowed read on that key. `did_txn_cnt_all` stays as an audited value
(not a model input) because it is one integer and makes the difference visible in the audit
table.

**5. Not adopted: monotone constraint on `did_other_uid_all`.** Unconstrained, the score
falls back to about 0.52 at five or more other accounts (the training window's shared
devices were January bot devices whose later purchases were mostly legitimate). A constraint
fixes that at −0.006 test PR-AUC [−0.010, −0.001] and no cost difference. Left as an open
decision because it is outside the "within noise" re-ship rule.

**6. Load test of the new lookups** (`client_mimic.py --rps 200 --duration 30`, one client,
local Docker, before/after the rebuild): p50 7 → 7–8 ms, p95 15 → 11–13 ms, p99 24 → 16–21 ms.
No regression visible; single 30-second runs, so differences of a few ms are noise. The
caveat is that this local database is full of replayed load-test rows, so few requests were
true first-seen devices.

**7. `training/evaluate.py`** now treats the served *review* threshold as the served alert
threshold (its cost view is "alert at $5 each"); with the deny threshold at "never" it had
reported the served model as doing nothing.

**Other trade-offs still open:** review cost and catch rate are assumptions; a reviewed
order counts as a successful purchase in the velocity counters; all-time device history is
arrival-order, not event-time; the two all-time Redis keys never expire.

**TODOs:**
- Decide on the monotone constraint (above).
- A named-stories smoke test after every ship is now `scripts/demo.py`; run it in CI
  against a Compose stack (needs the real-Postgres CI job from the hardening list).
- A load test on a clean database with mostly first-seen devices.
- Backfill script for `did:txn_all:*` / `did:uids_all:*` after a Redis flush (the fallback
  does it lazily, one device at a time).

## Monitoring and alerting; monotone model shipped (2026-11-15)

Runbook and full description: `docs/monitoring.md`. Decisions and their reasons:

**1. Prometheus rule files + Alertmanager, not Grafana-managed alerts.** Rules are plain
files in the repository: reviewed in a diff, unit-tested with `promtool test rules` in CI
(13 scenarios), and independent of the dashboard tool. Cost: one more container
(Alertmanager) and alert state shown in Grafana through a query on `ALERTS` instead of
Grafana's own alert UI.

**2. Drift computed in PromQL from bucket counters, not a batch job.** Training writes
bucket edges and expected shares for the score (deciles) and eight inputs to
`model_meta.json`; the API counts each scored request into its buckets; a recording rule
computes PSI over a 1-hour window. No new service, minute-level freshness, same alerting
path as everything else. Trade-offs accepted: a handful of buckets per feature (buckets
under 1% of traffic are merged, so rare-value shifts are invisible), only monitored
features, no joint distributions, and a 1-hour memory. One-hot country inputs are not
drift-monitored (the country panel shows the mix).

**3. Dependency health by in-process probes** (each worker pings Postgres and Redis every
10 s) rather than exporter sidecars. Fewer containers and it measures what matters, "can
the API reach it". It does not give database internals (connections, replication lag,
memory); `postgres_exporter` and `redis_exporter` are the upgrade.

**4. Thresholds are stated starting points.** Each alert's level and its reasoning are in the
runbook. Absolute where a level is meaningful everywhere (1% errors, PSI 0.10 / 0.25),
relative to the alert's own history or to the training-time expectation where it depends on
traffic (deny rate, fallback rate, review rate). Every ratio alert has a minimum-traffic
guard and a `for` duration.

**5. No notification channel in git.** Receivers are named and empty; the config shows where
a Slack receiver goes. A webhook URL is a secret.

**6. Monotone constraint shipped** (`xgboost-20261115170732-a5e527a`): test PR-AUC 0.136
(−0.006), policy cost $44,554 (−$174, interval [−$592, +$193]), review ≥ 0.762. Details
and the reasoning in `docs/feature_ablation.md` §8.

**7. Incident: a model that trained but could not be served** (same section). Process
changes that came out of it: `--ship` reloads the saved file the way the API does and
compares scores before replacing the served model; `tests/test_served_model.py` uses the
real model file.

**8. Verified live:** all rule groups healthy in Prometheus; `scripts/alert_drill.py`
(3 bot signups per second) took `FraudSignupBurstAttack` to pending in 45 s and firing in
161 s, and it arrived in Alertmanager one second later; load test with the new counters and
probes: p50 7 ms, p95 12 ms, p99 18 ms at 200 RPS, unchanged.

**TODOs:**
- **`amount_vs_user_avg_30d` is the purchase amount for every first-time customer**
  (average taken as 1), so the model reads a repeat customer's ratio of ~1 as a $1 purchase.
  Found by the drift alert on its first live run. Fix on both sides (ratio 1.0 when there is
  no history), retrain, ship. Then audit the remaining inputs for the same flaw.
- `country_txn_count_30d` differs by two orders of magnitude between a full-history replay
  and a short one: decide whether a country volume counter belongs in the model at all.
- Label endpoint, then live precision/recall and the measured review catch rate.
- A real notification receiver; exporter sidecars; `scripts/demo.py` and the drill in CI
  against a Compose stack.
- A batch drift job over the audit table (all features, longer windows) once labels exist.

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
threshold within the last hour might still be treated as "new" (originally by `ip_country_block`'s
Unknown-IP clause, removed in the 2026-11 rule review; today only the shadow branch
`device_many_users_90d` reads it). Acceptable — the alternative (DB on every request) is worse
for latency.

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

---

### Load test with repeat customers: `client_mimic.py --source sparkov` (2026-11)

**Problem:** Fraud_Data has one purchase per user, so a load test never grows the per-user Redis
sets past one entry. The window reads, the 90-day purge and Opt 7's Python-side filtering are
never exercised at realistic sizes.

**Change:** `--source sparkov` replays the first 100k rows of Sparkov `fraudTrain.csv` (922 cards,
median 106 purchases per card, Jan-Feb 2019). Sparkov has no device, IP, source, browser or signup
time, so the client makes them up: one device and one US IP per card, `Direct` / `Chrome`, and
signup 365 days before the card's first purchase (signup = first purchase would trip
`signup_burst` on every card; combined-dataset-model summary, recommendation 4).

**Trade-off:** This is for latency and Redis load only. The served model can't score
Sparkov-shaped traffic (combined-dataset-model, section 3), so the allow/deny decisions from this
mode are not an evaluation. The data stays in `data/external/` and never enters the Docker image.
