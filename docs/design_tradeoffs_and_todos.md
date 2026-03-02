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

---

### 2. Redis Connection Per Request

**Problem:** Creating a new `Redis(host=..., port=...)` object inside `get_redis()`
on every request opens a new TCP connection each time — expensive handshake overhead,
port exhaustion under load.

**Trade-off:** Shared client is efficient but requires careful lifecycle management
(startup / shutdown).

**Solution keywords:** `lifespan`, `app.state`, connection pool, `aclose()` on shutdown,
`asynccontextmanager`

---

### 3. Pydantic Model vs ORM Model for DB Writes

**Problem:** `db.add(response)` fails because SQLAlchemy's `add()` requires a mapped
ORM instance, not a Pydantic model. They serve different purposes: Pydantic validates
data shapes; SQLAlchemy maps Python objects to table rows.

**Trade-off:** Maintaining two parallel class hierarchies (Pydantic + ORM) adds
boilerplate but separates API contract from persistence schema cleanly.

**Solution keywords:** `DeclarativeBase`, `Mapped`, `mapped_column`, `__tablename__`,
`model_dump()` → ORM constructor, `DBWrite` as source of truth for columns

---

### 4. CPU-bound Model Inference Blocking Event Loop

**Problem:** `model.predict()` is CPU-bound (no I/O waits). Running it directly in
an `async def` function blocks the event loop for ~800ms — all other incoming requests
queue behind it.

**Trade-off:** Thread pool adds concurrency but Python GIL limits true parallelism for
pure-Python models. Process pool bypasses GIL but has higher IPC overhead.
Most ML libraries (XGBoost, scikit-learn, PyTorch) release the GIL so threads work.

**Solution keywords:** `ThreadPoolExecutor`, `ProcessPoolExecutor`, `run_in_executor`,
`get_running_loop`, GIL, `max_workers = n_cpu_cores`, `Depends(get_executor)`

---

### 5. Idempotency Race Condition (TOCTOU)

**Problem:** Current pattern — GET key → if missing → process → SET key — has a
time-of-check-time-of-use gap. Two concurrent requests with the same idempotency key
can both pass the `if cached` check and both process the transaction.

**Trade-off:** Atomic claim with `SET NX` prevents double processing but requires a
polling/retry loop for the losing request, adding latency and complexity.

**Solution keywords:** `SET NX EX` (atomic), placeholder sentinel value, polling loop,
`asyncio.sleep`, `HTTPException 503` on timeout, TOCTOU

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

---

### 9. Response Model Carries Too Much Data

**Problem:** `Response` currently extends `Transaction`, so it returns all raw input
fields back to the client (signup_time, ip_address, age, etc.). The client sent these
— echoing them back wastes bandwidth and leaks internal data.

**Trade-off:** Lean response (just transaction_id + fraud_decision) is clean but
requires separating the API contract from the persistence model explicitly.

**Solution keywords:** lean response model, API contract vs persistence schema,
`model_dump(include={...})`, separate `DBWrite` from `Response`

---

## To-Do Items (Current Codebase)

### Minimum to get a test request through (do these first)

- [ ] **`main.py`** — define `engine` and `SessionLocal` (currently referenced but never defined;
      server crashes on first request with `NameError`)
      `python
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)
`
- [x] **`features/velocity.py`, `features/db.py`, `features/request.py`** — all stubs replaced
      with real implementations; no longer returns `Ellipsis`
- [ ] **`main.py`** — comment out `db.add(response)` and `db.commit()` temporarily until
      `orm_models.py` exists; passing a Pydantic model to SQLAlchemy `add()` crashes

---

### `app/main.py`

- [ ] Replace `Session` / `SessionLocal` with `AsyncSession` / `async_sessionmaker`
- [ ] Move `get_redis()` into `lifespan`, store on `app.state`, close on shutdown
- [ ] Add `get_executor` dependency backed by `ThreadPoolExecutor` in `lifespan`
- [ ] Fix idempotency: replace GET→check→SET with `SET NX` atomic claim + polling
- [ ] Replace placeholder comments in `run_process` with actual feature + model calls
- [ ] Call `update_velocity_counters()` via `asyncio.create_task` after response built
- [ ] Fix `db.add(response)` — convert Pydantic `DBWrite` to ORM instance before insert

### `app/data/data_models.py`

- [ ] Make `Response` lean — remove `Transaction` inheritance, keep only
      `transaction_id` and `fraud_decision`
- [x] Add default values to all `Features` fields; add missing 7d/90d velocity and
      purchase fields to match `build_features` output

### `app/data/orm_models.py` _(new file)_

- [ ] Define `TransactionRecord(Base)` with all `DBWrite` fields as `Mapped` columns
- [ ] Add `__tablename__` and primary key on `transaction_id`

### `app/data/db_init/` _(new files)_

- [x] `003_create_device_data.sql` — `device_data` table with `PRIMARY KEY (device_id, user_id)`;
      mounted in `docker-compose.yaml`

### `app/decisions/features/` _(package)_

- [x] `request.py` — `get_days_since_signup` and `get_age_group` implemented
- [x] `db.py` — `get_ip_country` (queries `ip_country` GIST index) and `get_did_days_on_file`
      (upsert into `device_data`) implemented
- [x] `velocity.py` — all four `get_*_counts` functions implemented using sorted sets
      (`ZRANGEBYSCORE` with per-entity keys, `asyncio.gather` for parallel window queries)
- [x] `velocity.py` — `update_velocity_counters()` implemented with `asyncio.gather` + `ZADD`;
      conditional write to `purchase:success:{user_id}` on non-deny decisions;
      write-time pruning via `ZREMRANGEBYSCORE` (90-day cutoff)
- [ ] Add read-through DB fallback to every Redis `get_*` function
- [ ] Add `GETEX` TTL refresh on every cache hit
- [ ] Add cache miss warning log to every fallback path

### `app/decisions/rules.py`

- [ ] Fix `transaction["purchase_value"]` → `transaction.purchase_value`
      (Pydantic uses attribute access, not dict access)
- [ ] Expand rules (velocity limits, device rules, country block list, etc.)

### `app/decisions/models.py` _(currently empty)_

- [ ] Load trained model at import time (not per request)
- [ ] Expose a `predict(features: Features) -> str` sync function for executor use
