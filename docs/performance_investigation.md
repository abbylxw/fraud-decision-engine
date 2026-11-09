# Performance Investigation — Docker Load Test (2026-04-08)

> **Superseded (2026-11-08):** the ceiling and "400 RPS safe" figures below no longer hold.
> Re-measured at ~50 RPS (repeat customers) and under 30 RPS (new users), because of a missing
> index on `transactions.user_id`. See [performance_2026-11.md](performance_2026-11.md).
> This file is kept as the history of the April investigation.

## Complete Investigation Summary (2026-04-08/09)

### Phase 1 — Initial Grafana observations at 150 RPS

Grafana showed feature extraction ~1,000ms, model inference ~400ms, DB write ~300ms. Initial hypothesis: Docker virtual network adds 100ms+ per Redis/Postgres round-trip, and cold cache DB fallbacks are stacking sequentially.

---

### Phase 2 — Isolating individual components (all fast in isolation)

| Test | Result | Conclusion |
|---|---|---|
| Cold vs warm request (Test 5) | cold=7.9ms, warm=1.9ms | Cache fallbacks add ~6ms — not the bottleneck |
| Redis RTT under load (Test 2) | p50=0.43ms, p95=1.08ms | Redis is fast, not the bottleneck |
| Postgres commit under load (Test 3) | p50=1.24ms, p95=1.75ms | Postgres is fast, not the bottleneck |
| Model inference inside Docker | p50=13.9ms — identical to native | Docker does not penalize CPU |
| Pydantic construction (80 fields) | p50=0.003ms | Negligible |
| `_parse_purchase_entries` (50 entries) | p50=0.046ms | Negligible |
| DBWrite + TransactionRecord serialization | p50=0.060ms | Negligible |
| ThreadPoolExecutor capacity | 14 workers ÷ 0.014s = 1,000 RPS ceiling | Not the bottleneck |

**Key insight from cold vs warm:** both are fast (under 8ms). The 1,000ms latencies in Grafana were queue latency from sending 150 RPS into a server that could handle ~65 RPS — not slow individual requests.

---

### Phase 3 — Finding the true throughput ceiling (binary search)

| RPS | p50 | p95 | p99 | Verdict |
|---|---|---|---|---|
| 20 | 21ms | 38ms | 61ms | Comfortable |
| 40 | 7ms | 21ms | 23ms | Cache warming effect |
| 60 | 19ms | 75ms | 347ms | Healthy |
| 65 | 20ms | 146ms | 274ms | Marginal |
| 70 | 24ms | 447ms | 668ms | Breaking |
| 80 | 7ms | 1,616ms | 1,696ms | Saturated |

**Ceiling: ~65–70 RPS on a single uvicorn worker.**

---

### Phase 4 — Isolating background task contention

Disabled `asyncio.create_task(update_velocity_counters(...))` and retested:

| RPS | p95 (tasks ON) | p95 (tasks OFF) | Change |
|---|---|---|---|
| 70 | 447ms | 112ms | 4x improvement |
| 80 | 1,616ms | 588ms | 2.7x improvement |
| 100 | — | p50=4,400ms | Still a hard ceiling somewhere |

**Conclusion:** Background tasks contribute ~30% of ceiling reduction (65 → 75 RPS) but are not the sole cause. Something else has a hard ceiling around 75–80 RPS even without them.

---

### Phase 5 — In-process timing under real load

Added `perf_counter` instrumentation around each step inside `run_process`. Results at 80 RPS:

| Step | Isolated time | Under 80 RPS load |
|---|---|---|
| Feature extraction | ~1ms (Redis pipeline) | ~300ms |
| Model inference | 13.9ms (real CPU work) | ~200–400ms |
| DB write | 1.24ms (Postgres commit) | ~350ms, spikes to 1,700ms |

Every step is 20–100x slower under load than in isolation, despite the underlying operations (Redis, Postgres, model) being fast. This rules out all I/O and CPU as bottlenecks.

---

### Root Cause: Event Loop Callback Queue Saturation

The event loop is a single-threaded scheduler. Every `await` adds a callback to its queue. When a Redis response arrives in 0.4ms, it doesn't resume the coroutine immediately — it adds a "resume" notification to the queue. The coroutine wakes up only when the event loop processes that notification.

At 80 RPS with ~5 `await` points per request:
- **400 callbacks/second** competing for the single event loop
- A Redis response arriving in 0.4ms sits in the queue for ~300ms before being processed
- Model inference finishes in 14ms but the "I'm done" thread notification waits 200ms in the queue
- DB commit completes in 1ms but waits up to 1,700ms to be acknowledged

At 65 RPS (325 callbacks/sec) the queue just barely drains. At 70+ RPS it grows faster than it drains and wait times compound.

Background velocity tasks made this worse: at 65 RPS they added 65 × ~15 Redis commands = ~975 extra callbacks/second on top of the 325 from real requests.

---

### Fix Applied: Multiple Workers

**Dockerfile updated:** `--workers 4` added to uvicorn CMD.

Each worker is an independent Python process with its own event loop. 4 workers × 65 RPS each = **~260 RPS** while maintaining the same per-request latency profile seen at 60 RPS with one worker.

`client_mimic.py` updated to **60 RPS** (safe operating point for a single worker, useful for single-worker testing).

---

### Multi-Worker Ceiling Verification (2026-04-09)

Switched to 8 workers (`pool_size=5, max_overflow=6` per worker = 88 total connections).

| RPS | p50 | p95 | p99 | Verdict |
|---|---|---|---|---|
| 400 | ~0ms | ~40ms | ~800ms | Comfortable |
| 500 | ~0ms | ~70ms | ~1,000ms | Healthy |
| 575 | ~0ms | ~250ms+ | ~2,000ms+ | Breaking |
| 650 | ~0ms | ~750ms | ~2,500ms | Saturated |

**Ceiling: ~520 RPS = 8 workers × 65 RPS per worker.**

The per-worker event loop ceiling (65 RPS) is the binding constraint regardless of worker count. At the ceiling, Grafana shows model inference spiking first (most callbacks per request), then db_write — same callback queue saturation pattern as single-worker, just at 8× the RPS.

**Safe operating point: 400 RPS** (`client_mimic.py` set to 400 RPS, 120s).

**To raise the ceiling further:**
- Move `update_velocity_counters` off the event loop (separate process / Redis queue) — would raise per-worker ceiling from ~65 to ~75+ RPS
- Add more workers beyond 8 — requires increasing Postgres `max_connections` or adding PgBouncer

---

## Observation Gaps

| Metric | Theoretical (native) | Measured (Docker/Mac) | Gap |
|---|---|---|---|
| Request rate sustained | 150 req/s | ~2–3 req/s | 50–75x under |
| End-to-end p50 | ~20ms | ~300ms | 15x over |
| End-to-end p95 | ~25ms | ~1,000ms | 40x over |
| Feature extraction p95 | ~10ms | ~1,000ms | 100x over |
| Model inference p95 | ~13ms | ~400ms | 30x over |
| DB write p95 | ~4ms | ~300ms | 75x over |

**Anomaly:** Feature extraction is slower than model inference, despite model inference being CPU-bound (the step expected to suffer most from virtualization). This is the key observation — it rules out CPU overhead as the primary cause and implicates network I/O.

---

## Hypotheses

### H1 — Docker virtual network adds 50–150ms per Redis/Postgres round-trip

**Mechanism:** On macOS, Docker runs inside a Linux VM via Apple Hypervisor. Every network syscall (send/recv) crosses the VM boundary. Redis and Postgres both live in separate containers, so each `await redis.command()` or `await db.execute()` pays this tax.

**Assumptions:**
- Redis RTT inside Docker is materially higher than from the host
- Postgres RTT inside Docker is similarly elevated
- This would not appear on Linux bare metal

**Predicts:** Feature extraction slow even on warm cache; all steps affected proportionally to their RTT count, not their CPU cost.

---

### H2 — Cache is cold on most requests, forcing sequential DB fallbacks

**Mechanism:** `Fraud_Data.csv` has ~100k unique users. Redis TTLs are 24h (ip_country) and 1h (did_days_on_file). On every run, most users have never been seen — their keys don't exist, triggering:
- `get_ip_country` → DB query (~150ms)
- `get_did_days_on_file` → DB upsert (~150ms)
- `_purchase_counts_from_db` → DB query (~150ms, when `purchase:all:{user_id}` is empty)
- `_uid_last_txn_from_db` → DB query (~150ms, when `uid:last_txn:{user_id}` is missing)

These are sequential (shared `AsyncSession` can't be concurrent). Total cold path: 4 × 150ms = ~600ms on top of warm-path pipelines.

**Assumptions:**
- Most CSV user_ids have no Redis keys at test start
- Each DB fallback adds one full Postgres RTT sequentially
- Warm-cache requests would be significantly faster

**Predicts:** Second request for the same user (warm) will be much faster than first (cold).

---

### H3 — Pipelining is not working as intended

**Mechanism:** The code batches 38 Redis commands into one `pipeline.execute()`, but if redis-py is falling back to individual round-trips (wrong API usage, connection issue, or version mismatch), the pipeline provides no benefit and each of the 38 commands pays a full RTT.

**Assumptions:**
- `redis.pipeline(transaction=False)` actually sends all commands in one TCP write
- Redis MONITOR would show all 38 commands arriving in a single batch, not staggered

**Predicts:** Redis MONITOR shows commands arriving in tight bursts (pipeline working) vs spread over time (pipeline broken).

---

### H4 — Another hidden bottleneck (connection pooling, DNS, autocommit, logging, disk sync)

**Possible causes:**
- **Connection pool exhaustion:** At 150 req/s with slow requests, all 20 pool connections are held simultaneously. New requests block waiting for a free connection.
- **DNS resolution:** Container hostname resolution (`redis`, `db`) adds latency on first lookup if not cached.
- **SQLAlchemy autocommit / implicit transactions:** Extra round-trips from transaction management overhead.
- **Logging I/O:** JSON log writes to stdout are synchronous; under load they may block.
- **Disk sync on Postgres:** Docker Desktop on Mac runs Postgres with a virtual disk; fsync on commit is slow through the VM storage layer.

**Assumptions:** One or more of these is adding latency independent of the network path.

**Predicts:** Some of these would affect native runs too; others (disk sync, DNS) are Docker-specific.

---

## Test Plans

### Test 1 — Docker vs native API latency (tests H1, H4)

Isolates whether the bottleneck is Docker environment vs the application code.

**Steps:**

1. Measure latency with API in Docker (current state):
```bash
curl -s -o /dev/null -w "%{time_total}s\n" \
  -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: test-docker-$(date +%s)" \
  -d '{"user_id":999888777,"purchase_value":100,"device_id":"t1","ip_address":3627734734,"signup_time":"2023-01-01T00:00:00","purchase_time":"2024-01-01T00:00:00","source":"SEO","browser":"Chrome","sex":"M","age":30}'
```

2. Stop the Docker api service, run natively (Redis/Postgres still in Docker):
```bash
docker compose stop api
source .venv/bin/activate
uvicorn app.main:app --port 8000 &
# repeat same curl with a new Idempotency-Key
```

**Interpret:**
- Native 10x+ faster → Docker VM overhead is the primary cause (H1/H4 confirmed)
- Native similarly slow → bottleneck is in the code, not the environment

---

### Test 2 — Redis RTT inside Docker vs host native (tests H1)

Directly measures the Docker virtual network tax on Redis.

**Inside container:**
```bash
docker compose exec api python -c "
import asyncio, time
from redis.asyncio import Redis
async def measure():
    r = Redis(host='redis', port=6379, decode_responses=True)
    await r.ping()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        await r.ping()
        times.append((time.perf_counter() - t0) * 1000)
    await r.aclose()
    times.sort()
    print(f'Redis RTT inside Docker: p50={times[10]:.2f}ms  p95={times[19]:.2f}ms')
asyncio.run(measure())
"
```

**From host:**
```bash
source .venv/bin/activate
python -c "
import asyncio, time
from redis.asyncio import Redis
async def measure():
    r = Redis(host='localhost', port=6379, decode_responses=True)
    await r.ping()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        await r.ping()
        times.append((time.perf_counter() - t0) * 1000)
    await r.aclose()
    times.sort()
    print(f'Redis RTT host native: p50={times[10]:.2f}ms  p95={times[19]:.2f}ms')
asyncio.run(measure())
"
```

**Interpret:**
- Inside Docker > 10ms → virtual network overhead confirmed
- Host native < 2ms → confirms the environment is the cause, not Redis itself

---

### Test 3 — Postgres query RTT inside Docker vs host native (tests H1, H4)

Same as Test 2 but for Postgres. Also catches disk sync overhead.

**Inside container:**
```bash
docker compose exec api python -c "
import asyncio, time
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
async def measure():
    engine = create_async_engine('postgresql+asyncpg://postgres:postgres@db/fraud')
    async with engine.connect() as conn:
        await conn.execute(text('SELECT 1'))
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            await conn.execute(text('SELECT 1'))
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f'Postgres RTT inside Docker: p50={times[10]:.2f}ms  p95={times[19]:.2f}ms')
asyncio.run(measure())
"
```

**From host:** same script, change URL host from `db` to `localhost`.

**Interpret:**
- Both environments slow → disk sync or query overhead (not networking)
- Inside slow, outside fast → VM network is the cause

---

### Test 4 — Count actual Redis round-trips per request (tests H3)

Verifies pipelining is working as intended.

```bash
# Terminal 1 — watch all Redis commands in real time
docker compose exec redis redis-cli monitor

# Terminal 2 — send exactly one request with a unique user never seen before
curl -s -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: monitor-$(date +%s)" \
  -d '{"user_id":111222333,"purchase_value":100,"device_id":"monitor-d1","ip_address":3627734734,"signup_time":"2023-01-01T00:00:00","purchase_time":"2024-01-01T00:00:00","source":"SEO","browser":"Chrome","sex":"M","age":30}'
```

Count lines in MONITOR output between request start and finish.

**Interpret:**
- ~5–7 batches of commands arriving close together → pipeline working (expected: 2 GETEX pipeline + SET + 38-command pipeline + SET + ZADDs)
- 40+ individual commands staggered over time → pipeline broken (H3 confirmed)

---

### Test 5 — Warm cache vs cold cache, same user (tests H2)

Isolates the cost of DB fallbacks by comparing first vs second request for the same user.

```bash
# Cold — user never seen before
time curl -s -o /dev/null \
  -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: cold-$(date +%s)" \
  -d '{"user_id":444555666,"purchase_value":100,"device_id":"warm-d1","ip_address":3627734734,"signup_time":"2023-01-01T00:00:00","purchase_time":"2024-01-01T00:00:00","source":"SEO","browser":"Chrome","sex":"M","age":30}'

# Warm — same user_id, different idempotency key (forces full pipeline, skips idempotency cache)
time curl -s -o /dev/null \
  -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: warm-$(date +%s)" \
  -d '{"user_id":444555666,"purchase_value":100,"device_id":"warm-d1","ip_address":3627734734,"signup_time":"2023-01-01T00:00:00","purchase_time":"2024-01-01T00:00:00","source":"SEO","browser":"Chrome","sex":"M","age":30}'
```

**Interpret:**
- Warm significantly faster (>2x) → cold cache fallbacks are a major cost (H2 confirmed)
- Warm and cold similar → bottleneck is in the warm path itself (pipelines, model, or DB write), not the fallbacks

---

## Decision Tree

```
Test 2: Redis RTT inside Docker > 10ms?
  ├── YES → Docker virtual network is a real factor
  │         Run API natively (Test 1) to confirm magnitude
  └── NO  → Networking is not the cause; focus on H3 and H4

Test 5: Warm cache >> Cold cache latency?
  ├── YES → DB fallbacks are dominant; Redis TTLs or pre-seeding would help
  └── NO  → Bottleneck is in the warm path itself
            Check Test 4 (pipeline) and connection pool exhaustion (H4)

Test 4: Redis MONITOR shows > 6 round-trip batches?
  ├── YES → Pipelining broken (H3); check redis-py version and pipeline API usage
  └── NO  → Pipelining working as designed; per-RTT cost is the issue, not RTT count

Test 1: Native API similar speed to Docker?
  ├── YES → Bottleneck is in application code (check H4: pool exhaustion, logging)
  └── NO  → Docker VM overhead is the primary cause; native deployment resolves it
```
