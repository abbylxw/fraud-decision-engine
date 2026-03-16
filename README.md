# Fraud Decision Engine

Real-time fraud detection API for e-commerce transactions. Receives a transaction via HTTP POST,
runs rule-based and ML-based checks in sequence, writes the result to PostgreSQL, and returns
a fraud decision — all within a single async request.

## Architecture

```
POST /api/transactions
        │
        ▼
 [Hard Rules]  ←── purchase limit, age check, IP country block
        │  deny → short-circuit (model never runs)
        ▼
 [ML Model]    ←── RandomForest on ~100 velocity + derived features
        │
        ▼
 [Soft Rules]  ←── rate limiter, model threshold
        │
        ▼
 PostgreSQL (audit log) + Redis (velocity counters update, fire-and-forget)
        │
        ▼
   { transaction_id, fraud_decision, response_time }
```

Services: **FastAPI** API · **PostgreSQL 16** · **Redis 7**

## Quick Start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

The first start loads ~5M IP→country rows into PostgreSQL automatically. This takes ~30 seconds.

## Quick Start (Local)

Requires PostgreSQL and Redis running on localhost.

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

## Example Request

```bash
curl -X POST http://localhost:8000/api/transactions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: test-001" \
  -d '{
    "user_id": "user_123",
    "signup_time": "2024-01-01T00:00:00Z",
    "purchase_time": "2025-03-14T12:00:00Z",
    "purchase_value": 250.00,
    "device_id": "device_abc",
    "source": "SEO",
    "browser": "Chrome",
    "sex": "M",
    "age": 30,
    "ip_address": 3663237372,
    "class": 0
  }'
```

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "fraud_decision": "allow",
  "response_time": "2025-03-14T12:00:01.234Z"
}
```

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "db": "ok", "redis": "ok"}
```

## Load Testing

```bash
python app/data/client_mimic.py   # 50 RPS for 120s, cycles through Fraud_Data.csv
```

## Running Tests

```bash
pytest tests/ -v
```

## Interesting Engineering Decisions

**Async throughout** — `AsyncSession` + asyncpg for PostgreSQL, `redis.asyncio` for Redis.
No blocking calls on the event loop. ML model inference (CPU-bound) runs in a `ThreadPoolExecutor`
via `loop.run_in_executor`, keeping latency predictable under concurrent load.

**Sliding-window velocity counters** — Redis sorted sets keyed by `(user_id, device_id)` etc.,
scored by Unix timestamp. Each window query is a `ZRANGEBYSCORE` between `now - window` and `now`.
90-day pruning keeps memory bounded. Counters are updated fire-and-forget after the response
is returned, so they don't add to request latency.

**Atomic idempotency** — `SET NX EX 600` ensures only one response is ever stored per
`Idempotency-Key`, even under concurrent duplicate requests. Avoids the GET→check→SET
TOCTOU race condition.

**Read-through cache** — On a Redis cache miss (cold start, TTL expiry), the app reconstructs
velocity counts from PostgreSQL `COUNT`/`SUM` queries and continues without error.

**Two-phase rules pipeline** — Hard rules (no model needed) short-circuit before inference.
Soft rules (velocity anomalies, model threshold) run after. This avoids model overhead for
clear-cut cases and keeps rule logic separated from model logic.

See [docs/design_tradeoffs_and_todos.md](docs/design_tradeoffs_and_todos.md) for full trade-off notes.

## Project Structure

```
app/
  main.py              # FastAPI app, endpoint, lifespan, /health
  config.py            # pydantic-settings (REDIS_HOST, DATABASE_URL, etc.)
  data/
    data_models.py     # Pydantic models: Transaction → Features → DBWrite
    orm_models.py      # SQLAlchemy ORM: TransactionRecord
    client_mimic.py    # Load test client (50 RPS)
    db_init/           # SQL init scripts (auto-run by postgres on first start)
  decisions/
    rules.py           # Rule functions (hard + soft)
    models.py          # ML inference (RandomForest, loads model.joblib at startup)
    features/          # Feature extraction: request.py, db.py, velocity.py
tests/
  test_rules.py        # Unit tests for each rule function
  test_features.py     # Unit tests for request-based feature calculations
  test_api.py          # Integration tests for /health and /api/transactions
docs/
  design_tradeoffs_and_todos.md   # Architecture decisions + known TODOs
  system_architecture.md
```
