# Fraud Decision Engine

Real-time fraud detection API for e-commerce transactions. Receives a transaction via HTTP POST,
runs rule-based and ML-based checks in sequence, writes the result to PostgreSQL, and returns
a fraud decision — all within a single async request.

## Architecture

```
POST /api/transactions
        │
        ▼
 [Hard Rules]  ←── purchase limit, signup burst (<60s), age check, IP country block
        │  deny → short-circuit (model never runs)
        ▼
 [ML Model]    ←── XGBoost on 25 velocity + derived features (see Model Training)
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
python app/data/client_mimic.py   # 400 RPS for 120s, cycles through Fraud_Data.csv
```

## Running Tests

```bash
pytest tests/ -v
```

## Model Evaluation

```bash
python -m training.evaluate          # ~30 s; writes results/<timestamp>/metrics.json + charts
```

Time-based split (train Jan–Aug 5 / validation Aug 5–30 / test Aug 30–Dec 16). Early stopping
and threshold choice use validation only; the test period is scored once. Full write-up:
[notebooks/evaluation_report.ipynb](notebooks/evaluation_report.ipynb).

| Model (test period, fraud rate 4.6%) | PR-AUC [95% CI] | ROC-AUC |
|---|---|---|
| Random guessing | 0.046 | 0.50 |
| Previous RandomForest (notebook, 52 features) | 0.091 [0.080, 0.105] | 0.56 |
| **Served XGBoost** (`training/train.py`, 25 features) | **0.092 [0.081, 0.106]** | 0.56 |
| XGBoost, early-stopped on validation | 0.088 [0.079, 0.101] | 0.56 |
| Logistic regression | 0.049 [0.046, 0.054] | 0.50 |

**What the evaluation found:**
- **The previous model barely beat random guessing.** At its hardcoded threshold of 0.7 it
  flagged 10 of 30,223 test transactions and caught 4 of 1,389 frauds.
- **25 of the 52 served features never change.** Every user in the dataset has exactly one
  transaction, so every per-user counter is always 0.
- **January was a bot attack** (76% fraud; 7,600 purchases within 1 second of signup), and that
  pattern never appears in the test period. Training without it barely changes the tree models.
- **`month_of_year` is XGBoost's 3rd most important feature,** but test months (Sep–Dec) never
  appear in training, so it learns "January = fraud" and nothing that carries forward.
- **A cost-based threshold** (a $5 review per alert vs. losing the purchase value of missed
  fraud), chosen on validation, saves $3.1k out of $51.4k of test-period fraud, against $115
  at the current 0.7.

## Model Training

```bash
python -m training.train             # ~15 s; train + report to results/train_<timestamp>/
python -m training.train --ship      # also replace app/decisions/model.joblib + metadata
```

What the served model does differently from the notebook model, and why:

- **The January bot attack is a rule, not a model feature.** `signup_burst` denies purchases
  made less than 60 s after signup (all 7,600 such rows are fraud; the fastest legitimate one is
  137 s). Those rows are excluded from training because the model never scores them in production.
- **`month_of_year` and the 26 features that are constant on the training rows are dropped**
  (52 → 25). Nothing measurable is lost on test; the model now relies on device sharing
  (`did_uid_90d`, `did_uid_30d`) instead of the calendar.
- **Shallow XGBoost** (depth 3, 146 trees). With the notebook's depth-7 settings, early stopping
  fired after 3 trees because the signal left after January is weak and deep trees overfit it.
- **The threshold is chosen by cost on validation** (missed fraud = purchase value, false decline
  = $10) and stored in `app/decisions/model_meta.json` with the model version, training window
  and evaluation results. The API reads it from there; it is no longer hardcoded.
- **A ship gate:** `--ship` refuses unless test PR-AUC is at least the previous model's 95% CI
  lower bound.

Test period (Aug 30–Dec 16, scored once):

| | Denied | Fraud caught | False declines | Fraud $ stopped | Total cost* |
|---|---|---|---|---|---|
| Deny nothing | 0 | 0 / 1,389 | 0 | $0 | $51.4k |
| Previous model @ 0.7 | 10 | 4 | 6 | $165 | $51.3k |
| **Served model @ 0.849** | 327 (1.1%) | **98** | 229 | **$4,059** | **$49.6k** |

\* missed-fraud dollars + $10 per false decline. The ranking quality (PR-AUC) is unchanged;
the gain comes from a threshold chosen for this model by cost instead of a hardcoded 0.7.
On this dataset the ceiling is the data: after January, fraud is 4.5% and the main
remaining signal is device sharing.

## Interesting Engineering Decisions

**Async throughout** — `AsyncSession` + asyncpg for PostgreSQL, `redis.asyncio` for Redis.
No blocking calls on the event loop. ML model inference (CPU-bound) runs in a `ThreadPoolExecutor`
via `loop.run_in_executor`, keeping latency predictable under concurrent load.

**Sliding-window velocity counters** — Redis sorted sets keyed by `(user_id, device_id)` etc.,
scored by Unix timestamp. Count-only windows use `ZCOUNT` (returns an integer, not the full member
list). Purchase windows use `ZRANGE WITHSCORES` to fetch amounts for sum/median calculation.
All 38 Redis commands per request are batched into a single pipeline — one TCP round-trip.
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

**Multi-worker scaling** — 8 uvicorn workers, each with its own async event loop. The per-worker
ceiling is ~65 RPS (event loop callback queue saturation). Total ceiling: ~520 RPS on a 10-core
machine. DB connection pool sized at `pool_size=5, max_overflow=6` per worker (88 total
connections, within Postgres's default `max_connections=100`).

See [docs/design_tradeoffs_and_todos.md](docs/design_tradeoffs_and_todos.md) for full trade-off notes.
See [docs/performance_investigation.md](docs/performance_investigation.md) for load test methodology and findings.

## Project Structure

```
app/
  main.py              # FastAPI app, endpoint, lifespan, /health
  config.py            # pydantic-settings (REDIS_HOST, DATABASE_URL, etc.)
  data/
    data_models.py     # Pydantic models: Transaction → Features → DBWrite
    orm_models.py      # SQLAlchemy ORM: TransactionRecord
    client_mimic.py    # Load test client (400 RPS)
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
