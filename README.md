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
 [ML Model]    ←── XGBoost on 41 features: device/IP sharing counters, transaction fields,
        │            grouped IP country (one-hot)
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
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip          # dependency groups need pip >= 25.1
pip install --group dev            # API + tests; use --group train for training/notebooks
uvicorn app.main:app --reload --port 8000
```

Dependencies are declared in `pyproject.toml` as three groups: `api` (what the Docker image
installs), `train` (adds notebooks, torch, dataset download) and `dev` (adds pytest, ruff).

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
python app/data/client_mimic.py --source sparkov   # repeat customers (needs data/external/sparkov/)
python app/data/client_mimic.py --rps 50 --duration 60   # choose the rate and length
```

The summary prints the target and the achieved RPS (successful responses per second), plus
p50/p95/p99 latency.

`--source sparkov` replays Sparkov card transactions (~100 purchases per card), so the per-user
Redis windows fill up the way they would in production. Sparkov has no device, IP, source,
browser or signup time, so the client makes them up: use it for latency and load, not to judge
the model's decisions.

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
| XGBoost (`training/train.py`, 25 features, alphabetical country code) | 0.092 [0.081, 0.106] | 0.56 |
| **Served XGBoost** (41 features, grouped country one-hot) | **0.092 [0.081, 0.107]** | 0.56 |
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
- **`month_of_year` and the features that are constant on the training rows are dropped.**
  Nothing measurable is lost on test; the model relies on device sharing (`did_uid_90d`,
  `did_txn_cnt_30d`) instead of the calendar.
- **IP country is grouped and one-hot encoded.** It used to be an alphabetical index over 182
  countries fed in as a number, so the model saw a fake order (Albania < Zimbabwe). Now: the 10
  highest-volume countries (~80% of traffic) each get a column, `Unknown` (IP not in the
  table) gets one, and every other country falls into its continent
  (`app/decisions/country_groups.json`, 18 groups). One `country_group()` in
  `app/decisions/models.py` is used by both training and the API, and
  `tests/test_encoding_contract.py` checks all 182 names. PR-AUC is unchanged (per-country fraud
  rates are nearly flat, 3.8–5.0%), so the gain is a correct, explainable feature rather than
  accuracy.
- **Shallow XGBoost** (depth 3, 203 trees). With the notebook's depth-7 settings, early stopping
  fired after 3 trees because the signal left after January is weak and deep trees overfit it.
- **The threshold is chosen by cost on validation** (missed fraud = purchase value, false decline
  = $10) and stored in `app/decisions/model_meta.json` with the model version, training window
  and evaluation results. The API reads it from there; it is no longer hardcoded.
- **"Elevated score" for the rate-limiter combos is model-relative.** The soft rules that
  combine a score with velocity (e.g. score + 5 purchases in 24h) used a fixed 0.5. That value
  is meaningless for an uncalibrated model (the served XGBoost's median score is ~0.47), so the
  level is now the top 3% of validation scores (0.591; 24% precision on validation, 23% on
  test), stored in `model_meta.json`. The velocity cut-offs themselves are expert-set and
  untested: with one transaction per user, none of those branches can fire on this dataset.
- **Every decision is auditable.** The `transactions` table records `model_version`,
  `model_score` (NULL when a hard rule denied first) and `decision_rule`
  (`app/data/db_init/005_add_model_audit_columns.sql`; idempotent, apply by hand to an
  existing database).
- **A ship gate:** `--ship` refuses unless test PR-AUC is at least the previous model's 95% CI
  lower bound.

Test period (Aug 30–Dec 16, scored once):

| | Denied | Fraud caught | False declines | Fraud $ stopped | Total cost* |
|---|---|---|---|---|---|
| Deny nothing | 0 | 0 / 1,389 | 0 | $0 | $51.4k |
| Previous model @ 0.7 | 10 | 4 | 6 | $165 | $51.3k |
| 25-feature model @ 0.849 | 327 (1.1%) | 98 | 229 | $4,059 | $49.6k |
| **Served model @ 0.843** (grouped country) | 365 (1.2%) | **106** | 259 | **$4,271** | **$49.7k** |

\* missed-fraud dollars + $10 per false decline. The ranking quality (PR-AUC) is unchanged;
the gain over 0.7 comes from a threshold chosen for this model by cost. The two cost-tuned
models are a wash: more fraud caught, more false declines, same total cost.
On this dataset the ceiling is the data: after January, fraud is 4.5% and the main
remaining signal is device sharing.

## Does the Velocity Layer Work? (Datasets With Repeat Users)

Fraud_Data.csv can't answer this: every user has exactly one transaction, so per-user
counters are always 0. `training/compare_datasets.py` runs the **same velocity replay**
(`training/build_features.velocity_features`, the code the parity test checks against the
live API) on datasets where users repeat, and trains one fixed XGBoost configuration with and
without the velocity features. Time split 70/10/20, test scored once, 95% bootstrap CIs.

```bash
pip install --group train
python -m training.datasets.download     # needs a Kaggle API token
python -m training.compare_datasets      # ~5 min; caches features in results/cache/
```

| Dataset (test period) | Fraud rate | Static features | + Velocity | + Spend spike | Velocity features that vary |
|---|---|---|---|---|---|
| Fraud_Data (151K txns, 1 per user) | 4.6% | PR-AUC 0.050 [0.045, 0.055] | 0.095 [0.084, 0.110] | 0.086 [0.079, 0.098]† | 20 / 54 |
| **Sparkov** (1.85M txns, 999 cards) | 0.36% | PR-AUC 0.878 [0.864, 0.891] | **0.974 [0.967, 0.980]** | 0.974 [0.967, 0.979] | 54 / 54 |

† All 9 spend-spike features are constant on Fraud_Data, so this move is run-to-run noise
(XGBoost samples columns at random); it shows roughly how large a difference has to be to mean
anything here.

On Sparkov, at a cost-chosen threshold ($5 per review, missed fraud = purchase value),
velocity features **halve the alert rate (0.94% → 0.43%) while catching more fraud
(1,265 → 1,296 of 1,349)**, cutting total cost from $29.3k to $21.4k (−27%). The top features
shift from amount / hour / category to `purchase_amt_24h`, `high_amount_vs_hist` and
`purchase_amt_1h`: spending relative to the card's own recent history.

**Spend-spike features** (`build_features.spend_spike_features`, training only): today's
spend (UTC day, including the current purchase) divided by the median daily total over the
card's active days in the last 14 / 30 / 90 days (at least 5 active days, else "no
baseline"), plus 3× and 5× flags: 9 features. On their own they are strong signals on the
Sparkov test period:

| Flag | Fires on | Fraud rate when on | when off |
|---|---|---|---|
| `spend_spike_3x_30d` | 7.8% of transactions | 3.4% | 0.11% |
| `spend_spike_5x_30d` | 2.6% | 8.6% | 0.15% |

But they **don't improve the model**: PR-AUC stays at 0.974 with the same CI. The model does
pick them up (`spend_spike_3x_90d` becomes the #3 feature) largely in place of
`purchase_amt_24h` and `high_amount_vs_hist`, which already encode "more than usual". Total
cost at the cost-chosen threshold is lower ($21.4k → $18.9k), but that comes from 4 extra
frauds caught at one threshold and has no CI, so it isn't evidence of a gain. Verdict: not
worth porting to the live API (it would need a daily-totals structure in Redis) unless it helps
on real data, though the flags are a readable basis for a rule or a decline reason.

Honest caveats:
- **Sparkov is simulated.** Static features alone already reach 0.88 PR-AUC; real fraud is
  harder, so the size of the gain on real traffic is still an open question.
- **Datasets without devices or IPs use proxies** (Sparkov: merchant as "device", home zip as
  "IP", merchant category as "country"); mappings are documented in `training/datasets/`.
- **Label-history features are excluded from the headline numbers.** `purchase_success_*`
  counts a user's past *non-fraud* purchases, which the replay knows instantly but production
  only learns weeks later from chargebacks. Adding them lifts Sparkov to 0.991 and makes
  `purchase_success_cnt_24h` one of the top features: an upper bound, not a deployable result.

Full numbers: [docs/evaluation/multi_dataset.json](docs/evaluation/multi_dataset.json).

## Interesting Engineering Decisions

**Async throughout** — `AsyncSession` + asyncpg for PostgreSQL, `redis.asyncio` for Redis.
No blocking calls on the event loop. ML model inference (CPU-bound) runs in a `ThreadPoolExecutor`
via `loop.run_in_executor`, keeping latency predictable under concurrent load.

**Sliding-window velocity counters** — Redis sorted sets keyed by `(user_id, device_id)` etc.,
scored by Unix timestamp. Count-only windows use `ZCOUNT` (returns an integer, not the full member
list). Purchase windows use `ZRANGE WITHSCORES` to fetch amounts for sum/median calculation.
All 38 Redis commands per request are batched into a single pipeline — one TCP round-trip.
90-day pruning keeps memory bounded. Counters are updated fire-and-forget after the DB commit
is returned, so they don't add to request latency.

**Atomic idempotency** — the `Idempotency-Key` is reserved with `SET NX` *before* the
pipeline runs, so concurrent duplicates can't both write an audit row or double-count
velocity. A duplicate that arrives mid-flight gets `409` + `Retry-After: 1`; one that arrives
later gets the stored response (60 s TTL); a request that fails releases the key so the
client can retry. Tested with real concurrency in `tests/test_idempotency.py`.

**Tracked background tasks** — the velocity update runs after the DB commit (a failed audit
write never counts toward velocity), the app holds a reference to each task so it can't be
garbage-collected, failures are logged and counted in `fraud_background_task_failures_total`,
and shutdown waits up to 5 s for in-flight updates.

**Read-through cache** — When Redis has no purchase history for a user (cold start, eviction),
the app rebuilds the purchase counts from PostgreSQL `COUNT`/`SUM` queries and continues without
error. IP-to-country and device-age lookups are cached in Redis in front of Postgres.

**Training-serving parity test** — Velocity windows count back from the transaction's own
`purchase_time` (event time), exactly like the training replay, not from the server clock.
`tests/test_feature_parity.py` sends the same transactions through the live feature code
(against an in-memory fake Redis) and through `training/build_features.py`, and fails if any
numeric feature differs. Written before the fix, it failed on about 50 features. It also caught
two smaller counter bugs: same-second collisions and an off-by-one in the 90-day purge.
See trade-off #12 in the design doc.

**Two-phase rules pipeline** — Hard rules (no model needed) short-circuit before inference.
Soft rules (velocity anomalies, model threshold) run after. This avoids model overhead for
clear-cut cases and keeps rule logic separated from model logic.

**Multi-worker scaling** — 8 uvicorn workers, each with its own async event loop, and a DB
connection pool of `pool_size=5, max_overflow=6` per worker (88 total connections, within
Postgres's default `max_connections=100`). In April this reached ~520 RPS. The November
re-measurement found ~50 RPS for repeat customers and under 30 RPS for new users. The cause is
a missing index on the audit table's `user_id`: the Redis-miss fallback scans the whole table,
and the table grows with every load test. Fix and numbers:
[docs/performance_2026-11.md](docs/performance_2026-11.md).

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
    models.py          # ML inference (loads model.joblib + model_meta.json at startup)
    features/          # Feature extraction: request.py, db.py, velocity.py
tests/
  test_rules.py        # Unit tests for each rule function
  test_features.py     # Unit tests for request-based feature calculations
  test_feature_parity.py   # Live feature code vs training feature code, row by row
  test_velocity_fallback.py  # Redis counters + Postgres fallback (fake Redis)
  test_encoding_contract.py  # Category encodings match between training and serving
  test_api.py          # Integration tests for /health and /api/transactions
  test_datasets.py     # External-dataset loaders: column mapping, no future leakage
  test_spend_spike.py  # Today's spend vs median daily spend: windows, day boundary, baseline
training/
  build_features.py    # Feature replay used for training (shared encodings)
  train.py             # Retrain, evaluate, ship gate (--ship replaces the served model)
  evaluate.py          # Model comparison with bootstrap CIs
  compare_datasets.py  # Velocity ablation across Fraud_Data and Sparkov
  datasets/            # Kaggle download + per-dataset loaders (training/eval only)
docs/
  design_tradeoffs_and_todos.md   # Architecture decisions + known TODOs
  system_architecture.md
```
