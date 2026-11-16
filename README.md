# Fraud Decision Engine

Real-time fraud detection API for e-commerce transactions. Receives a transaction via HTTP POST,
runs rule-based and ML-based checks in sequence, writes the result to PostgreSQL, and returns
a fraud decision — all within a single async request.

## Architecture

```
POST /api/transactions
        │
        ▼
 [Hard Rules]  ←── purchase limit, signup burst (<60s), 2+ users on a device in 1h, age check, IP country block
        │  deny → short-circuit (model never runs)
        ▼
 [ML Model]    ←── XGBoost on 33 features: device/IP sharing counters, transaction fields,
        │            grouped IP country (one-hot)
        │
        ▼
 [Soft Rules]  ←── model score: allow / review / deny (two cost-based thresholds);
        │            rate limiter in shadow mode
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
    "ip_address": 3663237372
  }'
```

```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "fraud_decision": "allow",
  "response_time": "2025-03-14T12:00:01.234Z"
}
```

`fraud_decision` is one of:

| Value | Meaning for the caller |
|---|---|
| `allow` | Approve the order. |
| `review` | Do not auto-approve: run a step-up check (3-D Secure, SMS code) or queue it for a person. Returned when the model score is between the review and deny thresholds. |
| `deny` | Decline. A hard rule fired, or the score is high enough that a decline pays for itself. |

Both thresholds are chosen by cost on validation data and stored in
`app/decisions/model_meta.json` ([docs/feature_ablation.md](docs/feature_ablation.md) §7).

## Demo

```bash
docker compose up -d --build
python scripts/demo.py            # eight stories, then ~2,000 replayed transactions
python scripts/alert_drill.py     # bot burst -> alert pending -> firing -> Alertmanager (~3 min)
```

The script sends real requests and prints each decision with the reason, read back from
the audit table: a normal customer (allow), a bot buying one second after signup (deny),
a second account on one device within the hour (deny), a sanctioned country (deny), a
device returning after 120 days with a different account (review), a loyal customer's third
purchase on their own device (allow), a device that six other accounts used months apart
(review), and a retried request answered from the idempotency cache. The replay then fills the Grafana dashboard
(http://localhost:3000, row "Decisions and model"): decision mix, review rate, decisions
by rule, the score distribution against the served thresholds, and the share of traffic
from devices that other accounts have used before.

A one-page tour for readers who will not run anything is in `docs/demo/tour.html` (open it
in a browser): the request flow, the eight stories, the rolling-origin chart with and
without the device feature, and a cost console that re-picks the review and deny thresholds
for any decline cost, review cost and catch rate. It is built from the evaluation outputs by
`python -m scripts.export_demo_data`, which writes aggregated curves only, no transaction
rows.

## Monitoring and Alerting

```bash
docker compose up -d --build     # API, Postgres, Redis, Prometheus, Alertmanager, Grafana
```

| | Address | What is there |
|---|---|---|
| Grafana | http://localhost:3000 | Dashboard "Fraud Engine": traffic and latency, "Decisions and model", "Health and drift" |
| Prometheus | http://localhost:9090 | Rules and alert states (`/alerts`, `/rules`) |
| Alertmanager | http://localhost:9093 | Grouped, routed alerts |

Four layers, each with metrics, a dashboard row and alerts:

| Layer | What is watched |
|---|---|
| **Four golden signals** | **Latency** (p50, p99, per step) · **Traffic** (rate against its 1-day average, requests in flight) · **Errors** (our 5xx, invalid 422s, duplicate 409s) · **Saturation** (worker CPU, event-loop lag, Postgres pool and connections, Redis memory) |
| **Dependencies and feature data** | Postgres and Redis probes, Redis key count, features falling back to Postgres, failed counter updates |
| **Action rates** | Share of allow / review / deny against the 1-day normal and the training-time expectation; share per rule |
| **Model and feature drift** | PSI of the score and nine model inputs against buckets saved at training time; shared-device share; model version |

29 alerts, each with a severity, a minimum-traffic guard and a runbook section in
[docs/monitoring.md](docs/monitoring.md). The rules are unit-tested with `promtool` in CI
(19 scenarios), and `scripts/alert_drill.py` proves the chain end to end on the running
stack: a bot burst is denied by the rule, detected by Prometheus and delivered to
Alertmanager.

Decision accuracy is deliberately not monitored live: live traffic has no labels, so action
rates are the proxy, and accuracy is measured offline on labelled history at every retrain.
No notification channel is configured in the repository (a webhook is a secret).

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "db": "ok", "redis": "ok"}
```

## Load Testing

```bash
python app/data/client_mimic.py   # rate mode: mean 400 RPS for 120s, cycles through Fraud_Data.csv
python app/data/client_mimic.py --source sparkov   # repeat customers (needs data/external/sparkov/)
python app/data/client_mimic.py --rps 50 --duration 60 --seed 1   # choose the rate; repeatable
python app/data/client_mimic.py --speedup 1000 --duration 3600    # replay mode: realistic traffic
```

Two modes, for two different questions:

- **Rate mode (`--rps`)** answers "how much load can the API take?". Arrivals are Poisson
  (random gaps with mean 1/RPS), because an evenly spaced drip never piles up and so understates
  tail latency. Rows keep their 2015 timestamps and repeat after one pass through the file, so
  the velocity features drift (the same user "buys again" at the same second). Use it for
  throughput and latency, not to judge decisions.
- **Replay mode (`--speedup K`)** answers "what does the system do under continuous, realistic
  traffic?". Rows are sorted by `purchase_time` and sent when a virtual clock running K times
  faster than wall time reaches them. When the file ends, every timestamp (purchase and signup)
  is shifted forward by the file's span, so time never goes backwards. At `--speedup 1000`,
  Fraud_Data's 8 months replay in ~6 hours at ~6 req/s, which is this dataset's honest rate:
  151k transactions over 8 months is 0.006/s, so sending 400/s with realistic timestamps would
  give country-level velocity counts in the thousands, which the model never saw.
  - **Returning vs new users.** On the second pass and later, each row is either the same
    customer coming back (same ids, signup kept, so the account has aged by ~8 months) or, with
    probability `--new-user-share` (default 0.2), a brand-new signup with fresh user and device
    ids. Returning users' `days_since_signup` then exceeds anything in the training data; that
    is realistic, and a good way to see how the model behaves on it.
  - **Retries after a decline, `--retry-declined N`.** A declined customer tries again with
    probability `--retry-prob` (default 0.5): a new transaction from the same user, device and
    IP, 15 s to 10 min later, for a smaller or equal amount, up to N attempts, stopping at the
    first allow. Exercises the velocity windows and the rules the way real repeat attempts do.
    Example: `python app/data/client_mimic.py --speedup 1000 --duration 3600 --retry-declined 10`.

The summary prints the achieved RPS (successful responses per second), the error count by kind
(`ReadTimeout`, `409`, ...), the allow/deny split, retries sent, and p50/p95/p99 latency over
every completed request.

One client process tops out around 400–500 RPS. For more load, run several in parallel: each
adds a random run id to its idempotency keys, so they never collide. Grafana's "Requests per
Worker" panel shows how the load spreads over the 8 uvicorn workers. Not implemented yet:
duplicate-key resends (the 409/replay path) and a daily traffic rhythm.

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
python -m training.evaluate          # ~2 min; writes results/<timestamp>/metrics.json + charts
```

Time-based split (train Jan–Aug 5 / validation Aug 5–30 / test Aug 30–Dec 16). Early stopping
and threshold choice use validation only; the test period is scored once. Full write-up:
[notebooks/evaluation_report.ipynb](notebooks/evaluation_report.ipynb).

| Model (test period, fraud rate 4.6%) | PR-AUC [95% CI] | ROC-AUC |
|---|---|---|
| Random guessing | 0.046 | 0.50 |
| Logistic regression | 0.094 [0.085, 0.106] | 0.63 |
| Random forest (depth 8, 200 trees) | 0.136 [0.122, 0.154] | 0.64 |
| XGBoost (depth 3, 80 trees, monotone in the device feature) | 0.140 [0.124, 0.157] | 0.64 |
| LightGBM (7 leaves, 110 trees) | 0.142 [0.126, 0.159] | 0.64 |
| CatBoost (depth 4, 110 trees) | 0.139 [0.123, 0.157] | 0.64 |
| **Served XGBoost** (33 features, refit on train + validation) | **0.140 [0.124, 0.158]** | 0.64 |
| Previously served XGBoost (41 features, no all-time device feature) | 0.092 [0.081, 0.107] | 0.56 |
| Original RandomForest (notebook, 52 features) | 0.091 [0.080, 0.105] | 0.56 |

All tree models are trained without the January bot rows, with the tree count chosen on
validation; hyper-parameters come from `python -m training.tune`. The four tree ensembles
tie: every confidence interval contains every other point estimate. How they differ and why
they tie here: [docs/model_candidates.md](docs/model_candidates.md).

**The algorithm did not matter; one feature did.** Before the last feature was added, the
same four algorithms tied at about 0.09, and a rolling-origin check
(`python -m training.rolling_eval`: train on the past, score the next month, seven times)
showed the tie holds over time (LightGBM and CatBoost vs XGBoost: −0.001 [−0.005, +0.002]).
Then one feature, `did_other_uid_all` (how many *other* accounts have ever used this device;
every earlier device counter forgot after 90 days), lifted all four together: the served
model went from 0.092 to 0.141 (paired difference +0.049 [+0.040, +0.059]). The feature
study, with the things that did not help, is in
[docs/feature_ablation.md](docs/feature_ablation.md).

**Then nine inputs were taken out, and nothing was lost.** Fraud_Data has no repeat
customers, so a feature can be right on it and wrong on real traffic. Live testing and the
drift alert each caught one; an audit of every input
([docs/feature_audit.md](docs/feature_audit.md)) found the rest: inputs that count a
customer's own purchases as device risk, a "ratio to your usual spend" that was really the
purchase amount, raw traffic-volume counters, and an account age that only grows in this
dataset and made the score distribution drift. The served model has 33 inputs instead of
42, the same accuracy (0.140), a score that never falls as more accounts share a device,
and a score distribution that stays put on unseen months (PSI at most 0.004).

**Better ranking was not automatically more money.** The fraud the new feature finds sits at
about 23% precision. With a $37 average fraud and $10 per wrongly declined customer, a
decline breaks even at 21% ($10 / ($10 + $37)), so those extra declines earn almost nothing:
behind allow/deny the better model caught four times the fraud for the same total cost. A $5
review breaks even at 13.5%. It ships
with a third decision, `review`, and both thresholds chosen by cost:

| Policy on the test period (30,223 orders, 1,389 frauds) | Reviewed | Denied by model | Fraud stopped | Legit orders touched | Total cost* |
|---|---|---|---|---|---|
| Do nothing | | | 0 | 0 | $51,395 |
| Previous model, deny ≥ 0.843 | 0 | 365 (1.2%) | 106 | 259 declined | $49,714 |
| **Served model, review ≥ 0.762** | 2,043 (6.8%) | 0 | 469 | 1,574 reviewed | **$44,305** |

\* missed-fraud dollars + $10 per false decline + $5 per reviewed order, a reviewed fraud
counted as stopped. Difference vs the previous model: −$5,409. (Measured in steps: the new
feature and review band, −$4,986 with a 95% interval of [−$6,348, −$3,778]; the later
robustness changes, monotone constraint and feature audit, each moved cost by a few hundred
dollars with intervals that include zero.) The $5 and the "review stops the fraud" are assumptions; the gain holds
for a review up to about $5 that stops at least 80% of what it sees
([sensitivity table](docs/feature_ablation.md)).

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

### Why these numbers look low next to other Fraud_Data projects

Public write-ups on this dataset report Random Forest precision ≈ 0.98, recall ≈ 0.54,
ROC-AUC ≈ 0.77. Those come from a **random** `train_test_split` with `device_id` label-encoded
and calendar parts (month, minute, second) as features. Rerunning that recipe here
(`python -m training.random_split_baseline`) reproduces it, and shows where the recall comes from:

| Same Random Forest, same features | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Random 80/20 split, all test fraud | 0.99 | 0.54 | 0.77 |
| … bot rows only (1,533 of 2,850 test frauds) | | **1.00** | |
| … non-bot fraud only | | **0.002** | 0.50 |
| Time split (train Jan–Jun, test Jul–Aug) | 0.00 | 0.00 | 0.50 |

The January bot is 7,600 rows, all fraud, 54% of all fraud in the file. A random split puts
one fifth of them in the test set, where the same device ids already appear with nine fraud rows
each in training; the forest memorises the device label and "month = 1" and catches exactly
those rows. Recall 0.54 is the bot's share of fraud. On every other fraud case the model is a
coin flip, and trained on the past and tested on the future it catches nothing.

This project splits by time, hands the bot to hard rules (`signup_burst`,
`device_many_users_1h`: 100% of bot rows, 0 legitimate rows hit) and scores the model only on
the diffuse 4.6% that remains, where PR-AUC 0.092 vs 0.046 random is twice chance, not a
memorised attack.

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
  Nothing measurable is lost on test; the model relies on device sharing
  (`did_other_uid_all`, `did_uid_30d`, `did_uid_90d`) instead of the calendar.
- **IP country is grouped and one-hot encoded.** It used to be an alphabetical index over 182
  countries fed in as a number, so the model saw a fake order (Albania < Zimbabwe). Now: the 10
  highest-volume countries (~80% of traffic) each get a column, `Unknown` (IP not in the
  table) gets one, and every other country falls into its continent
  (`app/decisions/country_groups.json`, 18 groups). One `country_group()` in
  `app/decisions/models.py` is used by both training and the API, and
  `tests/test_encoding_contract.py` checks all 182 names. PR-AUC is unchanged (per-country fraud
  rates are nearly flat, 3.8–5.0%), so the gain is a correct, explainable feature rather than
  accuracy.
- **Four tree candidates, one winner by validation PR-AUC.** Random forest, XGBoost, LightGBM
  and CatBoost, each tuned by a small validation-only grid (`python -m training.tune`). All
  boosters are shallow (depth 3 / 7 leaves): with the notebook's depth-7 settings early
  stopping fired after 3 trees, because the signal left after January is weak and deep trees
  overfit it. XGBoost stays served unless a challenger beats its 95% CI on test; see
  [docs/model_candidates.md](docs/model_candidates.md) for how the four differ and why they tie.
- **Two thresholds are chosen by cost on validation** (missed fraud = purchase value, false
  decline = $10, review = $5): scores above the review threshold get `review`, scores above
  the deny threshold get `deny`. Among policies within 1% of the cheapest, the one that
  touches the fewest orders wins (`training/policy.py`). Both are stored in
  `app/decisions/model_meta.json` with the model version, training window and evaluation
  results; the API reads them from there. For the served model the deny threshold is "never":
  no score is confident enough for a decline to beat a review, so declines come from the
  hard rules.
- **"Elevated score" for the rate-limiter combos is model-relative.** The soft rules that
  combine a score with velocity (e.g. score + 5 purchases in 24h) used a fixed 0.5. That value
  is meaningless for an uncalibrated model (the served XGBoost's median score is ~0.47), so the
  level is now the top 3% of validation scores (0.823; 24% precision on validation, 26% on
  test), stored in `model_meta.json`. The velocity cut-offs themselves are expert-set and
  untested: with one transaction per user, none of those branches can fire on this dataset.
- **Every decision is auditable.** The `transactions` table records `model_version`,
  `model_score` (NULL when a hard rule denied first) and `decision_rule`
  (`app/data/db_init/005_add_model_audit_columns.sql`; idempotent, apply by hand to an
  existing database).
- **Upgrading an existing database:** `app/data/db_init/008_add_device_all_time_count.sql`
  and `009_add_device_other_accounts.sql` add two audit columns and a
  `(device_id, purchase_time)` index. Fresh Docker volumes get them automatically; an
  existing database needs both applied by hand before starting this version of the API, or
  every audit insert fails on the missing columns.
- **A ship gate with two conditions:** `--ship` refuses unless test PR-AUC is at least the
  previous model's 95% CI lower bound **and** the new policy costs no more on test than the
  served one. The second condition was added after a model passed the first while being
  worth $0.

Test period (Aug 30–Dec 16, scored once):

| | Denied | Fraud caught | False declines | Fraud $ stopped | Total cost* |
|---|---|---|---|---|---|
| Deny nothing | 0 | 0 / 1,389 | 0 | $0 | $51.4k |
| Previous model @ 0.7 | 10 | 4 | 6 | $165 | $51.3k |
| 25-feature model @ 0.849 | 327 (1.1%) | 98 | 229 | $4,059 | $49.6k |
| 41-feature model @ 0.843 (grouped country; served until 2026-11-15) | 365 (1.2%) | 106 | 259 | $4,271 | $49.7k |
| **Served model** (33 features, review ≥ 0.762, no model declines) | 0 denied, 2,043 reviewed (6.8%) | **469** (reviewed) | 0 (1,574 reviewed) | **$17,305** | **$44.3k**† |

\* missed-fraud dollars + $10 per false decline. † plus $5 per reviewed order, a reviewed
fraud counted as stopped. For the first three models ranking quality (PR-AUC) is unchanged
and the two cost-tuned ones are a wash: more fraud caught, more false declines, same total
cost. The served model is the first with better ranking (one new device feature) and the
first to lower total cost, and it does so by reviewing rather than declining.
After January, fraud is 4.5% and the signal is who else has used the device.

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
| Fraud_Data (151K txns, 1 per user) | 4.6% | PR-AUC 0.050 [0.045, 0.055] | 0.134 [0.122, 0.148] | 0.135 [0.121, 0.151]† | 21 / 56 |
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
re-measurement found ~50 RPS for repeat customers and under 30 RPS for new users. The cause was
a missing index on the audit table's `user_id`: the Redis-miss fallback scanned the whole table
(~46 ms at 433k rows), and the table grows with every load test. With the index
(`db_init/006`, 0.22 ms per lookup; on an existing database apply it by hand, the command is
in the script) and a single fallback query, both modes run without errors
and with p95 under 250 ms up to ~400 RPS, at ~4–6 ms of server time per request. Details:
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
