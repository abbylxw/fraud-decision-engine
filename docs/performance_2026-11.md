# Throughput Re-measurement (2026-11-08)

Replaces the "~520 RPS ceiling / 400 RPS safe" numbers from
[performance_investigation.md](performance_investigation.md) (April 2026), which no longer hold.

## Summary

**Update (2026-11-09):** fixes 1 and 2 below are in. The ceiling is now about **400 RPS for
repeat customers and 300–400 RPS for new users**, with server time ~4–6 ms per request.
See [After fixes 1–2](#after-fixes-12-2026-11-09). The rest of this section describes the
state before the fix.

- **The ceiling before the fix was about 50 RPS for repeat customers and under 30 RPS for new users**, not 520.
- **Cause: the `transactions` audit table has no index on `user_id`.**
  - For every user Redis has no history for (a new user, or after a Redis flush or eviction),
    `_purchase_counts_from_db` runs two queries filtered by `user_id`.
  - Each is a full parallel scan: about 30 ms and ~270 MB of buffer reads at 431k rows.
  - The table grows with every load test, so this got slower over time. It was small in April.
- The model (~2.5 ms) and the audit write (~1 ms) are not the problem.
  Feature extraction is 90%+ of server time.
- The load client is not the limit: it sustains ~800 RPS against a stub server.

## Setup

| | |
|---|---|
| Machine | Apple M4, 10 cores; Docker Desktop VM with 10 CPUs and 8 GB |
| API | 8 uvicorn workers; SQLAlchemy pool 5 + 6 per worker; Postgres `max_connections=100` |
| Also running | Prometheus and Grafana (idle) |
| Data in Postgres | `transactions` 431k rows (left over from earlier load tests), `device_data` 152k, `ip_country` 139k |
| Client | `app/data/client_mimic.py` on the host, one request every 1/RPS seconds, 30 s timeout |
| Per step | Redis `FLUSHALL`, then a 60 s run. Server step times are diffs of `/metrics` histograms (summed over all 8 workers since 84951d2) |

"Achieved RPS" is successful responses per second of wall time, including the time to drain
in-flight requests after the send window. When the API falls more than 1 s behind, the
client skips send slots instead of piling them up.

## Results

**Client ceiling** (stub ASGI server, 4 workers, instant 201):

```
python app/data/client_mimic.py --source sparkov --rps <N> --duration 15 --url http://127.0.0.1:8001/api/transactions
```

| Target | Achieved | p95 |
|---|---|---|
| 400 | 399 | 2 ms |
| 800 | 799 | 1 ms |
| 1500 | 1007 | 92 ms |

**API sweep:**

```
python app/data/client_mimic.py --source <mode> --rps <N> --duration 60
```

| Mode | Target RPS | Achieved RPS | Errors | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| sparkov (repeat customers) | 50 | 50 | 0 | 8 ms | 81 ms | 119 ms |
| | 100 | 92 | 1 | 54 ms | 8.3 s | 10.8 s |
| | 200 | 131 | 1 | 1.1 s | 14.6 s | 18.1 s |
| | 300 | 134 | 3 | 1.3 s | 15.6 s | 22.9 s |
| | 400 | 189 | 0 | 322 ms | 14.3 s | 17.9 s |
| | 500 | 128 | 5 | 2.4 s | 17.3 s | 23.3 s |
| fraud_data (every user new) | 50 | 41 | 0 | 5.0 s | 15.0 s | 17.1 s |
| | 100 | 32 | 864 | 14.1 s | 30.5 s | 31.2 s |
| | 200 | 31 | 683 | 20.1 s | 30.8 s | 31.4 s |
| | 300–500 | 30–33 | 494–783 | ~20 s | ~30.5 s | ~31.2 s |

Errors are client timeouts (30 s). Fraud_Data mode flattens at ~30 RPS whatever the target:
that is the cold-path capacity.

**Where the time goes at 30 RPS** (below saturation, 30 s per mode, server-side averages from
`/metrics`):

| Step | sparkov | fraud_data |
|---|---|---|
| feature_extraction | 33.1 ms | 57.3 ms |
| model_inference | 2.4 ms | 2.7 ms |
| db_write | 1.0 ms | 1.1 ms |
| whole request | 36.9 ms | 61.5 ms |

Fraud_Data requests are all cold: two table scans, about 2 × 30 ms. The Sparkov run is mostly
cold too, because Redis was just flushed and 899 requests over 922 cards are mostly each
card's first.

**Query plans** (`EXPLAIN ANALYZE`):
- **Purchase-history fallback:** `SELECT purchase_value FROM transactions WHERE user_id = ... AND
  purchase_time BETWEEN ...` is a Parallel Seq Scan, 31.5 ms, removing 431k rows, 34.8k
  buffers. The audit table's only indexes are the primary key and `model_version`.
- **IP lookup:** `ip_country WHERE ip_range @> ip` uses the GiST index, 0.03 ms. Fine.

## Proposed improvements (ranked)

| # | Change | Expected effect | Effort | Status |
|---|---|---|---|---|
| 1 | **Index `transactions (user_id, purchase_time)`.** New `db_init/006_*.sql` for fresh databases, plus a one-off `CREATE INDEX CONCURRENTLY` for the existing one | The fallback goes from a ~30 ms full scan to an index lookup under 1 ms. This removes the DB CPU saturation behind both modes' collapse, and the cost stops growing with table size | Small | Done 2026-11-09 |
| 2 | **One fallback query instead of two.** Fetch the user's 90-day rows once and compute the counts and median in Python, reusing `_parse_purchase_entries` | Halves the cold-path DB round-trips | Small | Done 2026-11-09 |
| 3 | **Benchmark hygiene.** Record the table sizes with every result, and start each sweep from the same database state | Makes runs comparable. Today's numbers depend on how many earlier tests filled the audit table | Small | Partly: row counts are now recorded per step. The table is not reset between sweeps; with the index, lookup cost barely depends on table size |
| 4 | **Re-measure, then look at the per-worker event loop.** April found ~65 RPS per worker from event-loop saturation. Velocity updates and ~5 JSON log lines per request run on the loop | Unknown until #1 is in. Only worth doing if the new ceiling is still well below April's 520 | Medium | Investigated 2026-11-09: logging and velocity updates are <5% of worker time. Fixed instead: XGBoost thread oversubscription (`n_jobs=1` at serving). Above ~500 RPS the laptop setup, not the API, is the limit. See [Fix 4](#fix-4-what-limits-throughput-above-400-rps-2026-11-09) |

Each change gets the same sweep before and after, with the same command lines.

## After fixes 1–2 (2026-11-09)

Same machine, same commands and same procedure as above (Redis `FLUSHALL` and a 60 s run per
step, 30 s warm-up per mode). The index was built on the live database with
`CREATE INDEX CONCURRENTLY` (7.8 MB), and the API image was rebuilt with the one-query fallback.

**Query plan:** the purchase-history fallback for one user went from a Parallel Seq Scan at
46.4 ms (433k rows) to an Index Scan on `idx_transactions_user_time` at 0.22 ms.

**Sweep** (`transactions` row count at the start of each step; it grew from 434,581 to
583,875 during the sweep, because every request adds an audit row):

| Mode | Target RPS | Achieved RPS | Errors | p50 | p95 | p99 | Server time per request | Rows at start |
|---|---|---|---|---|---|---|---|---|
| sparkov | 50 | 50 | 0 | 8 ms | 9 ms | 11 ms | 4.9 ms | 434,581 |
| | 100 | 100 | 0 | 6 ms | 8 ms | 11 ms | 3.5 ms | 437,580 |
| | 200 | 200 | 0 | 5 ms | 8 ms | 10 ms | 2.7 ms | 443,577 |
| | 300 | 293 | 0 | 5 ms | 214 ms | 452 ms | 3.9 ms | 455,570 |
| | 400 | 400 | 0 | 7 ms | 132 ms | 323 ms | 4.8 ms | 473,181 |
| | 500 | 382 | 0 | 117 ms | 423 ms | 655 ms | 5.5 ms | 497,167 |
| fraud_data | 50 | 50 | 0 | 8 ms | 10 ms | 12 ms | 5.6 ms | 521,618 |
| | 100 | 100 | 0 | 6 ms | 7 ms | 9 ms | 3.8 ms | 524,617 |
| | 200 | 200 | 0 | 6 ms | 8 ms | 38 ms | 3.9 ms | 530,615 |
| | 300 | 300 | 0 | 6 ms | 7 ms | 10 ms | 3.8 ms | 542,610 |
| | 400 | 387 | 0 | 6 ms | 125 ms | 359 ms | 4.5 ms | 560,599 |
| | 500 | 448 | 0 | 50 ms | 369 ms | 588 ms | 6.6 ms | 583,875 |

"Server time per request" is the `/metrics` request histogram, averaged over the step.
Per step at 300 RPS: feature_extraction 0.6 ms (sparkov) / 1.2 ms (fraud_data), down from
33 / 57 ms at 30 RPS before; model_inference ~2 ms; db_write ~0.5–0.8 ms.

**Reading it:**
- No errors at any rate in either mode (before: up to 864 timeouts per step).
- Both modes hold p95 under 250 ms up to 400 RPS. New-user traffic (fraud_data) is now as fast as
  repeat customers, because the cold path no longer scans the table.
- Sparkov at 300 RPS had a short stall (676 skipped slots, p95 214 ms) while 400 RPS was clean.
  This looks like a one-off hiccup (for example autovacuum on the growing table), not a ceiling.
  A repeat run would confirm it.
- From ~400 RPS, client-side latency rises while server time per request stays at ~4–6 ms.
  The extra waiting happens outside the measured handler: in the uvicorn/event-loop queue, the
  fire-and-forget velocity updates and the per-request log lines. The client is not the limit
  (~800 RPS against the stub). That is improvement #4. The new ceiling (~400) is close to April's
  520, so #4 is optional, not urgent.

## Fix 4: what limits throughput above ~400 RPS (2026-11-09)

**1. The single load client was the limit.** One `client_mimic.py` process at 500 RPS reached
406 RPS (p95 427 ms). Two processes at 250 each, same total, reached the full 500 (p95 ~35 ms).
One Python process juggling hundreds of in-flight requests adds its own delay, which shows up as
latency and skipped send slots. The ~400 RPS "ceiling" in the sweep above is the client's.

Running several clients needed care: the client's idempotency key was `{user_id}-{counter}`, so
two clients replaying the same file sent identical keys and the API correctly answered 409.
Runs below pair one `fraud_data` client with one `sparkov` client (user ids never overlap).
Since then each client process adds a random run id to its keys; two `fraud_data` clients at
250 RPS each sent ~12.3k requests with zero 409s.

**2. The API's workers are not CPU-bound.** At 500–800 RPS, per-worker CPU (from
`/proc/<pid>/stat`) was 0–60% for most workers. It was also uneven: each client keep-alive
connection stays on the worker that accepted it, so two workers sat near 0% while one hit ~100%.
Postgres used ~20% of a core and Redis ~16%. The Grafana panel "Requests per Worker" (counter
`fraud_worker_requests_total{worker=<pid>}`) now shows this imbalance: in one 2 × 250 RPS run
the busiest worker handled 2,574 requests and the quietest 728.

**3. Profile** (`py-spy record` on all 8 workers, 25 s at ~600 RPS, run from a throwaway
container with `--pid=container:<api>`; share of non-idle samples):

| Where | Share |
|---|---|
| uvicorn / uvloop event loop and network I/O | ~47–52% |
| XGBoost `predict_proba` (incl. `to_numpy`) | ~28% |
| FastAPI routing and dependencies | 15% |
| SQLAlchemy/asyncpg and Redis client | ~17% |
| JSON logging | 4% |
| Velocity counter updates (background) | 0.4% |
| Prometheus metrics | 0.3% |

Logging and the velocity updates, the two suspects from April, are too small to be worth changing.

**4. Fixed: XGBoost thread oversubscription.** The model is trained with `n_jobs=-1`, and
joblib keeps that setting, so every one-row prediction fanned out over all 10 cores, in each of
8 workers. Benchmark inside the API container, 8 processes predicting at once for 10 s:

| Serving setting | p50 per prediction | p99 | Predictions/s (all 8) |
|---|---|---|---|
| `n_jobs=-1` (as trained) | 6.7 ms | 15 ms | 1,125 |
| `n_jobs=1` | 1.2 ms | 4.6 ms | 4,308 |

`app/decisions/models.py` now sets `n_jobs=1` after loading. Scores are bit-for-bit identical
(37k rows in batch and 500 single-row calls compared against the model as saved). Training is
unchanged. At 600 RPS the busiest worker dropped from ~106% to ~80% CPU.

**5. Above ~500 RPS the test machine is the limit.** During a 600 RPS run the Docker Desktop VM
process used ~630% of host CPU while the containers inside reported ~300–400%. The rest is
virtualization and host-to-VM port forwarding. The M4 has 4 performance and 6 efficiency cores,
so beyond ~4 cores the VM partly runs on the slower cores. Identical 600 RPS runs gave anywhere
from 371 to 599 RPS achieved, and one reached the full 600 at p95 ~40 ms.

**Conclusion:** on this laptop the API sustains ~500 RPS cleanly (two clients, p95 under 50 ms)
and sometimes 600. A real ceiling needs the load generator on a separate machine (or a Linux
host without the VM) and a load balancer that spreads connections across workers.
