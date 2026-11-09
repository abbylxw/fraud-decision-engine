# Throughput Re-measurement (2026-11-08)

Replaces the "~520 RPS ceiling / 400 RPS safe" numbers from
[performance_investigation.md](performance_investigation.md) (April 2026), which no longer hold.

## Summary

- **Today's ceiling is about 50 RPS for repeat customers and under 30 RPS for new users**, not 520.
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

## Proposed improvements (ranked; none implemented yet)

| # | Change | Expected effect | Effort |
|---|---|---|---|
| 1 | **Index `transactions (user_id, purchase_time)`.** New `db_init/006_*.sql` for fresh databases, plus a one-off `CREATE INDEX CONCURRENTLY` for the existing one | The fallback goes from a ~30 ms full scan to an index lookup under 1 ms. This removes the DB CPU saturation behind both modes' collapse, and the cost stops growing with table size | Small |
| 2 | **One fallback query instead of two.** Fetch the user's 90-day rows once and compute the counts and median in Python, reusing `_parse_purchase_entries` | Halves the cold-path DB round-trips | Small |
| 3 | **Benchmark hygiene.** Record the table sizes with every result, and start each sweep from the same database state | Makes runs comparable. Today's numbers depend on how many earlier tests filled the audit table | Small |
| 4 | **Re-measure, then look at the per-worker event loop.** April found ~65 RPS per worker from event-loop saturation. Velocity updates and ~5 JSON log lines per request run on the loop | Unknown until #1 is in. Only worth doing if the new ceiling is still well below April's 520 | Medium |

Each change gets the same sweep before and after, with the same command lines.
