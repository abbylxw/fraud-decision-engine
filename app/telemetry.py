"""
Operational and drift metrics that are not tied to one request step.

app/main.py holds the request counters and histograms. This module holds the metrics that
other packages need to touch (the feature code counts its Postgres fallbacks) and the
drift counters described in app/monitoring.py. Alert rules that read these metrics are in
prometheus/rules/fraud_engine.yml; what each alert means is in docs/monitoring.md.
"""
import asyncio
import logging
import os
import resource
import sys
import time

from prometheus_client import Counter, Gauge
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.decisions.models import MODEL_VERSION, MONITORING_REFERENCE
from app.monitoring import bucket_index

logger = logging.getLogger("fraud_engine")

# A 500 used to be invisible to Prometheus: the request counter went up and nothing else.
REQUEST_ERRORS = Counter(
    "fraud_request_errors_total",
    "Requests that failed with an unhandled error, by kind", ["kind"])

# Features read from Postgres because Redis had nothing for the key. Normal for a first-seen
# user or device; a surge on known traffic means Redis lost data.
FEATURE_FALLBACK = Counter(
    "fraud_feature_fallback_total",
    "Feature lookups answered by Postgres because Redis had no data", ["source"])

# 1 = reachable. Every worker probes on its own; "livemin" reports the worst live worker.
DEPENDENCY_UP = Gauge(
    "fraud_dependency_up", "Dependency reachable from the API (1) or not (0)",
    ["dependency"], multiprocess_mode="livemin")
# Number of keys in Redis. All velocity state lives there; a sudden drop is data loss
# (a restart without persistence, a FLUSHALL, an eviction storm).
REDIS_KEYS = Gauge(
    "fraud_redis_keys", "Keys in the Redis database (DBSIZE)", multiprocess_mode="livemax")

# --- Saturation: how full is each thing that can run out -------------------------------
# (The fourth "golden signal". Latency, traffic and errors are measured in app/main.py.)
# Requests being processed right now, summed over workers.
INFLIGHT_REQUESTS = Gauge(
    "fraud_inflight_requests", "Requests currently being processed", multiprocess_mode="livesum")
# The event loop is the one thread that serves every request of a worker. If something
# blocks it, a 0.5 s sleep wakes up late; the lateness is the lag. The earliest sign that a
# worker is overloaded, before latency percentiles move.
EVENT_LOOP_LAG = Gauge(
    "fraud_event_loop_lag_seconds", "Worst event-loop lag of any worker in the last probe window",
    multiprocess_mode="livemax")
DB_POOL_IN_USE = Gauge(
    "fraud_db_pool_in_use", "Postgres connections checked out of the pool, all workers",
    multiprocess_mode="livesum")
DB_POOL_CAPACITY = Gauge(
    "fraud_db_pool_capacity", "Postgres pool size plus overflow, all workers",
    multiprocess_mode="livesum")
# One series per worker (label pid): the share of one CPU core the worker used.
WORKER_CPU = Gauge(
    "fraud_worker_cpu_utilization", "CPU used by a worker over the last probe window (1 = one full core)",
    multiprocess_mode="liveall")
WORKER_MEMORY = Gauge(
    "fraud_worker_memory_bytes", "Resident memory of a worker", multiprocess_mode="liveall")
POSTGRES_CONNECTIONS = Gauge(
    "fraud_postgres_connections", "Connections open on the Postgres server", multiprocess_mode="livemax")
POSTGRES_MAX_CONNECTIONS = Gauge(
    "fraud_postgres_max_connections", "Postgres max_connections setting", multiprocess_mode="livemax")
REDIS_MEMORY_USED = Gauge(
    "fraud_redis_memory_used_bytes", "Memory used by Redis", multiprocess_mode="livemax")
REDIS_MEMORY_MAX = Gauge(
    "fraud_redis_memory_max_bytes", "Redis maxmemory (0 = no limit configured)",
    multiprocess_mode="livemax")

MODEL_INFO = Gauge(
    "fraud_model_info", "Served model version (value is always 1)", ["version"],
    multiprocess_mode="livemax")
MODEL_INFO.labels(version=MODEL_VERSION).set(1)

# What training said to expect, for "live vs expected" alerts.
EXPECTED_RATE = Gauge(
    "fraud_expected_rate", "Rates measured on validation data when the model was trained",
    ["kind"], multiprocess_mode="livemax")

# Drift: one counter per bucket of the score and of each monitored feature, and the share
# of traffic training expected in that bucket. Counted for scored requests only (the
# reference was computed on rows the model scores, i.e. not hard-rule denials).
DRIFT_BUCKET = Counter(
    "fraud_drift_bucket_total",
    "Scored requests by bucket of the model score or of a monitored feature",
    ["feature", "bucket"])
DRIFT_REFERENCE = Gauge(
    "fraud_drift_reference_share",
    "Expected share of scored requests in each bucket (validation data at training time)",
    ["feature", "bucket"], multiprocess_mode="livemax")

SCORE = "model_score"   # the score is monitored like a feature, under this name
_edges: dict[str, list[float]] = {}


def _load_reference(reference: dict | None) -> None:
    """Export the reference shares and create every bucket's counter at 0, so a bucket
    that never receives traffic still has a series (PSI needs the empty buckets too)."""
    if not reference:
        return   # a model shipped before drift monitoring existed: nothing to compare with
    for kind, value in reference.get("expected", {}).items():
        EXPECTED_RATE.labels(kind=kind).set(value)
    for name, buckets in {SCORE: reference["score"], **reference["features"]}.items():
        _edges[name] = list(buckets["edges"])
        for i, share in enumerate(buckets["shares"]):
            DRIFT_REFERENCE.labels(feature=name, bucket=str(i)).set(share)
            DRIFT_BUCKET.labels(feature=name, bucket=str(i))


_load_reference(MONITORING_REFERENCE)


def observe_drift(features, score: float) -> None:
    """Count one scored request into its score bucket and its bucket of each monitored
    feature. A few bisects per request; never allowed to fail the request."""
    try:
        for name, edges in _edges.items():
            value = score if name == SCORE else getattr(features, name)
            DRIFT_BUCKET.labels(feature=name, bucket=str(bucket_index(edges, value))).inc()
    except Exception:
        logger.exception("drift.observe_failed")


PROBE_INTERVAL_SECONDS = 10
LOOP_LAG_SLEEP_SECONDS = 0.5


def _memory_bytes() -> int:
    """Resident memory of this process. Linux (the container) reads /proc; elsewhere the
    peak from getrusage is close enough for a dashboard."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


class _LoopLag:
    """Keeps the worst lag seen since the last read."""
    worst = 0.0


async def watch_event_loop() -> None:
    """Sleep LOOP_LAG_SLEEP_SECONDS over and over and record how late each wake-up is."""
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(LOOP_LAG_SLEEP_SECONDS)
        _LoopLag.worst = max(_LoopLag.worst, loop.time() - start - LOOP_LAG_SLEEP_SECONDS)


async def _probe_postgres(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        used, limit = (await conn.execute(text(
            "SELECT (SELECT count(*) FROM pg_stat_activity), "
            "current_setting('max_connections')::int"))).one()
    POSTGRES_CONNECTIONS.set(used)
    POSTGRES_MAX_CONNECTIONS.set(limit)


async def _probe_redis(redis: Redis) -> None:
    REDIS_KEYS.set(await redis.dbsize())
    memory = await redis.info("memory")
    REDIS_MEMORY_USED.set(memory.get("used_memory", 0))
    REDIS_MEMORY_MAX.set(memory.get("maxmemory", 0))


async def probe_dependencies(engine: AsyncEngine, redis: Redis) -> None:
    """Runs for the life of the worker: every PROBE_INTERVAL_SECONDS check that Postgres
    and Redis answer and how full they are, and record this worker's own CPU, memory,
    connection-pool use and event-loop lag. Independent of traffic, so an outage or a
    stuck worker is seen even when no request arrives."""
    lag_watcher = asyncio.create_task(watch_event_loop(), name="loop_lag")
    cpu_before, wall_before = time.process_time(), time.monotonic()
    try:
        while True:
            for name, probe, target in (("postgres", _probe_postgres, engine),
                                        ("redis", _probe_redis, redis)):
                try:
                    await asyncio.wait_for(probe(target), timeout=3)
                    DEPENDENCY_UP.labels(dependency=name).set(1)
                except Exception as e:
                    DEPENDENCY_UP.labels(dependency=name).set(0)
                    logger.error("dependency.down", extra={"dependency": name, "error": str(e)})
            try:
                pool = engine.pool
                DB_POOL_IN_USE.set(pool.checkedout())
                DB_POOL_CAPACITY.set(pool.size() + max(getattr(pool, "_max_overflow", 0), 0))
                cpu_now, wall_now = time.process_time(), time.monotonic()
                WORKER_CPU.set((cpu_now - cpu_before) / max(wall_now - wall_before, 1e-9))
                cpu_before, wall_before = cpu_now, wall_now
                WORKER_MEMORY.set(_memory_bytes())
                EVENT_LOOP_LAG.set(max(_LoopLag.worst, 0.0))
                _LoopLag.worst = 0.0
            except Exception:
                logger.exception("probe.self_failed")
            await asyncio.sleep(PROBE_INTERVAL_SECONDS)
    finally:
        lag_watcher.cancel()
