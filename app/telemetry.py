"""
Operational and drift metrics that are not tied to one request step.

app/main.py holds the request counters and histograms. This module holds the metrics that
other packages need to touch (the feature code counts its Postgres fallbacks) and the
drift counters described in app/monitoring.py. Alert rules that read these metrics are in
prometheus/rules/fraud_engine.yml; what each alert means is in docs/monitoring.md.
"""
import asyncio
import logging

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


async def _ping_postgres(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def probe_dependencies(engine: AsyncEngine, redis: Redis) -> None:
    """Runs for the life of the worker: every PROBE_INTERVAL_SECONDS check that Postgres
    and Redis answer, and record how many keys Redis holds. Independent of traffic, so an
    outage is seen even when no request arrives."""
    while True:
        try:
            await asyncio.wait_for(_ping_postgres(engine), timeout=3)
            DEPENDENCY_UP.labels(dependency="postgres").set(1)
        except Exception as e:
            DEPENDENCY_UP.labels(dependency="postgres").set(0)
            logger.error("dependency.down", extra={"dependency": "postgres", "error": str(e)})
        try:
            REDIS_KEYS.set(await asyncio.wait_for(redis.dbsize(), timeout=3))
            DEPENDENCY_UP.labels(dependency="redis").set(1)
        except Exception as e:
            DEPENDENCY_UP.labels(dependency="redis").set(0)
            logger.error("dependency.down", extra={"dependency": "redis", "error": str(e)})
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)
