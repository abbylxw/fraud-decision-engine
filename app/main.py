import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.data.data_models import DBWrite, DecisionResponse, Transaction
from app.data.orm_models import TransactionRecord
from app.decisions import execute_hard_rules, execute_soft_rules
from app.decisions.features import build_features, event_ts, update_velocity_counters
from app.decisions.models import MODEL_VERSION
from app.decisions.models import score as model_score

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

_BUILT_IN_LOG_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class _JSONFormatter(logging.Formatter):
    """Emits one JSON object per log line, including any extra= fields."""
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _BUILT_IN_LOG_FIELDS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


logger = logging.getLogger("fraud_engine")


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "fraud_requests_total", "Total incoming requests including idempotency hits")
DECISIONS_TOTAL = Counter(
    "fraud_decisions_total", "Total fraud decisions", ["decision"])
RULE_DENIALS_TOTAL = Counter(
    "fraud_rule_denials_total", "Denials by rule name", ["phase", "rule"])
COUNTRY_REQUESTS = Counter(
    "fraud_country_requests_total", "Requests by ip_country and decision",
    ["country", "decision"])
REQUEST_LATENCY = Histogram(
    "fraud_request_duration_seconds", "End-to-end latency for POST /api/transactions",
    buckets=[.01, .025, .05, .1, .25, .5, 1.0, 2.5])
STEP_LATENCY = Histogram(
    "fraud_step_duration_seconds", "Processing time per pipeline step",
    ["step"],
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0])
MODEL_SCORE = Histogram(
    "fraud_model_score_value", "Distribution of raw fraud probability scores",
    buckets=[.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
IDEMPOTENCY_HITS = Counter(
    "fraud_idempotency_hits_total", "Requests served from idempotency cache")
IDEMPOTENCY_CONFLICTS = Counter(
    "fraud_idempotency_conflicts_total",
    "Duplicate requests rejected (409) because the original is still being processed")
BACKGROUND_TASK_FAILURES = Counter(
    "fraud_background_task_failures_total", "Fire-and-forget tasks that raised", ["task"])


# ---------------------------------------------------------------------------
# Fire-and-forget tasks
# ---------------------------------------------------------------------------
# asyncio only keeps a weak reference to tasks, so an unreferenced task can be
# garbage-collected mid-flight and its exception is never seen. Keep a strong
# reference until it finishes, and log + count failures.

_background_tasks: set[asyncio.Task] = set()


def _on_background_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        BACKGROUND_TASK_FAILURES.labels(task=task.get_name()).inc()
        logger.error("background_task.failed", extra={"task": task.get_name()}, exc_info=exc)


def spawn_background(coro, name: str) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_background_done)
    return task


# ---------------------------------------------------------------------------
# Database engine
# ---------------------------------------------------------------------------

engine = create_async_engine(settings.database_url, pool_size=5, max_overflow=6)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    app.state.redis = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    app.state.executor = ThreadPoolExecutor()
    yield
    # Let in-flight velocity updates finish before the Redis client closes.
    if _background_tasks:
        await asyncio.wait(_background_tasks, timeout=5)
    await app.state.redis.aclose()
    app.state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


# --- Dependency providers ---
async def get_db() -> AsyncSession:
    async with AsyncSession(engine) as db:
        yield db


async def get_redis() -> Redis:
    return app.state.redis


def get_executor() -> ThreadPoolExecutor:
    return app.state.executor


async def run_process(transaction: Transaction,
                  db: AsyncSession,
                  redis: Redis,
                  executor: ThreadPoolExecutor,
) -> DecisionResponse:
    logger.info("request.received", extra={
        "user_id": transaction.user_id,
        "purchase_value": transaction.purchase_value,
        "device_id": transaction.device_id,
    })
    with STEP_LATENCY.labels(step="feature_extraction").time():
        features = await build_features(transaction, db, redis)
    hard_decision, hard_rule = execute_hard_rules(transaction, features)
    prob: float | None = None
    if hard_decision == 'deny':
        fraud_decision = 'deny'
        decision_rule = hard_rule
        RULE_DENIALS_TOTAL.labels(phase="hard", rule=hard_rule).inc()
        logger.info("hard_rule.deny", extra={
            "user_id": transaction.user_id,
            "rule": hard_rule,
            "ip_country": features.ip_country,
            "did_uid_days_on_file": features.did_uid_days_on_file,
            "purchase_value": transaction.purchase_value,
            "age": transaction.age,
        })
    else:
        loop = asyncio.get_running_loop()
        with STEP_LATENCY.labels(step="model_inference").time():
            prob = await loop.run_in_executor(executor, model_score, features)
        MODEL_SCORE.observe(prob)
        soft_decision, soft_rule = execute_soft_rules(transaction, features, prob)
        fraud_decision = soft_decision
        decision_rule = soft_rule
        if soft_decision == 'deny' and soft_rule:
            RULE_DENIALS_TOTAL.labels(phase="soft", rule=soft_rule).inc()
        logger.info("model.scored", extra={
            "user_id": transaction.user_id,
            "prob": round(prob, 4),
            "decision": fraud_decision,
            "rule": soft_rule,
        })
    DECISIONS_TOTAL.labels(decision=fraud_decision).inc()
    COUNTRY_REQUESTS.labels(country=features.ip_country, decision=fraud_decision).inc()
    response = DecisionResponse(
        transaction_id=uuid4(),
        fraud_decision=fraud_decision,
        response_time=datetime.now(tz=timezone.utc),
    )
    db_record = DBWrite(
        **features.model_dump(by_alias=False), **response.model_dump(),
        created_at=datetime.now(tz=timezone.utc),
        model_version=MODEL_VERSION, model_score=prob, decision_rule=decision_rule,
    )
    db.add(TransactionRecord(**db_record.model_dump()))
    with STEP_LATENCY.labels(step="db_write").time():
        await db.commit()
    # Only after the commit: if the audit write fails, the transaction never happened,
    # so it must not count toward anyone's velocity windows. Still off the request path.
    spawn_background(
        update_velocity_counters(
            user_id=str(transaction.user_id),
            device_id=str(transaction.device_id),
            ip=transaction.ip_address,
            ip_country=features.ip_country,
            transaction_id=response.transaction_id,
            purchase_value=transaction.purchase_value,
            fraud_decision=response.fraud_decision,
            event_ts=event_ts(transaction),
            redis=redis,
        ),
        name="update_velocity_counters",
    )
    logger.info("decision.written", extra={
        "transaction_id": str(response.transaction_id),
        "fraud_decision": fraud_decision,
    })
    return response

@app.get("/health")
async def health(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> dict:
    checks: dict[str, str] = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, **checks}


# Idempotency: the key is reserved atomically (SET NX) BEFORE processing, so two
# concurrent requests with the same key can't both run the pipeline, write two audit
# rows and double-count velocity. States of idem:<key>:
#   absent          -> claim it and process
#   _IDEM_PENDING   -> the original is in flight: 409 + Retry-After
#   response JSON   -> replay the stored response
IDEMPOTENCY_TTL_SECONDS = 60    # how long a finished response is replayed (was 600;
                                # cut to 60 in ab0f917 to keep load-test reruns from replaying)
IDEMPOTENCY_LOCK_SECONDS = 30   # reservation lifetime; far above worst-case request time,
                                # so a crashed worker can't block the key for long
_IDEM_PENDING = "__pending__"


def _idempotency_conflict() -> HTTPException:
    IDEMPOTENCY_CONFLICTS.inc()
    return HTTPException(
        status_code=409,
        detail="A request with this Idempotency-Key is still being processed; retry shortly.",
        headers={"Retry-After": "1"},
    )


@app.post("/api/transactions", status_code=201, response_model = DecisionResponse)
async def process_transaction(
    transaction: Transaction,
    idempotency_key: str = Header(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    executor: ThreadPoolExecutor = Depends(get_executor),
) -> DecisionResponse:
    REQUESTS_TOTAL.inc()
    idem_key = f"idem:{idempotency_key}"
    claimed = await redis.set(idem_key, _IDEM_PENDING, nx=True, ex=IDEMPOTENCY_LOCK_SECONDS)
    if not claimed:
        existing = await redis.get(idem_key)
        if existing is None or existing == _IDEM_PENDING:
            # In flight (or the reservation expired between SET and GET): never process twice.
            logger.info("idempotency.conflict", extra={"idempotency_key": idempotency_key})
            raise _idempotency_conflict()
        IDEMPOTENCY_HITS.inc()
        logger.info("idempotency.hit", extra={"idempotency_key": idempotency_key})
        return DecisionResponse.model_validate_json(existing)

    _t0 = asyncio.get_event_loop().time()
    try:
        response = await run_process(transaction, db, redis, executor)
    except BaseException:
        # Nothing was committed (the DB write is the last step that can fail), so release
        # the key and let the client retry instead of getting 409s for 30 seconds.
        try:
            await redis.delete(idem_key)
        except Exception:
            logger.exception("idempotency.release_failed", extra={"idempotency_key": idempotency_key})
        raise
    REQUEST_LATENCY.observe(asyncio.get_event_loop().time() - _t0)
    await redis.set(idem_key, response.model_dump_json(), ex=IDEMPOTENCY_TTL_SECONDS)
    return response
