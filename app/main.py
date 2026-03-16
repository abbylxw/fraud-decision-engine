from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fastapi import FastAPI, Header, Depends
import asyncio
import json
import logging
from sqlalchemy import text
from app.data.data_models import DBWrite, Transaction, DecisionResponse
from app.config import settings
from app.data.orm_models import TransactionRecord
from app.decisions.features import build_features, update_velocity_counters
from app.decisions import execute_hard_rules, execute_soft_rules
from app.decisions.models import score as model_score
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from redis.asyncio import Redis
from contextlib import asynccontextmanager




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
# Database engine
# ---------------------------------------------------------------------------

engine = create_async_engine(settings.database_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    app.state.redis = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    app.state.executor = ThreadPoolExecutor()
    yield
    await app.state.redis.aclose()
    app.state.executor.shutdown(wait=False)
 

app = FastAPI(lifespan=lifespan)


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
    features = await build_features(transaction, db, redis)
    hard_decision, hard_rule = execute_hard_rules(transaction, features)
    if hard_decision == 'deny':
        fraud_decision = 'deny'
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
        prob = await loop.run_in_executor(executor, model_score, features)
        soft_decision, soft_rule = execute_soft_rules(transaction, features, prob)
        fraud_decision = soft_decision
        logger.info("model.scored", extra={
            "user_id": transaction.user_id,
            "prob": round(prob, 4),
            "decision": fraud_decision,
            "rule": soft_rule,
        })
    response = DecisionResponse(
        transaction_id=uuid4(),
        fraud_decision=fraud_decision,
        response_time=datetime.now(tz=timezone.utc),
    )
    asyncio.create_task(
        update_velocity_counters(
            user_id=str(transaction.user_id),
            device_id=str(transaction.device_id),
            ip=transaction.ip_address,
            ip_country=features.ip_country,
            transaction_id=response.transaction_id,
            purchase_value=transaction.purchase_value,
            fraud_decision=response.fraud_decision,
            redis=redis,
        )
    )
    db_record = DBWrite(**features.model_dump(by_alias=False), **response.model_dump(), created_at=datetime.now(tz=timezone.utc))
    db.add(TransactionRecord(**db_record.model_dump()))
    await db.commit()
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


@app.post("/api/transactions", status_code=201, response_model = DecisionResponse)
async def process_transaction(
    transaction: Transaction,
    idempotency_key: str = Header(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    executor: ThreadPoolExecutor = Depends(get_executor),
) -> DecisionResponse:
    cached = await redis.getex(f"idem:{idempotency_key}", ex=600) # Check cache with 10 minutes expiration
    if cached:
        logger.info("idempotency.hit", extra={"idempotency_key": idempotency_key})
        return DecisionResponse.model_validate_json(cached)

    response = await run_process(transaction, db, redis, executor)
    await redis.set(f"idem:{idempotency_key}", response.model_dump_json(), nx=True, ex=600)  # Cache for 10 minutes
    return response