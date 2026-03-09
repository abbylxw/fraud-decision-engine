from datetime import datetime, timezone
from fastapi import FastAPI, Header, Depends
import asyncio
from app.data.data_models import DBWrite, Transaction, DecisionResponse
from app.config import settings
from app.data.orm_models import TransactionRecord
from app.decisions.features import build_features, update_velocity_counters
from app.decisions import execute_rules
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from redis.asyncio import Redis
from contextlib import asynccontextmanager




engine = create_async_engine(settings.database_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
    yield
    await app.state.redis.close()
 

app = FastAPI(lifespan=lifespan)


# --- Dependency providers ---
async def get_db() -> AsyncSession:
    async with AsyncSession(engine) as db:
        yield db


async def get_redis() -> Redis:
    return app.state.redis


async def run_process(transaction: Transaction,
                  db: AsyncSession,
                  redis: Redis,
) -> DecisionResponse:
    features = await build_features(transaction, db, redis)
    ## rule decision
    ## model decision
    ## based on all above, return fraud_decision, and write the reponse into database.
    fraud_decision = execute_rules(transaction, features)
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
    return response

@app.post("/api/transactions", status_code=201, response_model = DecisionResponse)
async def process_transaction(
    transaction: Transaction,
    idempotency_key: str = Header(...),
    db: AsyncSession = Depends(get_db), 
    redis: Redis = Depends(get_redis), 
) -> DecisionResponse:
    cached = await redis.getex(f"idem:{idempotency_key}", ex=600) # Check cache with 10 minutes expiration
    if cached:
        return DecisionResponse.model_validate_json(cached)

    response = await run_process(transaction, db, redis)
    await redis.set(f"idem:{idempotency_key}", response.model_dump_json(), nx=True, ex=600)  # Cache for 10 minutes
    return response