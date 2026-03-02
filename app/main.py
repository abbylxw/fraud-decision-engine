from fastapi import FastAPI, Header, Depends
from app.data.data_models import Transaction, Response
from app.config import settings
from app.decisions.features import build_features
from uuid import uuid7
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import Session
from redis.asyncio import Redis


app = FastAPI()
# --- Dependency providers ---
def get_db() -> AsyncSession:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis() -> Redis:
    return Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)


async def run_process(transaction: Transaction,
                  db: Session,
                  redis: Redis,
) -> Response:
    features = await build_features(transaction, db, redis)
    ## rule decision
    ## model decision
    ## based on all above, return fraud_decision, and write the reponse into database.
    fraud_decision = ''
    response = Response(
        **transaction.model_dump(),
        transaction_id=uuid7(),
        fraud_decision=fraud_decision,
    )

    db.add(response)
    db.commit()
    return response

@app.post("/api/transactions", status_code=201)
async def process_transaction(
    transaction: Transaction,
    idempotency_key: str = Header(...),
    db: Session = Depends(get_db), 
    redis: Redis = Depends(get_redis), 
) -> Response:
    cached = await redis.get(f"idem:{idempotency_key}")
    if cached:
        return Response.model_validate_json(cached)

    response = await run_process(transaction, db, redis)
    await redis.setex(f"idem:{idempotency_key}", 86400, response.model_dump_json())
    return response