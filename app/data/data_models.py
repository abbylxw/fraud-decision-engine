from typing import Annotated, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, time, timedelta
from uuid import UUID, uuid4

class Transaction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_id: str
    signup_time: datetime | None = None
    purchase_time: datetime 
    purchase_value: float
    device_id: str | None = None
    source: str | None = None
    browser: str | None = None
    sex:str | None = None
    age: int | None = None
    ip_address: int | None = None
    fraud_class: Literal[0, 1] = Field(alias='class')

class DecisionResponse(BaseModel):
    transaction_id: UUID
    fraud_decision: str
    response_time: datetime

class Features(Transaction):
    days_since_signup: int = 0
    ip_country: str = 'Unknown'
    age_group: str = 'Unknown'
    did_uid_days_on_file: int = 0
    did_uid_2h: int = 0
    did_uid_24h: int = 0
    did_uid_7d: int = 0
    did_uid_30d: int = 0
    did_uid_90d: int = 0
    uid_did_2h: int = 0
    uid_did_24h: int = 0
    uid_did_7d: int = 0
    uid_did_30d: int = 0
    uid_did_90d: int = 0
    ip_uid_2h: int = 0
    ip_uid_24h: int = 0
    ip_uid_7d: int = 0
    purchase_cnt_24h: int = 0
    purchase_amt_24h: float = 0
    purchase_cnt_7d: int = 0
    purchase_amt_7d: float = 0
    purchase_cnt_30d: int = 0
    purchase_amt_30d: float = 0
    purchase_cnt_90d: int = 0
    purchase_amt_90d: float = 0
    purchase_success_cnt_24h: int = 0
    purchase_success_amt_24h: float = 0
    purchase_success_cnt_7d: int = 0
    purchase_success_amt_7d: float = 0
    purchase_success_cnt_30d: int = 0
    purchase_success_amt_30d: float = 0
    purchase_success_cnt_90d: int = 0
    purchase_success_amt_90d: float = 0

class DBWrite(Features, DecisionResponse):
    created_at: datetime   


