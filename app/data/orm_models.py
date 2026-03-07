from typing import Optional
from datetime import datetime
from uuid import UUID
from sqlalchemy import UUID as SAUUID
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass

class TransactionRecord(Base):
    __tablename__ = "transactions"

    # from DecisionResponse -- primary key
    transaction_id: Mapped[UUID] = mapped_column(SAUUID(as_uuid=True), primary_key=True)
    fraud_decision: Mapped[str]
    response_time: Mapped[datetime]

    # from Transaction
    user_id: Mapped[str]
    signup_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    purchase_time: Mapped[datetime]
    purchase_value: Mapped[float]
    device_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    source: Mapped[Optional[str]] = mapped_column(nullable=True)
    browser: Mapped[Optional[str]] = mapped_column(nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(nullable=True)
    age: Mapped[Optional[int]] = mapped_column(nullable=True)
    ip_address: Mapped[Optional[int]] = mapped_column(nullable=True)
    fraud_class: Mapped[int] = mapped_column(name='fraud_class')

    # from Features
    days_since_signup: Mapped[int] = mapped_column(default=0)
    ip_country: Mapped[str] = mapped_column(default='Unknown')
    age_group: Mapped[str] = mapped_column(default='Unknown')
    did_uid_days_on_file: Mapped[int] = mapped_column(default=0)
    did_uid_2h: Mapped[int] = mapped_column(default=0)
    did_uid_24h: Mapped[int] = mapped_column(default=0)
    did_uid_7d: Mapped[int] = mapped_column(default=0)
    did_uid_30d: Mapped[int] = mapped_column(default=0)   
    did_uid_90d: Mapped[int] = mapped_column(default=0)
    uid_did_2h: Mapped[int] = mapped_column(default=0)
    uid_did_24h: Mapped[int] = mapped_column(default=0)
    uid_did_7d: Mapped[int] = mapped_column(default=0)
    uid_did_30d: Mapped[int] = mapped_column(default=0)
    uid_did_90d: Mapped[int] = mapped_column(default=0)
    ip_uid_2h: Mapped[int] = mapped_column(default=0)
    ip_uid_24h: Mapped[int] = mapped_column(default=0)
    ip_uid_7d: Mapped[int] = mapped_column(default=0)
    purchase_cnt_24h: Mapped[int] = mapped_column(default=0)
    purchase_amt_24h: Mapped[float] = mapped_column(default=0)
    purchase_cnt_7d: Mapped[int] = mapped_column(default=0)
    purchase_amt_7d: Mapped[float] = mapped_column(default=0)
    purchase_cnt_30d: Mapped[int] = mapped_column(default=0)
    purchase_amt_30d: Mapped[float] = mapped_column(default=0)
    purchase_cnt_90d: Mapped[int] = mapped_column(default=0)
    purchase_amt_90d: Mapped[float] = mapped_column(default=0)
    purchase_success_cnt_24h: Mapped[int] = mapped_column(default=0)
    purchase_success_amt_24h: Mapped[float] = mapped_column(default=0)
    purchase_success_cnt_7d: Mapped[int] = mapped_column(default=0)
    purchase_success_amt_7d: Mapped[float] = mapped_column(default=0)
    purchase_success_cnt_30d: Mapped[int] = mapped_column(default=0)
    purchase_success_amt_30d: Mapped[float] = mapped_column(default=0)
    purchase_success_cnt_90d: Mapped[int] = mapped_column(default=0)
    purchase_success_amt_90d: Mapped[float] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())          
