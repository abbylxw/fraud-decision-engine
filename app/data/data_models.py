from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_id: str
    # Required (HTTP 422 if missing): without it signup_burst can't run, and the model sees
    # days_since_signup = -999, a value it never trained on and scores as low risk.
    signup_time: datetime
    purchase_time: datetime
    purchase_value: float
    # Required: the device and IP lookups can't run without them, so a missing value is a
    # bad request (HTTP 422), not a crash and not a fraud decision. An empty device id would
    # put every such request on one shared device.
    device_id: str = Field(min_length=1)
    # Required model inputs: leaving them out let ~60% of model-caught fraud through
    # (rule-logic-review code_critique.md).
    source: str
    browser: str
    sex: str
    # Required: the model never saw a missing age in training. With age left out, its fraud
    # catches on the Fraud_Data test period drop from 106 to 59 (age-flip-check).
    age: int
    ip_address: int = Field(ge=0)

class DecisionResponse(BaseModel):
    transaction_id: UUID
    # "review" = do not auto-approve: step up (3-D Secure, SMS code) or queue for a person.
    fraud_decision: Literal["allow", "review", "deny"]
    response_time: datetime

class Features(Transaction):
    days_since_signup: int = 0
    days_since_signup_group: str = '0-7d'  # one of: 0-7d,8-14d,15-30d,31-90d,91-180d,181-360d,361-720d,720+d
    ip_country: str = 'Unknown'
    did_uid_days_on_file: int = 0
    did_uid_5min: int = 0
    did_uid_10min: int = 0
    did_uid_1h:  int = 0
    did_uid_2h:  int = 0
    did_uid_24h: int = 0
    did_uid_7d:  int = 0
    did_uid_30d: int = 0
    did_uid_90d: int = 0
    uid_did_5min: int = 0
    uid_did_10min: int = 0
    uid_did_1h:  int = 0
    uid_did_2h:  int = 0
    uid_did_24h: int = 0
    uid_did_7d:  int = 0
    uid_did_30d: int = 0
    uid_did_90d: int = 0
    ip_uid_5min: int = 0
    ip_uid_10min: int = 0
    ip_uid_1h:  int = 0
    ip_uid_2h:  int = 0
    ip_uid_24h: int = 0
    ip_uid_7d:  int = 0
    purchase_cnt_5min:  int = 0
    purchase_amt_5min:  float = 0
    purchase_cnt_10min: int = 0
    purchase_amt_10min: float = 0
    purchase_cnt_1h:    int = 0
    purchase_amt_1h:    float = 0
    purchase_cnt_24h:   int = 0
    purchase_amt_24h:   float = 0
    purchase_cnt_7d:    int = 0
    purchase_amt_7d:    float = 0
    purchase_cnt_30d:   int = 0
    purchase_amt_30d:   float = 0
    purchase_cnt_90d:   int = 0
    purchase_amt_90d:   float = 0
    purchase_success_cnt_5min:  int = 0
    purchase_success_amt_5min:  float = 0
    purchase_success_cnt_10min: int = 0
    purchase_success_amt_10min: float = 0
    purchase_success_cnt_1h:    int = 0
    purchase_success_amt_1h:    float = 0
    purchase_success_cnt_24h:   int = 0
    purchase_success_amt_24h:   float = 0
    purchase_success_cnt_7d:    int = 0
    purchase_success_amt_7d:    float = 0
    purchase_success_cnt_30d:   int = 0
    purchase_success_amt_30d:   float = 0
    purchase_success_cnt_90d:   int = 0
    purchase_success_amt_90d:   float = 0
    did_txn_cnt_5min:  int = 0
    did_txn_cnt_10min: int = 0
    did_txn_cnt_1h:    int = 0
    did_txn_cnt_2h:    int = 0
    did_txn_cnt_24h:   int = 0
    did_txn_cnt_7d:    int = 0
    did_txn_cnt_30d:   int = 0
    did_txn_cnt_all:   int = 0   # all-time, no window
    did_other_uid_all: int = 0   # other accounts ever seen on the device (model input)
    pur_median_amt_90d:       float = 0.0
    amount_vs_user_avg_30d:   float = 1.0
    age_account_gap:          float = 0.0
    high_amount_vs_hist:      int   = 0
    time_since_last_txn_uid:  float = -1.0   # -1 = no earlier purchase by this user
    country_txn_count_24h:   int = 0
    country_txn_count_7d:    int = 0
    country_txn_count_30d:   int = 0
    hours_to_purchase:       float = -1.0
    hour_of_day:             int   = 0
    day_of_week:             int   = 0
    month_of_year:           int   = 1

class DBWrite(Features, DecisionResponse):
    created_at: datetime
    # Audit fields: which model made the decision, its raw score, and which rule fired.
    # model_score is None when a hard rule short-circuited before the model ran.
    model_version: str
    model_score: float | None = None
    decision_rule: str | None = None
