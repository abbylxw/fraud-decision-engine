"""
client_mimic.py --source sparkov turns Sparkov rows into API requests, inventing the fields
Sparkov lacks. These checks need no CSV: the API must accept the request, and the made-up
values must be stable per card so the Redis windows build real per-user history.
"""
from datetime import datetime

from app.data.client_mimic import sparkov_request, summary
from app.data.data_models import Transaction


def _row(time, amt="4.97"):
    return {"trans_date_trans_time": time, "cc_num": "2703186189652095", "amt": amt,
            "gender": "F", "dob": "1988-03-09"}


def test_sparkov_request_is_a_valid_transaction():
    signup = datetime(2018, 1, 1, 0, 0, 18)    # 365 days before the first purchase
    req = sparkov_request(_row("2019-01-01 00:00:18"), signup, card_index=0)
    t = Transaction(**req)
    assert (t.purchase_time - t.signup_time).days == 365
    assert t.age == 30                          # birthday (Mar 9) not reached yet on Jan 1
    assert t.sex == "F" and t.purchase_value == 4.97


def test_same_card_gets_same_device_ip_and_signup():
    signup = datetime(2018, 1, 1)
    a = sparkov_request(_row("2019-01-01 00:00:18"), signup, card_index=7)
    b = sparkov_request(_row("2019-01-05 12:00:00", "80.0"), signup, card_index=7)
    for field in ("user_id", "device_id", "ip_address", "signup_time"):
        assert a[field] == b[field]
    assert sparkov_request(_row("2019-01-01 00:00:18"), signup, card_index=8)["ip_address"] \
        != a["ip_address"]


def test_summary_reports_achieved_not_target_rps():
    # Asked for 400/s for 10 s, but only 1,500 came back OK in 12.5 s: achieved is 120/s.
    out = summary(rps=400, sent=1600, ok=1500, err=100, skipped=2400, elapsed=12.5,
                  latencies=[float(i) for i in range(1, 101)])
    assert "target RPS=400" in out and "achieved RPS=120" in out
    assert "skipped=2400" in out and "p50=51ms" in out
