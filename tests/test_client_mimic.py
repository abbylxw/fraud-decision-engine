"""
client_mimic.py --source sparkov turns Sparkov rows into API requests, inventing the fields
Sparkov lacks. These checks need no CSV: the API must accept the request, and the made-up
values must be stable per card so the Redis windows build real per-user history.
"""
import itertools
import random
from datetime import datetime, timedelta

from app.data.client_mimic import (
    idempotency_key,
    purchase_ts,
    rate_schedule,
    replay_schedule,
    shift_row,
    sparkov_request,
    summary,
)
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


def test_parallel_clients_never_share_idempotency_keys():
    # Two clients replaying the same file send the same user at the same position. Without
    # a per-run prefix the keys collide and the API answers 409 instead of deciding.
    a = {idempotency_key("run_a", "22058", n) for n in range(100)}
    b = {idempotency_key("run_b", "22058", n) for n in range(100)}
    assert len(a) == 100 and not a & b


def test_summary_groups_errors_by_kind():
    out = summary(rps=None, sent=10, ok=7, err=3, skipped=0, elapsed=1.0, latencies=[1.0],
                  errors={"ReadTimeout": 2, 409: 1})
    assert "target RPS" not in out                      # replay mode has no target
    assert "errors by kind: ReadTimeout=2, 409=1" in out


def test_rate_schedule_gaps_are_random_with_the_right_mean():
    rng = random.Random(1)
    dues = [d for d, _ in itertools.islice(rate_schedule([{"a": 1}], 400, rng, start=0.0), 10_000)]
    gaps = [b - a for a, b in zip(dues, dues[1:])]
    assert abs(sum(gaps) / len(gaps) - 1 / 400) < 0.05 / 400    # mean within 5% of 1/rps
    assert len(set(round(g, 6) for g in gaps)) > 1000           # not a metronome


def test_shift_row_moves_both_timestamps_by_the_same_delta():
    row = {"user_id": "1", "signup_time": "2015-01-01 10:00:00", "purchase_time": "2015-01-03 12:30:00"}
    out = shift_row(row, timedelta(days=30))
    assert out["signup_time"] == "2015-01-31 10:00:00"
    assert out["purchase_time"] == "2015-02-02 12:30:00"
    assert row["purchase_time"] == "2015-01-03 12:30:00"        # original untouched


def test_replay_schedule_sorts_and_never_goes_backwards():
    rows = [{"user_id": u, "signup_time": "2015-01-01 00:00:00", "purchase_time": pt}
            for u, pt in (("a", "2015-03-01 00:00:00"), ("b", "2015-01-05 00:00:00"),
                          ("c", "2015-02-01 00:00:00"))]
    two_passes = list(itertools.islice(replay_schedule(rows, speedup=1000, start=0.0), 6))
    times = [purchase_ts(r) for _, r in two_passes]
    assert [r["user_id"] for _, r in two_passes] == ["b", "c", "a", "b", "c", "a"]
    assert times == sorted(times) and len(set(times)) == 6   # strictly increasing across the wrap
    dues = [d for d, _ in two_passes]
    assert dues[0] == 0.0 and dues == sorted(dues)
    # Second pass starts span + 1 day after the first: (Mar 1 - Jan 5) + 1 day = 56 days.
    assert times[3] - times[0] == 56 * 86400
