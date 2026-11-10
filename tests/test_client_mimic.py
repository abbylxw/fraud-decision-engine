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
    plan_retry,
    purchase_ts,
    rate_schedule,
    replay_schedule,
    retry_row,
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


def test_summary_prints_extra_lines():
    out = summary(rps=50, sent=1, ok=1, err=0, skipped=0, elapsed=1.0, latencies=[1.0],
                  extra=["decisions: allow=1 deny=0 (deny rate=0.0%)", "retries=0"])
    assert "  decisions: allow=1" in out and "  retries=0" in out


def test_rate_schedule_gaps_are_random_with_the_right_mean():
    rng = random.Random(1)
    dues = [d for d, _ in itertools.islice(rate_schedule([{"a": 1}], 400, rng, start=0.0), 10_000)]
    gaps = [b - a for a, b in zip(dues, dues[1:])]
    assert abs(sum(gaps) / len(gaps) - 1 / 400) < 0.05 / 400    # mean within 5% of 1/rps
    assert len(set(round(g, 6) for g in gaps)) > 1000           # not a metronome


def test_shift_row_keeps_signup_fixed_by_default():
    # A returning customer buys later, but signed up when they signed up: the account ages.
    row = {"user_id": "1", "signup_time": "2015-01-01 10:00:00", "purchase_time": "2015-01-03 12:30:00"}
    out = shift_row(row, timedelta(days=30))
    assert out["signup_time"] == "2015-01-01 10:00:00"
    assert out["purchase_time"] == "2015-02-02 12:30:00"
    assert row["purchase_time"] == "2015-01-03 12:30:00"        # original untouched


def test_shift_row_can_move_signup_too_for_a_new_signup():
    row = {"user_id": "1", "signup_time": "2015-01-01 10:00:00", "purchase_time": "2015-01-03 12:30:00"}
    out = shift_row(row, timedelta(days=30), fields=("purchase_time", "signup_time"))
    assert out["signup_time"] == "2015-01-31 10:00:00"
    assert out["purchase_time"] == "2015-02-02 12:30:00"


def _three_rows():
    return [{"user_id": u, "device_id": f"dev-{u}", "signup_time": "2015-01-01 00:00:00",
             "purchase_time": pt, "purchase_value": "40.0"}
            for u, pt in (("a", "2015-03-01 00:00:00"), ("b", "2015-01-05 00:00:00"),
                          ("c", "2015-02-01 00:00:00"))]


def test_replay_second_pass_returning_users_keep_ids_and_signup():
    second = [r for _, r in itertools.islice(replay_schedule(_three_rows(), 1000, 0.0,
                                                             random.Random(1), new_share=0.0), 3, 6)]
    assert [r["user_id"] for r in second] == ["b", "c", "a"]
    assert all(r["signup_time"] == "2015-01-01 00:00:00" for r in second)
    assert second[0]["purchase_time"] == "2015-03-02 00:00:00"       # Jan 5 + 56 days


def test_replay_second_pass_new_users_get_fresh_ids_and_a_fresh_signup():
    second = [r for _, r in itertools.islice(replay_schedule(_three_rows(), 1000, 0.0,
                                                             random.Random(1), new_share=1.0), 3, 6)]
    assert [r["user_id"] for r in second] == ["b-c1", "c-c1", "a-c1"]
    assert [r["device_id"] for r in second] == ["dev-b-c1", "dev-c-c1", "dev-a-c1"]
    assert all(r["signup_time"] == "2015-02-26 00:00:00" for r in second)   # Jan 1 + 56 days


def test_retry_row_is_the_same_customer_a_bit_later_for_a_smaller_amount():
    row = _three_rows()[0] | {"ip_address": "123", "purchase_value": "100.0"}
    rng = random.Random(3)
    for _ in range(50):
        out = retry_row(row, rng)
        for field in ("user_id", "device_id", "ip_address", "signup_time"):
            assert out[field] == row[field]
        gap = purchase_ts(out) - purchase_ts(row)
        assert 15 <= gap <= 600
        assert 50.0 <= out["purchase_value"] <= 100.0
        assert out["purchase_value"] == round(out["purchase_value"], 2)


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_plan_retry_only_after_a_deny_with_attempts_left():
    always = _FixedRng(0.0)
    assert plan_retry("deny", 0, 10, 0.5, always)
    assert plan_retry("deny", 9, 10, 0.5, always)
    assert not plan_retry("deny", 10, 10, 0.5, always)       # 10 attempts used up
    assert not plan_retry("allow", 0, 10, 0.5, always)       # got through: no retry
    assert not plan_retry(None, 0, 10, 0.5, always)          # no decision (error)
    never = _FixedRng(0.99)
    assert not plan_retry("deny", 0, 10, 0.5, never)         # this customer gives up


def test_replay_schedule_sorts_and_never_goes_backwards():
    rows = _three_rows()
    two_passes = list(itertools.islice(replay_schedule(rows, speedup=1000, start=0.0), 6))
    times = [purchase_ts(r) for _, r in two_passes]
    assert [r["user_id"] for _, r in two_passes] == ["b", "c", "a", "b", "c", "a"]
    assert times == sorted(times) and len(set(times)) == 6   # strictly increasing across the wrap
    dues = [d for d, _ in two_passes]
    assert dues[0] == 0.0 and dues == sorted(dues)
    # Second pass starts span + 1 day after the first: (Mar 1 - Jan 5) + 1 day = 56 days.
    assert times[3] - times[0] == 56 * 86400
