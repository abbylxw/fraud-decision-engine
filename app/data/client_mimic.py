"""
Load-test client for POST /api/transactions. Two ways to turn a CSV into traffic:

  rate mode (default, --rps N): rows in file order, forever, with Poisson arrivals (random gaps
      whose mean is 1/N s). For throughput and latency measurements. The rows keep their 2015
      timestamps, so after one pass through the file the Redis velocity windows drift: the same
      user buys the same thing at the same second again. Fine for load, not for decisions.

  replay mode (--speedup K): rows sorted by purchase_time and sent when a virtual clock running
      K times faster than wall time reaches them. When the file ends, every timestamp is shifted
      forward by the file's span, so time never goes backwards and the second pass is the same
      customers coming back months later. At K=1000, Fraud_Data's 8 months take ~6 h at ~6 req/s,
      which is the honest rate for this dataset. For realistic, continuous traffic; --rps is ignored.

Open loop: requests are sent on schedule whether or not earlier ones have been answered, so a
slow API shows up as latency and skipped slots, not as a lower send rate.
"""
import argparse
import asyncio
import csv
import itertools
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import httpx

_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def load_csv():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "Fraud_Data.csv"
    with open(DATA_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

# Sparkov (Kaggle kartik2112/fraud-detection): 999 cards with many purchases each, so the
# per-user Redis windows get real history. LOAD TEST ONLY: Sparkov has no device, IP,
# source, browser or signup time, so these are made up and the decisions mean nothing.
SPARKOV_PATH = Path(__file__).resolve().parents[2] / "data/external/sparkov/fraudTrain.csv"
SPARKOV_IP_BASE = 872415232   # start of a 16M-address range the IP table maps to United States


def sparkov_request(row, signup, card_index):
    """One Sparkov CSV row -> API request body. signup is 365 days before the card's first
    purchase: using the first purchase itself would trip signup_burst on every card."""
    pt = datetime.strptime(row["trans_date_trans_time"], _TIME_FMT)
    dob = datetime.strptime(row["dob"], "%Y-%m-%d")
    age = pt.year - dob.year - ((pt.month, pt.day) < (dob.month, dob.day))
    return {
        "user_id": row["cc_num"],
        "signup_time": signup.strftime(_TIME_FMT),
        "purchase_time": row["trans_date_trans_time"],
        "purchase_value": float(row["amt"]),
        "device_id": f"sp-dev-{row['cc_num']}",
        "source": "Direct",
        "browser": "Chrome",
        "sex": row["gender"],
        "age": age,
        "ip_address": SPARKOV_IP_BASE + card_index,
    }


def load_sparkov(rows=100_000):
    """First `rows` rows of fraudTrain.csv (already in time order; 100k is ~2.5 weeks)."""
    first_seen = {}   # cc_num -> (signup, card_index)
    out = []
    with open(SPARKOV_PATH, newline='') as csvfile:
        for row in itertools.islice(csv.DictReader(csvfile), rows):
            card = row["cc_num"]
            if card not in first_seen:
                pt = datetime.strptime(row["trans_date_trans_time"], _TIME_FMT)
                first_seen[card] = (pt - timedelta(days=365), len(first_seen))
            out.append(sparkov_request(row, *first_seen[card]))
    return out

def preprocess(row):
    """CSV rows are all strings. Fix fields Pydantic won't coerce automatically."""
    out = dict(row)
    if out.get("ip_address"):
        out["ip_address"] = int(float(out["ip_address"]))  # "3987484328.51882" → 3987484328
    out.pop("class", None)  # training label — not part of the live request schema
    return out


def purchase_ts(row) -> float:
    """purchase_time as a Unix timestamp (both sources use the same string format)."""
    return datetime.strptime(row["purchase_time"], _TIME_FMT).timestamp()


def shift_row(row, delta: timedelta):
    """Copy of row with purchase_time AND signup_time moved by delta, so days_since_signup and
    hours_to_purchase are unchanged and only "when" the purchase happened moves."""
    out = dict(row)
    for field in ("purchase_time", "signup_time"):
        if out.get(field):
            t = datetime.strptime(out[field], _TIME_FMT) + delta
            out[field] = t.strftime(_TIME_FMT)
    return out


# ---------------------------------------------------------------------------
# Schedules: each yields (due, row), due in time.monotonic() seconds, non-decreasing.
# ---------------------------------------------------------------------------

def rate_schedule(rows, rps, rng, start):
    """Rows in file order, forever, Poisson arrivals: gaps are exponential with mean 1/rps.
    Real arrivals cluster and leave gaps; an even 1/rps drip never piles up and so
    understates tail latency."""
    due = start
    for row in itertools.cycle(rows):
        due += rng.expovariate(rps)
        yield due, row


def replay_schedule(rows, speedup, start):
    """Rows sorted by purchase_time, due when a virtual clock (speedup x wall time) reaches
    them. After the last row the whole file is shifted forward by its span + 1 day and
    replayed, so timestamps only ever increase and users return as repeat customers."""
    ordered = sorted(rows, key=purchase_ts)
    first, last = purchase_ts(ordered[0]), purchase_ts(ordered[-1])
    span = timedelta(seconds=last - first) + timedelta(days=1)
    for cycle in itertools.count():
        offset = span * cycle
        for row in ordered:
            shifted = shift_row(row, offset) if cycle else row
            yield start + (purchase_ts(shifted) - first) / speedup, shifted


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000/api/transactions"


def idempotency_key(run_id, user_id, n):
    """Unique per request. run_id (random, one per client process) keeps two clients
    replaying the same file, or a rerun within the API's 60 s replay window, from sending
    the same key and getting 409s or replayed responses instead of real decisions."""
    return f"{run_id}-{user_id}-{n}"


async def send(client, row, url, idempotency_key):
    """Returns (outcome, latency_ms). outcome is the HTTP status, or the exception class name
    (e.g. 'ReadTimeout') so the summary can say what kind of failure it was."""
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=preprocess(row), headers={"Idempotency-Key": idempotency_key})
        outcome = r.status_code
    except Exception as e:
        outcome = type(e).__name__
    return outcome, (time.perf_counter() - t0) * 1000


async def main(rps=500, duration_seconds=30, source="fraud_data", url=API_URL,
               speedup=None, seed=None):
    rows = load_sparkov() if source == "sparkov" else load_csv()
    run_id = uuid.uuid4().hex[:8]
    seed = random.randrange(1 << 31) if seed is None else seed
    rng = random.Random(seed)

    start = time.monotonic()
    end = start + duration_seconds
    if speedup:
        schedule = replay_schedule(rows, speedup, start)
        print(f"replay mode: virtual clock x{speedup}, rows in purchase_time order (--rps ignored)")
    else:
        schedule = rate_schedule(rows, rps, rng, start)
        print(f"rate mode: Poisson arrivals, mean {rps}/s, seed={seed}")

    total_sent = 0
    skipped = 0      # due slots dropped because the loop fell > 1 s behind
    ok_total = 0
    errors = Counter()   # outcome -> count, for everything that was not a 201
    latencies = []       # every completed request, timeouts included
    next_progress = start + 1.0

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=400),
        timeout=30.0,
    ) as client:
        pending: set[asyncio.Task] = set()
        due, row = next(schedule)

        while time.monotonic() < end or pending:
            # Drain finished tasks
            done = {t for t in pending if t.done()}
            for t in done:
                pending.discard(t)
                outcome, ms = t.result()
                latencies.append(ms)
                if outcome == 201:
                    ok_total += 1
                else:
                    errors[outcome] += 1
                    if sum(errors.values()) <= 3:
                        print(f"  error: {outcome}")

            now = time.monotonic()
            if now >= next_progress:
                print(f"[{total_sent:>6} sent] ok={ok_total} err={sum(errors.values())} "
                      f"in-flight={len(pending)}")
                next_progress = now + 1.0

            if now < end and now >= due:
                # Skip slots more than 1 s overdue instead of queueing them, and yield so the
                # new request actually starts. Without both, a late loop queued requests back
                # to back without ever awaiting: none ran and the client froze.
                if due < now - 1.0:
                    skipped += 1
                else:
                    key = idempotency_key(run_id, row.get('user_id'), total_sent)
                    task = asyncio.create_task(
                        send(client, row, url, idempotency_key=key),
                        name=f"req-{total_sent}",
                    )
                    pending.add(task)
                    total_sent += 1
                    await asyncio.sleep(0)
                due, row = next(schedule)
            else:
                sleep_for = max(0.0, due - time.monotonic()) if now < end else 0.005
                await asyncio.sleep(min(sleep_for, 0.005))

        # Clean shutdown — cancel tasks still in flight after duration ends
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.monotonic() - start
    print("\n" + summary(rps if not speedup else None, total_sent, ok_total,
                         sum(errors.values()), skipped, elapsed, latencies, errors=errors))


def summary(rps, sent, ok, err, skipped, elapsed, latencies, errors=None):
    """Result lines. 'achieved' is successful requests per second of wall time (including
    the drain after the send window), so it shows what the API really kept up with.
    rps=None means replay mode (no target rate)."""
    target = f"target RPS={rps}  " if rps is not None else ""
    lines = [f"{target}achieved RPS={ok / max(elapsed, 1e-9):.0f}  "
             f"(ok={ok} in {elapsed:.1f}s)",
             f"  sent={sent} skipped={skipped} err={err} "
             f"({100 * ok / max(sent, 1):.1f}% success)"]
    if errors:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(errors.items(), key=lambda kv: -kv[1]))
        lines.append(f"  errors by kind: {kinds}")
    if latencies:
        lat = sorted(latencies)
        n = len(lat)
        lines.append(f"  p50={lat[n//2]:.0f}ms  p95={lat[int(n*.95)]:.0f}ms  p99={lat[int(n*.99)]:.0f}ms")
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the fraud API.")
    parser.add_argument("--source", choices=["fraud_data", "sparkov"], default="fraud_data",
                        help="sparkov: repeat customers, made-up fields (load test only)")
    parser.add_argument("--rps", type=int, default=400, help="mean requests per second (rate mode)")
    parser.add_argument("--duration", type=int, default=120, help="send window in seconds")
    parser.add_argument("--speedup", type=float, default=None,
                        help="replay mode: send rows in purchase_time order at this multiple of "
                             "real time (e.g. 1000). Ignores --rps.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the random arrival gaps; printed at start so a run can be repeated")
    parser.add_argument("--url", default=API_URL,
                        help="endpoint to load (e.g. a stub server, to find the client's own ceiling)")
    args = parser.parse_args()
    asyncio.run(main(rps=args.rps, duration_seconds=args.duration, source=args.source,
                     url=args.url, speedup=args.speedup, seed=args.seed))
