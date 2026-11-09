import argparse
import asyncio
import csv
import itertools
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx


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
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


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

async def send(client, row, url, idempotency_key):
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=preprocess(row), headers={"Idempotency-Key": idempotency_key})
        ms = (time.perf_counter() - t0) * 1000
        return r.status_code, ms
    except Exception:
        return 0, 0.0

API_URL = "http://127.0.0.1:8000/api/transactions"


async def main(rps=500, duration_seconds=30, source="fraud_data", url=API_URL):
    rows = load_sparkov() if source == "sparkov" else load_csv()
    infinite_rows = itertools.cycle(rows)

    # One request every 1/rps seconds — steady drip, not per-second bursts.
    interval = 1.0 / rps
    start_time = time.time()
    end_time = start_time + duration_seconds
    total_sent = 0
    skipped = 0      # send slots dropped because the loop fell > 1 s behind
    ok_total = 0
    err_total = 0
    latencies = []

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=400),
        timeout=30.0,
    ) as client:
        pending: set[asyncio.Task] = set()
        next_send = time.monotonic()

        while time.time() < end_time or pending:
            # Drain finished tasks
            done = {t for t in pending if t.done()}
            for t in done:
                pending.discard(t)
                status, ms = t.result()
                if status == 201:
                    ok_total += 1
                    latencies.append(ms)
                else:
                    err_total += 1
                    if err_total <= 3 or err_total % 200 == 0:
                        print(f"  error: HTTP {status}")

            # Schedule next request
            now = time.monotonic()
            if time.time() < end_time and now >= next_send:
                row = next(infinite_rows)
                key = f"{row.get('user_id')}-{total_sent}"
                task = asyncio.create_task(
                    send(client, row, url, idempotency_key=key),
                    name=f"req-{total_sent}",
                )
                pending.add(task)
                total_sent += 1
                # Progress line about once per second (here, so it prints once per send)
                if total_sent % rps == 0:
                    print(f"[{total_sent:>6} sent] ok={ok_total} err={err_total} in-flight={len(pending)}")
                # Skip slots more than 1 s overdue, and yield so the new request actually
                # starts. Without both, a late loop queued requests back to back without
                # ever awaiting: none ran, the scan above grew, and the client froze.
                if now - 1.0 > next_send + interval:
                    skipped += int((now - 1.0 - next_send) / interval)
                next_send = max(next_send + interval, now - 1.0)
                await asyncio.sleep(0)
            else:
                sleep_for = max(0.0, next_send - time.monotonic())
                await asyncio.sleep(min(sleep_for, 0.005))

        # Clean shutdown — cancel tasks still in flight after duration ends
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.time() - start_time
    print("\n" + summary(rps, total_sent, ok_total, err_total, skipped, elapsed, latencies))


def summary(rps, sent, ok, err, skipped, elapsed, latencies):
    """Result lines. 'achieved' is successful requests per second of wall time (including
    the drain after the send window), so it shows what the API really kept up with."""
    lines = [f"target RPS={rps}  achieved RPS={ok / max(elapsed, 1e-9):.0f}  "
             f"(ok={ok} in {elapsed:.1f}s)",
             f"  sent={sent} skipped={skipped} err={err} "
             f"({100 * ok / max(sent, 1):.1f}% success)"]
    if latencies:
        lat = sorted(latencies)
        n = len(lat)
        lines.append(f"  p50={lat[n//2]:.0f}ms  p95={lat[int(n*.95)]:.0f}ms  p99={lat[int(n*.99)]:.0f}ms")
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the fraud API.")
    parser.add_argument("--source", choices=["fraud_data", "sparkov"], default="fraud_data",
                        help="sparkov: repeat customers, made-up fields (load test only)")
    parser.add_argument("--rps", type=int, default=400, help="target requests per second")
    parser.add_argument("--duration", type=int, default=120, help="send window in seconds")
    parser.add_argument("--url", default=API_URL,
                        help="endpoint to load (e.g. a stub server, to find the client's own ceiling)")
    args = parser.parse_args()
    asyncio.run(main(rps=args.rps, duration_seconds=args.duration, source=args.source, url=args.url))
