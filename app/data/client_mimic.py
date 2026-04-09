import csv
from pathlib import Path
import itertools
import time
import httpx
import asyncio

def load_csv():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "Fraud_Data.csv"
    with open(DATA_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

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
    except Exception as e:
        return 0, 0.0

async def main(rps=500, duration_seconds=30):
    rows = load_csv()
    infinite_rows = itertools.cycle(rows)
    url = "http://127.0.0.1:8000/api/transactions"

    # One request every 1/rps seconds — steady drip, not per-second bursts.
    interval = 1.0 / rps
    end_time = time.time() + duration_seconds
    total_sent = 0
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

            # Progress line once per second
            if total_sent > 0 and total_sent % rps == 0:
                print(f"[{total_sent:>6} sent] ok={ok_total} err={err_total} in-flight={len(pending)}")

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
                next_send += interval
            else:
                sleep_for = max(0.0, next_send - time.monotonic())
                await asyncio.sleep(min(sleep_for, 0.005))

        # Clean shutdown — cancel tasks still in flight after duration ends
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    latencies.sort()
    n = len(latencies)
    print(f"\nRPS={rps} | sent={total_sent} ok={ok_total} err={err_total} ({100*ok_total/max(total_sent,1):.1f}% success)")
    if latencies:
        print(f"  p50={latencies[n//2]:.0f}ms  p95={latencies[int(n*.95)]:.0f}ms  p99={latencies[int(n*.99)]:.0f}ms")

asyncio.run(main(rps=400, duration_seconds=120))
