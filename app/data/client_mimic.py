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
    r = await client.post(url, json=preprocess(row), headers={"Idempotency-Key": idempotency_key})
    try:
        body = r.json()
    except Exception:
        body = {"_raw": r.text}
    return row.get("user_id", "?"), r.status_code, body

async def main(rps=20, duration_seconds=float("inf")):
    rows = load_csv()
    infinite_rows = itertools.cycle(rows)
    url = "http://127.0.0.1:8000/api/transactions"
    #url = "https://httpbin.org/post"

    end_time = time.time() + duration_seconds
    total_sent = 0

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
        timeout=30.0,
    ) as client:
        while time.time() < end_time:
            # grab a batch equal to desired rps
            batch = [next(infinite_rows) for _ in range(rps)]
            tasks = [
                send(client, row, url, idempotency_key=f"{row.get('user_id')}-{total_sent+i}")
                for i, row in enumerate(batch)
            ]

            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_sent += len(batch)

            ok = sum(1 for r in results if not isinstance(r, Exception) and r[1] == 201)
            errors = [r for r in results if isinstance(r, Exception) or r[1] != 201]
            err = len(errors)
            print(f"--- {total_sent} sent | {ok} ok, {err} errors ---")
            if errors:
                sample = errors[0]
                if isinstance(sample, Exception):
                    print(f"    sample error: {type(sample).__name__}: {sample}")
                else:
                    _, status, body = sample
                    print(f"    sample error: HTTP {status} — {body}")

            # sleep remaining time to stay at target rps
            elapsed = time.time() - start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

    print(f"Done: {total_sent} requests in {duration_seconds}s")

asyncio.run(main(rps=20))
