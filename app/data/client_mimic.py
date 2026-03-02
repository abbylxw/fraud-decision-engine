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
    
async def send(client, row, url):
    r = await client.post(url, json=row)
    return r.status_code

async def main(rps=50, duration_seconds=120):
    rows = load_csv()
    infinite_rows = itertools.cycle(rows)
    url = "http://127.0.0.1:8000/api/transactions"
    #url = "https://httpbin.org/post"
    
    end_time = time.time() + duration_seconds
    total_sent = 0

    async with httpx.AsyncClient() as client:
        while time.time() < end_time:
            # grab a batch equal to desired rps
            batch = [next(infinite_rows) for _ in range(rps)]
            tasks = [send(client, row, url) for row in batch]
            
            start = time.time()
            await asyncio.gather(*tasks)
            total_sent += len(batch)
            
            # sleep remaining time to stay at target rps
            elapsed = time.time() - start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            
            print(f"Sent {total_sent} total | {rps}/s target")

    print(f"Done: {total_sent} requests in {duration_seconds}s")

asyncio.run(main(rps=50, duration_seconds=120))


