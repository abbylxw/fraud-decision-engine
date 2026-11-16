"""
A guided demo of the fraud decision engine against the running Docker stack.

    docker compose up -d --build        # API on :8000, Grafana on :3000
    python scripts/demo.py              # eight stories, then a short traffic replay
    python scripts/demo.py --no-replay  # stories only

Each story sends one or two real requests to POST /api/transactions and prints what the
engine decided and why, read back from the audit table (the row every decision writes).
Together they show each layer: hard rules, the model's review band, and idempotency.

    1. A normal customer                        -> allow
    2. A bot: purchase 1 second after signup    -> deny   (hard rule signup_burst)
    3. A second account on one device in 1 hour -> deny   (hard rule device_many_users_1h)
    4. A request from a sanctioned country      -> deny   (hard rule ip_country_block)
    5. A device that comes back 120 days later
       with a different account                 -> review (model, feature did_other_uid_all)
    6. A loyal customer, third purchase on
       their own device                         -> allow  (history, but no OTHER account)
    7. A device six other accounts have used,
       spread over months                       -> review (the model must not relax as
                                                   the count grows: monotone constraint)
    8. The same request sent twice              -> one decision, served from cache

Story 5 returns "review" only when the served model has a review band (model_meta.json
has review_threshold); with an older model it prints the score and says so.

Every run uses fresh user and device ids, so it can be repeated. It writes ~10 audit rows
and, with the replay, about 2,000 more.
"""
import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
US_IP = 134744072        # 8.8.8.8, United States in the IP table
CUBA_IP = 2563637249     # inside a Cuban range of the IP table
AUDIT_COLUMNS = ("fraud_decision", "decision_rule", "model_score", "ip_country",
                 "did_uid_1h", "did_txn_cnt_all", "did_other_uid_all",
                 "amount_vs_user_avg_30d", "model_version")
BAR = "-" * 78


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def payload(user, device, purchase_time, signup_before, ip=US_IP, value=42.0):
    return {
        "user_id": user, "device_id": device,
        "signup_time": iso(purchase_time - signup_before),
        "purchase_time": iso(purchase_time),
        "purchase_value": value, "source": "SEO", "browser": "Chrome",
        "sex": "F", "age": 34, "ip_address": ip,
    }


def audit_row(transaction_id: str) -> dict | None:
    """The decision's audit row, read with psql inside the db container. Returns None if
    Docker or the container is not reachable; the demo then prints decisions only."""
    sql = (f"SELECT row_to_json(t) FROM (SELECT {', '.join(AUDIT_COLUMNS)} FROM transactions "
           f"WHERE transaction_id = '{uuid.UUID(transaction_id)}') t;")
    try:
        out = subprocess.run(
            ["docker", "compose", "exec", "-T", "db", "psql", "-U", "postgres", "-d", "fraud",
             "-At", "-c", sql],
            cwd=ROOT, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        return json.loads(out) if out else None
    except Exception:
        return None


class Demo:
    def __init__(self, url: str):
        self.client = httpx.Client(timeout=10)
        self.url = url
        self.run = uuid.uuid4().hex[:8]
        self.results = []

    def send(self, body: dict, key: str | None = None) -> dict:
        key = key or f"demo-{self.run}-{uuid.uuid4().hex[:8]}"
        r = self.client.post(self.url, json=body, headers={"Idempotency-Key": key})
        r.raise_for_status()
        # Velocity counters are written just after the response; give them a moment so
        # the next request of the same story sees this one.
        time.sleep(0.4)
        return r.json()

    def show(self, label: str, body: dict, resp: dict, expected: str, why: str):
        row = audit_row(resp["transaction_id"]) or {}
        decision = resp["fraud_decision"]
        ok = decision == expected
        self.results.append((label, expected, decision, ok))
        score = row.get("model_score")
        print(f"  request   user={body['user_id']}  device={body['device_id']}  "
              f"${body['purchase_value']:.0f}  signup {body['signup_time']}  "
              f"purchase {body['purchase_time']}")
        print(f"  decision  {decision.upper():<7} rule={row.get('decision_rule') or '-'}  "
              f"model_score={'not run (hard rule)' if score is None and row else score}")
        if row:
            print(f"  features  ip_country={row['ip_country']}  did_uid_1h={row['did_uid_1h']}  "
                  f"did_txn_cnt_all={row['did_txn_cnt_all']}  "
                  f"did_other_uid_all={row['did_other_uid_all']}  "
                  f"amount_vs_own_avg={row['amount_vs_user_avg_30d']:.2f}  model={row['model_version']}")
        print(f"  why       {why}")
        if not ok:
            print(f"  NOTE      expected {expected}, got {decision}")

    def story(self, n: int, title: str):
        print(f"\n{BAR}\n{n}. {title}\n{BAR}")


def run_stories(d: Demo):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    r = d.run

    d.story(1, "A normal customer: account 30 days old, new device, US IP")
    body = payload(f"u-normal-{r}", f"d-normal-{r}", now, timedelta(days=30))
    first = d.send(body, key=f"demo-{r}-normal")
    d.show("normal customer", body, first, "allow",
           "No hard rule fires and the model score is below the review threshold.")

    d.story(2, "A bot: the purchase arrives 1 second after signup")
    body = payload(f"u-bot-{r}", f"d-bot-{r}", now, timedelta(seconds=1))
    d.show("bot signup", body, d.send(body), "deny",
           "signup_burst: under 60 s from signup to purchase is scripted. In the training data "
           "all 7,600 such purchases were fraud; the fastest real customer took 137 s.")

    d.story(3, "Two different accounts on one device within the hour")
    a = payload(f"u-share-a-{r}", f"d-share-{r}", now - timedelta(minutes=10), timedelta(days=20))
    d.send(a)
    print("  (first account bought on this device 10 minutes earlier: allowed)")
    b = payload(f"u-share-b-{r}", f"d-share-{r}", now, timedelta(days=15))
    d.show("shared device", b, d.send(b), "deny",
           "device_many_users_1h: did_uid_1h counts distinct accounts on the device in the "
           "last hour, including this one. Two means the device is being passed around.")

    d.story(4, "A request from a comprehensively sanctioned country")
    body = payload(f"u-ofac-{r}", f"d-ofac-{r}", now, timedelta(days=45), ip=CUBA_IP)
    d.show("sanctioned country", body, d.send(body), "deny",
           "ip_country_block: a compliance rule, not a fraud score. The IP resolves to Cuba in "
           "the 5M-row IP table; the model never runs.")

    d.story(5, "A device that returns after 120 days with a different account")
    old = payload(f"u-old-{r}", f"d-return-{r}", now - timedelta(days=120), timedelta(days=30))
    d.send(old)
    print("  (a first account bought on this device 120 days ago: allowed)")
    new = payload(f"u-new-{r}", f"d-return-{r}", now, timedelta(days=25))
    d.show("returning device", new, d.send(new), "review",
           "The 90-day counters have forgotten the device; did_other_uid_all has not: one "
           "OTHER account used it before. Such devices were 5% of traffic and 23% fraud in "
           "training: too risky to approve, not risky enough to decline, so the caller is "
           "asked to step up or review.")

    d.story(6, "A loyal customer: third purchase on their own device")
    for days in (60, 30):
        d.send(payload(f"u-loyal-{r}", f"d-loyal-{r}", now - timedelta(days=days),
                       timedelta(days=200 - days)))
    print("  (the same account bought on this device 60 and 30 days ago: allowed)")
    body = payload(f"u-loyal-{r}", f"d-loyal-{r}", now, timedelta(days=200))
    d.show("loyal customer", body, d.send(body), "allow",
           "The device has history (did_txn_cnt_all=2) but no OTHER account has used it "
           "(did_other_uid_all=0), and the purchase is the customer's usual size "
           "(amount_vs_own_avg=1.00). Two earlier models got this customer wrong: one counted "
           "their own purchases as device risk and sent them to review, the next read the "
           "ratio of 1 as a one-dollar purchase. Neither input reaches the model now "
           "(docs/feature_audit.md).")

    d.story(7, "A device that six other accounts have used over the past months")
    for i, days in enumerate((220, 190, 160, 130, 110, 95)):
        d.send(payload(f"u-farm-{i}-{r}", f"d-farm-{r}", now - timedelta(days=days),
                       timedelta(days=20)))
    print("  (six different accounts bought on this device, 95 to 220 days ago: no rule fired,")
    print("   they were more than an hour apart)")
    body = payload(f"u-farm-new-{r}", f"d-farm-{r}", now, timedelta(days=20))
    d.show("device farm, slow", body, d.send(body), "review",
           "Too slow for the 1-hour hard rule, and outside every 90-day counter. The first "
           "model scored five or more other accounts as LOW risk (0.52, allow), because in "
           "training such devices were leftovers of the January bot attack. The served model "
           "is constrained so that more other accounts can never lower the score.")

    d.story(8, "The same request sent twice (a client retry after a timeout)")
    body = payload(f"u-normal-{r}", f"d-normal-{r}", now, timedelta(days=30))
    again = d.send(body, key=f"demo-{r}-normal")
    same = again["transaction_id"] == first["transaction_id"]
    d.results.append(("idempotent retry", "same id", "same id" if same else "new id", same))
    print(f"  first  transaction_id={first['transaction_id']}  {first['fraud_decision']}")
    print(f"  retry  transaction_id={again['transaction_id']}  {again['fraud_decision']}")
    print("  why       The Idempotency-Key is claimed in Redis before processing, so a retry "
          "gets the stored answer and no second audit row or velocity update.")


def summary(d: Demo) -> bool:
    print(f"\n{BAR}\nSummary\n{BAR}")
    for label, expected, got, ok in d.results:
        print(f"  {'ok ' if ok else 'DIFF'}  {label:<20} expected {expected:<8} got {got}")
    return all(ok for *_, ok in d.results)


def replay(rps: int, seconds: int):
    print(f"\n{BAR}\nReplaying real transactions for {seconds} s at ~{rps} requests/s "
          f"so the dashboards fill\n{BAR}")
    subprocess.run([sys.executable, str(ROOT / "app/data/client_mimic.py"),
                    "--rps", str(rps), "--duration", str(seconds)], cwd=ROOT, check=False)
    print("\nOpen Grafana: http://localhost:3000  (dashboard 'Fraud Engine', row "
          "'Decisions and model')")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000/api/transactions")
    ap.add_argument("--no-replay", action="store_true", help="skip the traffic replay")
    ap.add_argument("--rps", type=int, default=50)
    ap.add_argument("--seconds", type=int, default=40)
    args = ap.parse_args()

    d = Demo(args.url)
    try:
        d.client.get(args.url.replace("/api/transactions", "/health")).raise_for_status()
    except Exception as e:
        raise SystemExit(f"API not reachable at {args.url} ({e}). Start it with: "
                         f"docker compose up -d --build")
    run_stories(d)
    ok = summary(d)
    if not args.no_replay:
        replay(args.rps, args.seconds)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
