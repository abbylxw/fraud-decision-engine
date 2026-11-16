"""
Alert drill: prove that an attack on the running stack reaches the alerting system.

    docker compose up -d --build
    python scripts/alert_drill.py                 # ~3-4 minutes
    python scripts/alert_drill.py --wait-resolve  # also wait for the alert to clear (~6 more)

What it does:
  1. Sends a steady stream of bot signups (purchase 1 second after signup, fresh user and
     device each time). The signup_burst hard rule denies every one of them.
  2. Polls Prometheus and prints each change of state of the FraudSignupBurstAttack alert:
     inactive -> pending (the condition is true) -> firing (it stayed true for 2 minutes).
  3. Confirms the firing alert arrived in Alertmanager, with its severity and runbook link.

A drill like this is the only real test of alerting: the rule files can be perfect and the
alert still never arrive (rules not mounted, wrong Alertmanager address, a renamed metric).
It writes a few hundred denied rows to the audit table.
"""
import argparse
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

ALERT = "FraudSignupBurstAttack"


def bot_request(n: int, run: str) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return {"user_id": f"drill-{run}-{n}", "device_id": f"drill-dev-{run}-{n}",
            "signup_time": (now - timedelta(seconds=1)).strftime(fmt),
            "purchase_time": now.strftime(fmt), "purchase_value": 25.0,
            "source": "Ads", "browser": "Chrome", "sex": "M", "age": 28,
            "ip_address": 134744072}


def send_bots(api: str, rps: float, stop: threading.Event, counts: dict) -> None:
    run = uuid.uuid4().hex[:8]
    with httpx.Client(timeout=10) as client:
        n = 0
        while not stop.is_set():
            n += 1
            try:
                r = client.post(f"{api}/api/transactions", json=bot_request(n, run),
                                headers={"Idempotency-Key": f"drill-{run}-{n}"})
                counts[r.json().get("fraud_decision", r.status_code)] = \
                    counts.get(r.json().get("fraud_decision", r.status_code), 0) + 1
            except Exception as e:
                counts[type(e).__name__] = counts.get(type(e).__name__, 0) + 1
            time.sleep(1 / rps)


def alert_state(prom: str) -> tuple[str, dict]:
    alerts = httpx.get(f"{prom}/api/v1/alerts", timeout=5).json()["data"]["alerts"]
    for a in alerts:
        if a["labels"]["alertname"] == ALERT:
            return a["state"], a
    return "inactive", {}


def in_alertmanager(am: str) -> dict | None:
    for a in httpx.get(f"{am}/api/v2/alerts", timeout=5).json():
        if a["labels"]["alertname"] == ALERT and a["status"]["state"] == "active":
            return a
    return None


def watch(prom: str, target: str, timeout: float, t0: float) -> bool:
    last = None
    while time.time() - t0 < timeout:
        state, alert = alert_state(prom)
        if state != last:
            value = f"  signup_burst share of decisions = {float(alert['value']):.0%}" if alert else ""
            print(f"  [{time.time() - t0:5.0f} s] {ALERT}: {state}{value}")
            last = state
        if state == target:
            return True
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--prometheus", default="http://127.0.0.1:9090")
    ap.add_argument("--alertmanager", default="http://127.0.0.1:9093")
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--wait-resolve", action="store_true")
    args = ap.parse_args()

    for name, url in (("API", f"{args.api}/health"), ("Prometheus", f"{args.prometheus}/-/ready"),
                      ("Alertmanager", f"{args.alertmanager}/-/ready")):
        try:
            httpx.get(url, timeout=5).raise_for_status()
        except Exception as e:
            raise SystemExit(f"{name} not reachable at {url} ({e}). Start the stack with: "
                             f"docker compose up -d --build")

    print(f"Sending bot signups at {args.rps:g} per second (purchase 1 s after signup)...")
    stop, counts, t0 = threading.Event(), {}, time.time()
    sender = threading.Thread(target=send_bots, args=(args.api, args.rps, stop, counts), daemon=True)
    sender.start()
    fired = watch(args.prometheus, "firing", timeout=420, t0=t0)

    arrived = None
    if fired:
        for _ in range(12):   # Alertmanager's group_wait is 30 s
            arrived = in_alertmanager(args.alertmanager)
            if arrived:
                break
            time.sleep(5)
    stop.set()
    sender.join(timeout=5)
    print(f"  sent {sum(counts.values())} requests; API decisions: {counts}")

    if arrived:
        a = arrived
        print(f"  [{time.time() - t0:5.0f} s] in Alertmanager: severity={a['labels']['severity']}  "
              f"receiver={a['receivers'][0]['name']}")
        print(f"            summary: {a['annotations']['summary']}")
        print(f"            runbook: {a['annotations']['runbook_url']}")
    ok = bool(fired and arrived)
    print("\nDrill " + ("PASSED: the attack was denied by the rule, detected by Prometheus and "
                        "delivered to Alertmanager." if ok else
                        "FAILED: " + ("the alert fired but never reached Alertmanager."
                                      if fired else "the alert did not fire within 7 minutes.")))

    if ok and args.wait_resolve:
        print("Attack stopped; waiting for the alert to clear (the 5-minute rate has to drain)...")
        cleared = watch(args.prometheus, "inactive", timeout=600, t0=t0)
        print("  resolved." if cleared else "  still active after 10 minutes.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
