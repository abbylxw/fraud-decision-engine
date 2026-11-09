"""
/metrics must add up all uvicorn workers. Each worker is its own process with its own
counters; without multiprocess mode a scrape only sees the one worker that answered it.

PROMETHEUS_MULTIPROC_DIR has to be set before prometheus_client is imported, so each check
runs in a fresh Python process. Two "workers" each count 3 requests into the same folder;
a scrape from either one must report 6.
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER = textwrap.dedent("""
    from app.main import REQUESTS_TOTAL
    for _ in range(3):
        REQUESTS_TOTAL.inc()
""")

SCRAPE = textwrap.dedent("""
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).get("/metrics/")
    assert r.status_code == 200, r.status_code
    line = [x for x in r.text.splitlines() if x.startswith("fraud_requests_total ")]
    print(line[0].split()[1])
""")


def _run(code, env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_metrics_sum_across_worker_processes(tmp_path):
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    _run(WORKER, env)
    _run(WORKER, env)
    assert float(_run(SCRAPE, env)) == 6.0
