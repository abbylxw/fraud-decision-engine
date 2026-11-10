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
    from app.main import _WORKER_REQUESTS, REQUESTS_TOTAL
    for _ in range(3):
        REQUESTS_TOTAL.inc()
        _WORKER_REQUESTS.inc()
""")

SCRAPE = textwrap.dedent("""
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).get("/metrics/")
    assert r.status_code == 200, r.status_code
    line = [x for x in r.text.splitlines() if x.startswith("fraud_requests_total ")]
    print(line[0].split()[1])
""")

SCRAPE_PER_WORKER = textwrap.dedent("""
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).get("/metrics/")
    for x in r.text.splitlines():
        if x.startswith("fraud_worker_requests_total{"):
            print(x.split()[1])
""")


def _run(code, env):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_metrics_sum_across_worker_processes(tmp_path):
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    _run(WORKER, env)
    _run(WORKER, env)
    assert float(_run(SCRAPE, env)) == 6.0


def test_worker_counter_keeps_one_series_per_worker(tmp_path):
    # The total is summed across workers, but the worker-labelled counter must not be:
    # Grafana's "Requests per Worker" panel needs one series per process.
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    _run(WORKER, env)
    _run(WORKER, env)
    # The scraping process imports the app too, so it shows up as a third worker with 0.
    assert sorted(float(v) for v in _run(SCRAPE_PER_WORKER, env).split()) == [0.0, 3.0, 3.0]
