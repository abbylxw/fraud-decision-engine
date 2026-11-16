"""
Monitoring: the drift arithmetic, the metrics the API exports, and the alert rules file.

The alert EXPRESSIONS are tested by promtool (prometheus/tests/alerts_test.yml, run in CI).
Here: the Python side that feeds them, and that every alert is something a person can act
on (severity, summary, and a runbook section that exists).
"""
import pathlib
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fakeredis import aioredis as fake_aioredis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import telemetry
from app.data.data_models import Features, Transaction
from app.decisions.features import build_features
from app.main import app, get_db, get_redis
from app.monitoring import bucket_index, bucket_shares, psi, reference_buckets

ROOT = pathlib.Path(__file__).resolve().parents[1]
RULES = yaml.safe_load((ROOT / "prometheus/rules/fraud_engine.yml").read_text())
ALERTS = [r for g in RULES["groups"] for r in g["rules"] if "alert" in r]


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ---------------------------------------------------------------------------
# Buckets and PSI
# ---------------------------------------------------------------------------

class TestBuckets:
    @pytest.mark.parametrize("x, bucket", [(0.0, 0), (0.999, 0), (1.0, 1), (1.5, 1), (2.0, 2), (99, 2)])
    def test_an_edge_belongs_to_the_bucket_above_it(self, x, bucket):
        assert bucket_index([1.0, 2.0], x) == bucket

    def test_no_edges_is_one_bucket(self):
        assert bucket_index([], 123.0) == 0

    def test_deciles_of_a_spread_out_score(self):
        ref = reference_buckets([i / 1000 for i in range(1000)], 10)
        assert len(ref["edges"]) == 9
        assert all(abs(s - 0.1) < 0.005 for s in ref["shares"])

    def test_a_mostly_zero_count_keeps_its_rare_values_apart(self):
        # did_other_uid_all: 0 for 95% of traffic. Quantiles alone would give one bucket.
        ref = reference_buckets([0] * 9500 + [1] * 450 + [2] * 50, 5)
        assert ref["edges"] == [1.0]                 # {0} and {1 or more}
        assert ref["shares"] == [0.95, 0.05]

    def test_buckets_under_one_percent_are_merged(self):
        ref = reference_buckets([0] * 9000 + [1] * 900 + [2] * 95 + [3] * 5, 5)
        assert min(ref["shares"]) >= 0.0095
        assert abs(sum(ref["shares"]) - 1) < 1e-6


class TestPsi:
    def test_identical_distributions_score_zero(self):
        assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0

    def test_known_value(self):
        # (0.1-0.5)*ln(0.2) + (0.9-0.5)*ln(1.8): the same case promtool checks in PromQL.
        assert psi([0.5, 0.5], [0.1, 0.9]) == pytest.approx(0.8789, abs=1e-4)

    def test_empty_live_bucket_is_finite_and_large(self):
        value = psi([0.95, 0.05], [0.0, 1.0])
        assert 2 < value < 20

    def test_small_shift_stays_under_the_watch_level(self):
        ref = [0.1] * 10
        live = bucket_shares([i / 10 for i in range(1, 10)], [(i % 1000) / 1000 * 0.98 for i in range(5000)])
        assert psi(ref, live) < 0.10


# ---------------------------------------------------------------------------
# Metrics the API exports
# ---------------------------------------------------------------------------

def _features(**overrides) -> Features:
    t = Transaction(user_id="u", signup_time="2024-01-01T00:00:00Z",
                    purchase_time="2025-03-14T12:00:00Z", purchase_value=40.0, device_id="d",
                    source="SEO", browser="Chrome", sex="M", age=30, ip_address=1)
    return Features(**t.model_dump(by_alias=False)).model_copy(update=overrides)


class TestDriftCounters:
    def test_reference_is_exported_and_every_bucket_has_a_series(self):
        if not telemetry._edges:
            pytest.skip("served model has no monitoring reference")
        for name, edges in telemetry._edges.items():
            shares = [sample("fraud_drift_reference_share", feature=name, bucket=str(i))
                      for i in range(len(edges) + 1)]
            assert abs(sum(shares) - 1) < 1e-3, name
            for i in range(len(edges) + 1):
                # Created at 0 at startup: PSI needs the empty buckets too.
                assert REGISTRY.get_sample_value(
                    "fraud_drift_bucket_total", {"feature": name, "bucket": str(i)}) is not None

    def test_a_scored_request_lands_in_one_bucket_per_monitored_item(self):
        if not telemetry._edges:
            pytest.skip("served model has no monitoring reference")
        def totals():
            return {name: sum(sample("fraud_drift_bucket_total", feature=name, bucket=str(i))
                              for i in range(len(edges) + 1))
                    for name, edges in telemetry._edges.items()}
        before = totals()
        top = str(len(telemetry._edges[telemetry.SCORE]))
        top_before = sample("fraud_drift_bucket_total", feature=telemetry.SCORE, bucket=top)
        telemetry.observe_drift(_features(did_other_uid_all=3), 0.999)
        assert totals() == {k: v + 1 for k, v in before.items()}
        assert sample("fraud_drift_bucket_total", feature=telemetry.SCORE, bucket=top) == top_before + 1

    def test_a_broken_feature_never_fails_the_request(self):
        telemetry.observe_drift(object(), 0.5)   # no attributes at all: logged, not raised

    def test_model_version_is_exported(self):
        from app.decisions.models import MODEL_VERSION
        assert sample("fraud_model_info", version=MODEL_VERSION) == 1


class TestFallbackCounter:
    async def test_empty_redis_counts_each_postgres_fallback(self):
        redis = fake_aioredis.FakeRedis(decode_responses=True)
        result = MagicMock()
        result.all.return_value = []
        result.scalar_one.return_value = 0
        result.scalar_one_or_none.return_value = "United States"
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)
        sources = ("ip_country", "device_pair", "purchase_history", "device_history")
        before = {s: sample("fraud_feature_fallback_total", source=s) for s in sources}
        t = Transaction(user_id="u-fb", signup_time="2024-01-01T00:00:00Z",
                        purchase_time="2025-03-14T12:00:00Z", purchase_value=40.0,
                        device_id="d-fb", source="SEO", browser="Chrome", sex="M", age=30,
                        ip_address=1234567890)
        await build_features(t, db, redis)
        assert {s: sample("fraud_feature_fallback_total", source=s) for s in sources} == \
               {s: v + 1 for s, v in before.items()}
        await redis.aclose()


class TestErrorCounter:
    def test_an_unhandled_error_is_counted_and_the_key_released(self):
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        app.dependency_overrides[get_db] = lambda: AsyncMock()
        app.dependency_overrides[get_redis] = lambda: redis
        before = sample("fraud_request_errors_total", kind="error")
        payload = {"user_id": "u", "signup_time": "2024-01-01T00:00:00Z",
                   "purchase_time": "2025-03-14T12:00:00Z", "purchase_value": 10.0,
                   "device_id": "d", "source": "SEO", "browser": "Chrome", "sex": "M",
                   "age": 30, "ip_address": 1}
        try:
            with patch("app.main.build_features", new_callable=AsyncMock,
                       side_effect=RuntimeError("redis exploded")):
                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.post("/api/transactions", json=payload,
                                       headers={"Idempotency-Key": "err-test"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 500
        assert sample("fraud_request_errors_total", kind="error") == before + 1
        redis.delete.assert_awaited_with("idem:err-test")


class TestGoldenSignals:
    """Latency and traffic are the request histogram and counter that already existed.
    These cover the two that were missing: errors by status code, and saturation."""

    def _post(self, payload, key):
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        app.dependency_overrides[get_db] = lambda: AsyncMock()
        app.dependency_overrides[get_redis] = lambda: redis
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                return client.post("/api/transactions", json=payload,
                                   headers={"Idempotency-Key": key})
        finally:
            app.dependency_overrides.clear()

    def test_an_invalid_request_is_counted_as_422(self):
        # Validation rejects it before the handler runs, so only the middleware sees it.
        before = sample("fraud_http_responses_total", status="422")
        resp = self._post({"user_id": "u"}, "bad-payload")
        assert resp.status_code == 422
        assert sample("fraud_http_responses_total", status="422") == before + 1

    def test_a_failure_of_ours_is_counted_as_500_and_nothing_stays_in_flight(self):
        payload = {"user_id": "u", "signup_time": "2024-01-01T00:00:00Z",
                   "purchase_time": "2025-03-14T12:00:00Z", "purchase_value": 10.0,
                   "device_id": "d", "source": "SEO", "browser": "Chrome", "sex": "M",
                   "age": 30, "ip_address": 1}
        before = sample("fraud_http_responses_total", status="500")
        with patch("app.main.build_features", new_callable=AsyncMock, side_effect=RuntimeError("x")):
            resp = self._post(payload, "boom")
        assert resp.status_code == 500
        assert sample("fraud_http_responses_total", status="500") == before + 1
        assert sample("fraud_inflight_requests") == 0

    def test_other_paths_are_not_counted(self):
        total = lambda: sum(s.value for m in REGISTRY.collect() for s in m.samples  # noqa: E731
                            if s.name == "fraud_http_responses_total")
        before = total()
        with TestClient(app) as client:
            client.get("/metrics/")
        assert total() == before

    async def test_event_loop_lag_is_seen_when_the_loop_is_blocked(self):
        import asyncio
        import time
        telemetry._LoopLag.worst = 0.0
        watcher = asyncio.create_task(telemetry.watch_event_loop())
        await asyncio.sleep(0.05)
        time.sleep(0.7)            # block the loop: the watcher's 0.5 s sleep wakes up late
        await asyncio.sleep(0.05)
        watcher.cancel()
        assert telemetry._LoopLag.worst > 0.1

    def test_memory_reading_is_plausible(self):
        assert 10 * 2**20 < telemetry._memory_bytes() < 64 * 2**30


# ---------------------------------------------------------------------------
# The alert rules are actionable
# ---------------------------------------------------------------------------

class TestAlertRules:
    def test_there_are_alerts(self):
        assert len(ALERTS) >= 29

    @pytest.mark.parametrize("rule", ALERTS, ids=lambda r: r["alert"])
    def test_alert_can_be_acted_on(self, rule):
        assert rule["labels"]["severity"] in ("critical", "warning", "info")
        assert rule["alert"].startswith("Fraud")
        a = rule["annotations"]
        assert len(a["summary"]) > 15 and len(a["description"]) > 30
        # A condition must hold for a while before anyone is told.
        assert "for" in rule

    @pytest.mark.parametrize("rule", ALERTS, ids=lambda r: r["alert"])
    def test_runbook_section_exists(self, rule):
        doc, _, anchor = rule["annotations"]["runbook_url"].partition("#")
        text = (ROOT / doc).read_text()
        anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).strip().replace(" ", "-")
                   for h in re.findall(r"^#{2,4} +(.+)$", text, flags=re.M)}
        assert anchor in anchors, f"{doc} has no section for {anchor}"

    def test_every_metric_in_the_rules_is_exported_or_recorded(self):
        import app.main  # noqa: F401  (registers the request metrics)
        exported = {m.name for m in REGISTRY.collect()}
        exported |= {s.name for m in REGISTRY.collect() for s in m.samples}
        recorded = {r["record"] for g in RULES["groups"] for r in g["rules"] if "record" in r}
        used = set()
        for g in RULES["groups"]:
            for r in g["rules"]:
                expr = re.sub(r'"[^"]*"', "", r["expr"])   # label values are not metrics
                used |= set(re.findall(r"\b(fraud[:_][a-zA-Z0-9_:]+)", expr))
        unknown = {u for u in used if u not in exported and u not in recorded
                   and not any(u == e or u.startswith(e + "_") or e.startswith(u) for e in exported)}
        assert not unknown, f"rules use metrics nobody produces: {sorted(unknown)}"

    def test_ratio_alerts_have_a_traffic_guard(self):
        guards = ("fraud:requests:rate5m", "fraud:decisions:rate5m", "fraud:scored:increase1h",
                  "increase(fraud_decisions_total")
        for rule in ALERTS:
            if re.search(r"(ratio|share|psi|latency)", rule["expr"]):
                assert any(g in rule["expr"] for g in guards), rule["alert"]
