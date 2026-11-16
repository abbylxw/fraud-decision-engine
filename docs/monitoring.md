# Monitoring and alerting

What is measured, what wakes someone up, and what to do when it does. The alert rules link
to the sections of this page, so it doubles as the runbook.

- Metrics: exported by the API at `/metrics/` (all eight workers summed).
- Rules: [`prometheus/rules/fraud_engine.yml`](../prometheus/rules/fraud_engine.yml),
  unit-tested by [`prometheus/tests/alerts_test.yml`](../prometheus/tests/alerts_test.yml).
- Routing: [`alertmanager/alertmanager.yml`](../alertmanager/alertmanager.yml).
- Dashboards: Grafana, dashboard "Fraud Engine", rows "Decisions and model" and
  "Health and drift".

Local addresses: Prometheus `localhost:9090` (Alerts and Rules pages), Alertmanager
`localhost:9093`, Grafana `localhost:3000`.

## 1. What can go wrong, and how each failure is seen

A fraud engine fails in three different ways, and each needs its own kind of signal.

| Failure | Example | Who notices without monitoring | Signal here |
|---|---|---|---|
| **It is broken** | API down, Postgres unreachable, every request a 500 | The caller, immediately | Scrape health, dependency probes, error counter, latency |
| **It is quietly wrong** | Redis lost its data, so every counter reads zero and the model sees a first-time customer everywhere | Nobody, for days: decisions are still returned | Redis key count, Postgres-fallback rate, review rate against its expected value |
| **The world changed** | A new attack, a client sending device ids in a new format, a seasonal shift | Nobody, until chargebacks arrive weeks later | Rule fire rates, score drift, feature drift, shared-device share |

The second and third rows are why "the dashboard looks fine" is not enough for a model in
production: the service is healthy and the decisions are bad.

### What is not covered

**There is no live measure of whether decisions are right.** That needs labels (which
orders turned out to be fraud), and there is no label endpoint yet. Drift alerts say "the
inputs or the scores no longer look like training time"; they do not say "the model is
wrong", and a model can also become wrong with no drift at all, if fraudsters change what
fraud looks like while the traffic mix stays the same. Closing this is the next roadmap item
(`docs/production_readiness.md`).

## 2. Metrics added for alerting

| Metric | Meaning |
|---|---|
| `fraud_request_errors_total{kind}` | Requests that ended in an unhandled error (`error`) or a client that went away (`cancelled`). A 500 used to be invisible. |
| `fraud_dependency_up{dependency}` | 1 if the API can reach Postgres / Redis. Each worker probes every 10 s, independent of traffic; the worst live worker is reported. |
| `fraud_redis_keys` | Number of keys in Redis. All velocity state lives there. |
| `fraud_feature_fallback_total{source}` | Feature lookups answered by Postgres because Redis had nothing. |
| `fraud_model_info{version}` | Which model version is serving. |
| `fraud_expected_rate{kind}` | Review rate, deny rate and shared-device share measured on validation data at training time. |
| `fraud_drift_bucket_total{feature,bucket}` | Scored requests by bucket of the model score (`feature="model_score"`) or of a monitored input. |
| `fraud_drift_reference_share{feature,bucket}` | The share of traffic training expected in that bucket. |
| `fraud_model_threshold{band}`, `fraud_shared_device_total{decision}` | The served thresholds; requests from devices other accounts have used. |

### How drift is measured

At training time `training/train.py` cuts the validation scores into ten equal-sized
buckets (deciles) and each monitored feature into up to five, merging any bucket expected
to hold under 1% of traffic, and writes the cut points and expected shares to
`model_meta.json` (`monitoring_reference`). At serving time each scored request adds one to
the counter of the bucket it falls in. Prometheus turns the counters into live shares over
the last hour and compares them with the expected shares using the **Population Stability
Index**:

    PSI = sum over buckets of (live share − expected share) × ln(live share / expected share)

PSI is 0 when the two distributions are identical and grows as they move apart. The usual
reading, used for the alert levels: under 0.10 no real change, 0.10 to 0.25 worth watching,
over 0.25 a shift that should be acted on. Monitored inputs: `did_other_uid_all`,
`did_uid_90d`, `did_uid_30d`, `purchase_value`, `days_since_signup`, `age_account_gap`
(`app/monitoring.py`).

Design choice: drift is computed in PromQL from counters, not by a nightly batch job over
the audit table. It needs no extra service, updates every minute and is alertable with the
same tooling as everything else. The cost is coarseness (a handful of buckets per feature,
only the monitored features) and a one-hour memory. A batch job over the audit table can
look at every feature and at joint distributions; it is the right addition once labels
exist.

## 3. Alerts

Severity: **critical** = someone should act now; **warning** = look today; **info** =
context, routed nowhere. Every ratio alert also requires a minimum amount of traffic, so a
quiet system does not alert on three requests. Thresholds are starting points chosen from
this project's replays, with the reasoning given; a real deployment would tune them on its
own traffic for a few weeks.

### Availability

#### FraudApiDown
*Critical. Prometheus cannot scrape the API for 2 minutes.*
The API is down or unreachable; callers get no decisions. While this fires Alertmanager
mutes every other fraud alert, because they would all be consequences.
Check first: `docker compose ps api` and `docker compose logs api --tail 100`; whether the
model file loads (a bad `model.joblib` fails at import: `tests/test_served_model.py`
exists for this); whether the last deploy changed the image.

#### FraudDependencyDown
*Critical. A worker's Postgres or Redis probe has failed for 1 minute.*
Without Postgres no decision can be written, so requests fail. Without Redis every request
fails too (the idempotency key cannot be claimed).
Check first: the `dependency` label; `docker compose ps`; `/health`; connection limits on
Postgres (8 workers × pool of 5 + 6 overflow).

#### FraudHighErrorRate
*Critical. More than 1% of requests ended in an unhandled error for 5 minutes.*
1% is far above normal (zero in every load test here) and low enough to catch a failure
affecting one input shape. Client disconnects are counted separately and do not trigger it.
Check first: API logs for the traceback; whether a database migration is missing (an
audit insert on a missing column fails every request); `FraudDependencyDown`.

#### FraudLatencyP99High
*Warning. p99 latency above 250 ms for 10 minutes.*
Normal is about 20 ms at 200 requests per second on a laptop. 250 ms is roughly where a
checkout page starts to feel it. Check first: the "Step Latency Breakdown" panel to see
whether features, the model or the database write is slow; the Postgres-fallback panel
(fallback queries are the usual cause); request rate against the ~400 RPS ceiling in
`docs/performance_2026-11.md`.

#### FraudBackgroundTaskFailures
*Warning. At least one post-response velocity update failed in 10 minutes.*
Decisions are still returned, but the counters the next decision depends on are not being
written. Left alone this becomes "quietly wrong". Check first: logs for
`background_task.failed`; Redis memory and health.

#### FraudRedisDataLoss
*Critical. Redis holds less than half the keys it held within the last hour.*
A restart without persistence, a `FLUSHALL`, or an eviction storm. All velocity history is
gone: windowed counters read zero until traffic refills them, and the all-time device
features are rebuilt lazily from Postgres one device at a time. During that period the
model under-scores, so expect `FraudReviewRateLow` to follow.
Check first: `redis-cli info persistence` and `info memory`; whether someone flushed it.
Then decide whether to lower the review threshold temporarily or accept the gap.

#### FraudRedisFallbackSurge
*Warning. Purchase-history or device-history lookups from Postgres above 20% of requests and
over 3 times their 1-day average, for 10 minutes.*
The softer sign of the same problem, and of a traffic change (a wave of first-seen users).
The threshold is relative because the normal level depends on the traffic: on Fraud_Data
almost every user is new, so the baseline is near 100% and this alert cannot fire; on
traffic with repeat customers it is low. Check first: the Redis keys panel; whether the
surge is all one source.

#### FraudIdempotencyConflicts
*Warning. More than 2% of requests are answered 409 for 10 minutes.*
A client is retrying while its first request is still in flight, or reusing keys across
different purchases. Check first: latency (slow requests cause retry storms); which client.

#### FraudNoTraffic
*Info. No requests for 30 minutes.*
Expected on a development machine. In production it means the caller stopped asking, which
is someone else's outage but this system's blind spot.

### Decisions

#### FraudSignupBurstAttack
*Warning. More than 10% of decisions are `signup_burst` denials for 2 minutes.*
A scripted signup attack. The rule is holding (those purchases are denied); the alert makes
the attack visible, because the next variant may wait 61 seconds. 10% is twice the level
seen when replaying the whole training file, bot month included. This is the alert used by
the drill (`scripts/alert_drill.py`).
Check first: the "Requests by Country" panel and the audit table for the device and IP
pattern; whether `device_many_users_1h` is also firing; whether the shadow rules moved.

#### FraudSharedDeviceRule
*Warning. More than 1% of decisions are `device_many_users_1h` denials for 10 minutes.*
Either a device farm, or a client change that made device ids non-unique (every user behind
one id), in which case real customers are being declined. Check first: how many distinct
devices are involved; whether a client release changed how `device_id` is produced.

#### FraudDenyRateHigh
*Warning. Deny rate above 2% and over 3 times its 1-day average for 15 minutes.*
Declines cost real customers. Relative to its own history because the normal level depends
on the traffic. Check first: "Decisions by rule" to see which rule; then that rule's alert.

#### FraudReviewRateHigh
*Warning. Review share more than twice the validation-time expectation for 30 minutes.*
Each review costs money and friction, and the threshold was chosen for a 5% review rate.
Check first: score drift and feature drift panels; the shared-device share; whether a
client is replaying old traffic.

#### FraudReviewRateLow
*Warning. Review share under half the expectation for 30 minutes.*
The dangerous direction: the model has gone quiet and fraud passes unseen. A feature stuck
at its default (Redis data loss, a renamed request field) does exactly this.
Check first: `FraudRedisDataLoss`; the `did_other_uid_all` drift line; the audit table for
recent values of the model's top features.

### Model

#### FraudScoreDrift
*Warning at PSI above 0.10, critical above 0.25, for 1 hour, with at least 1,000 scored
requests in the hour.*
The score distribution no longer matches the one the thresholds were chosen on, so "review
above 0.76" no longer means "the riskiest 5%". Check first: which inputs drifted (feature
drift panel). Inputs drifted too → find out whether traffic changed (then retrain and
re-pick thresholds) or the feature pipeline broke (then fix the pipeline; the model is
fine). Inputs steady → the model file or its version changed (`FraudModelVersionSkew`).

#### FraudFeatureDrift
*Warning. PSI of a monitored model input above 0.25 for 1 hour.*
The `feature` label names it. Check first: is the feature stuck at one value (pipeline) or
spread differently (traffic)? Compare a few recent audit rows with the reference buckets in
`model_meta.json`.

#### FraudSharedDeviceShareShift
*Warning. The share of orders from devices used by other accounts is more than twice, or less
than half, its validation-time value for 1 hour.*
`did_other_uid_all` is the model's strongest input, and its meaning depends on device ids
being stable. A jump up: device ids collapsing (a client bug) or a real attack reusing
devices. A drop to zero: the all-time device history is not being read.

#### FraudModelVersionSkew
*Warning. More than one model version serving for 10 minutes.*
Normal for a moment during a rolling deploy. Longer means a worker or container did not
restart, and two requests for the same purchase can get different decisions.

### What the first live run showed (2026-11-15)

Two real findings in the first hour, both from a "normal" replay on the local stack:

1. **The score-drift alert went pending at PSI 0.62, and none of the six monitored inputs
   had moved.** The alert was right and could not say why. The cause was two inputs that
   were not monitored: `country_txn_count_30d` (30 on average live, 3,395 at training time)
   and `amount_vs_user_avg_30d` (1.04 live, 37 at training). Both were added to the
   monitored list; the feature-drift panel now shows them at PSI 5.8 and 7.5 next to the
   score. Lesson: monitor every input the model weights, not the ones that seem important.
2. **Why those two differ is itself a finding.** The local database is full of replayed
   traffic, so customers there have purchase history; in Fraud_Data nobody does.
   `amount_vs_user_avg_30d` divides the purchase by the customer's own average, and for a
   first-time customer that average is taken as 1, so on the training data the feature *is*
   the purchase amount. The model learned it as a second amount column, and reads a loyal
   customer's ratio of about 1 as "a one-dollar purchase". This is the same class of flaw
   as `did_txn_cnt_all` (docs/feature_ablation.md §8): right on one-purchase-per-user data,
   wrong on real traffic. It is not fixed yet (it needs a feature change on both the
   training and serving side and a retrain); it is recorded as the next model fix.

On data the model has not been polluted by, the picture is calm: computed offline on the
untouched test months, score PSI is 0.03 in September, October and November. One input does
drift there, `days_since_signup` (PSI 0.36 rising to 5.2), because in this dataset all
signups happen before September, so accounts only get older. The score does not follow.
That is the "input drifts, score steady" case in the runbook, and a reminder that feature
drift is a prompt to look, not a verdict.

## 4. Routing

Alertmanager groups alerts by name, waits 30 seconds to batch related ones, and repeats a
still-firing critical alert hourly and a warning every 4 hours. `FraudApiDown` mutes all
other fraud alerts; a critical score-drift alert mutes its warning. **No notification
channel is configured in the repository**, because a Slack webhook or pager key is a
secret: the three receivers (`critical`, `warning`, `quiet`) are named and empty, so alerts
are grouped and visible in the Alertmanager UI and in Grafana, and sent nowhere. The config
file shows where a Slack receiver goes.

## 5. Testing

- `promtool test rules` runs 13 scenarios against the rules: each important alert fires on
  series that should trigger it and stays quiet on normal ones, the PSI arithmetic is
  checked against a hand-computed value, and the minimum-traffic guard is exercised. It runs
  in CI together with `promtool check config` and `amtool check-config`.
- `tests/test_monitoring.py` checks the Python side: bucket edges, PSI, that each scored
  request lands in exactly one bucket per monitored item, that fallbacks and errors are
  counted, that every metric named in a rule is actually exported, and that every alert has
  a severity, a summary and a section on this page.
- `python scripts/alert_drill.py` proves the whole chain on the running stack: it sends a
  burst of bot signups and watches `FraudSignupBurstAttack` go from pending to firing in
  Prometheus and arrive in Alertmanager, then resolve.

## 6. Things worth looking up

- *The four golden signals* (latency, traffic, errors, saturation), from Google's SRE book:
  the standard checklist for the "it is broken" row.
- *Population Stability Index* and *Kolmogorov–Smirnov test*: two common drift measures.
- *Alert fatigue* and *symptom-based alerting*: why each alert here has a `for` duration, a
  traffic guard and a runbook, and why there are 20 alerts and not 100.
- *Concept drift vs data drift*: the difference between "inputs changed" (seen here) and
  "the relationship between inputs and fraud changed" (needs labels).
