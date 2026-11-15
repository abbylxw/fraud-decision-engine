# Feature study: all-time device history (`did_txn_cnt_all`)

Date: 2026-11-15. Status: **feature built on both the training and serving side, tested,
retrained candidate ready; not shipped.** The ship decision is open and explained in
section 6.

Short version: four tuned algorithms tied at PR-AUC 0.09, so the algorithm was not the
limit. One new feature, "how many transactions has this device ever made before this one",
moves the tuned XGBoost from 0.092 to 0.141 on the test period. It finds five times more
fraud, but at a precision (about 23%) that is too low to pay for an outright decline. It is
worth money only if the action is cheaper than a decline, for example a review or a
step-up check.

## 1. Where the idea came from

Every device counter in the feature set looked back over a window of at most 90 days
(`did_uid_90d`, `did_txn_cnt_30d`, ...). The rolling-origin evaluation showed all models
reading the same two or three device-sharing counters. So the question was whether a device
that returns after *more* than 90 days still carries signal. It does:

| Scored rows May to Dec, bot rows removed | Share of traffic | Fraud rate |
|---|---|---|
| Device never seen before | 95.1% | 3.6% |
| Device seen at least once before, at any time | 4.9% | **22.9%** |

In this dataset every user buys once, so "device seen before" always means "a different
user used this device earlier".

## 2. Method

`python -m training.feature_ablation`. An *ablation* fits the same model with and without a
group of features and compares. Here:

- Model: the tuned XGBoost from `train.make_candidates`.
- Data: the seven rolling-origin folds of `training/rolling_eval.py` (train on the past,
  pick the tree count on the next month, score the month after).
- Comparison: paired bootstrap of the PR-AUC difference from the baseline feature set,
  averaged over the seven folds. A gain is claimed only if the 95% interval excludes zero.

Candidate features, all computed in the event-time replay (`training/build_features.py`)
from transactions strictly before the current one:

| Feature | Meaning |
|---|---|
| `did_txn_cnt_all` | transactions ever seen on this device |
| `did_secs_since_first_txn` | seconds since the device's first transaction (−1 = new device) |
| `did_secs_since_last_txn` | seconds since the device's previous transaction (−1 = new) |
| `did_uid_all` | distinct users ever seen on the device, including this one |

## 3. Result

Results file: `docs/evaluation/feature_ablation.json`.

| Feature set | Mean PR-AUC (7 folds) | Difference vs baseline [95% CI] |
|---|---|---|
| Baseline (41 features) | 0.090 | |
| + `did_txn_cnt_all` | 0.123 | **+0.033 [+0.028, +0.038]** |
| + `did_secs_since_first_txn` | 0.121 | +0.031 [+0.026, +0.037] |
| + `did_secs_since_last_txn` | 0.120 | +0.030 [+0.025, +0.036] |
| + `did_uid_all` | 0.123 | +0.033 [+0.028, +0.038] |
| + count and both recency features | 0.124 | +0.034 [+0.027, +0.041] |
| + all four | 0.124 | +0.034 [+0.027, +0.042] |

Per scored period, baseline → with `did_txn_cnt_all`: May 0.068 → 0.071, Jun 0.096 → 0.107,
Jul 0.085 → 0.102, Aug 0.099 → 0.135, Sep 0.099 → 0.144, Oct 0.087 → 0.137,
Nov + Dec 0.093 → 0.162.

How to read it:

- **Any one of the four gives the whole gain.** They are four views of one fact, "this device
  has history". Adding the others on top adds 0.001, which is noise. `did_uid_all` is
  literally `did_txn_cnt_all + 1` on this data (one purchase per user).
- **The gain grows through the year** because history accumulates: in May few devices have
  had time to come back; by November many have. A windowed counter throws that history away
  after 90 days.
- **Decision: serve one feature, `did_txn_cnt_all`.** One counter is the cheapest thing to
  serve and to keep in parity, and the data cannot justify more. The other three stay in the
  offline replay so the ablation remains reproducible; they are not in the model's candidate
  list.

Things that were tried in the same probe and did **not** help (mean PR-AUC within ±0.002 of
baseline): the raw signup-to-purchase gap in seconds, signup hour and weekday, raw age,
whether the device or IP had appeared in the January bot attack, and the count of confirmed
frauds on the device or IP with a 30-day label delay (zero rows qualify: fraudulent devices
in this data do not come back after their fraud is known). These null results matter as much
as the positive one: the improvement is one specific signal, not "more features".

**Leakage check.** A feature that looks at later rows would make this result worthless.
`tests/test_feature_parity.py::test_all_time_device_count_uses_only_earlier_rows` builds
features on a frame and on its first half and requires identical values, and checks the
feature equals the count of earlier rows on the same device.

## 4. All four algorithms with the new feature

Re-tuned with the feature (`docs/evaluation/tune_sweep.json`) and re-run
(`docs/evaluation/rolling_origin_42_features.json`, `docs/evaluation/train_did_txn_cnt_all.json`):

| Model | Validation PR-AUC | Test PR-AUC [95% CI] | Test ROC-AUC | Rolling mean (7 folds) |
|---|---|---|---|---|
| Random forest | 0.129 | 0.131 [0.117, 0.146] | 0.643 | 0.115 |
| XGBoost (depth 3, lr 0.05, min_child_weight 100, 200 trees) | 0.149 | 0.141 [0.126, 0.160] | 0.643 | 0.124 |
| LightGBM (7 leaves, lr 0.03, min_child_samples 100) | 0.148 | 0.141 [0.127, 0.158] | 0.646 | 0.119 |
| CatBoost (depth 3, lr 0.05, l2 10) | 0.143 | 0.137 [0.123, 0.154] | 0.645 | 0.120 |
| Served XGBoost (41 features) | | 0.092 [0.081, 0.107] | 0.557 | |

Every algorithm moves up by 0.04 to 0.05 together. That is the cleanest evidence for the
claim in `docs/model_candidates.md`: on this data the features decide the score, the
algorithm does not. XGBoost stays the selected model (highest validation PR-AUC; over the
rolling folds it is ahead of LightGBM by 0.005 [0.001, 0.008] and of CatBoost by 0.003
[−0.001, 0.007]).

Retrained XGBoost against the served model on the same test rows, paired bootstrap:
**+0.049 PR-AUC [+0.040, +0.059]**. At a 5% alert rate, precision goes from 13.8% to 24.0%
and recall from 15.0% to 26.1%.

Honesty note: the test period has now been read many times across this project. The primary
evidence for this feature is the rolling-origin paired interval in section 3, which was the
pre-stated go/no-go rule; the test numbers are the confirmation, not the discovery.

## 5. Serving design

Training/serving parity is a hard rule of this project, so the feature was built on both
sides before any retrain is shipped.

- **Write:** `update_velocity_counters` does `INCR did:txn_all:{device_id}` after each
  decision. No TTL and no 90-day purge, unlike every other counter.
- **Read:** one extra `GET` in the existing Redis pipeline in `build_features` (39 commands,
  still one round trip).
- **Fallback:** if the key is missing (a new device, or Redis lost it), `COUNT(*)` on the audit
  table by `device_id` up to the transaction's own time, behind the new index
  `idx_transactions_device_time` (`db_init/008`). The answer is written back with `SET NX`, so
  the counter continues from the right number and the next request skips Postgres.
- **Audit:** the value is stored per transaction in the new `did_txn_cnt_all` column.
- **Tests:** the parity replay (synthetic and 5,000 real rows) compares the feature between
  training and serving; five fallback tests cover no-window counting, Redis hit, Redis miss,
  re-seeding from Postgres, and the arrival-order behaviour below.

Trade-offs accepted:

| Trade-off | Why accepted | What would change it |
|---|---|---|
| The counter is in **arrival order, not event time**. An out-of-order replay counts a transaction that has a later `purchase_time`. Every windowed counter is event-time correct; this one is not. | An event-time-correct version needs the device's full timestamp history in Redis (one sorted-set member per transaction, kept forever). Live traffic arrives in order, and the Postgres fallback *is* event-time correct. Pinned by a test so it cannot change silently. | Backfills or late-arriving events becoming normal. |
| **Redis memory grows with the number of devices** and never shrinks (one small integer per device; 137k devices in Fraud_Data). | The value of the feature is exactly that it does not forget. | Hundreds of millions of devices: move to a TTL of a year or two, or to Postgres only. |
| **One extra indexed Postgres query for every first-seen device**, which is most requests on this dataset (95% of rows are a new device). | Same pattern as the existing purchase-history fallback; correctness after a Redis loss matters more than the query. Latency impact has **not** been load-tested yet. | A measured p99 regression: cache "known new" in a Bloom filter or accept 0 on miss and backfill offline. |
| Counts **every** transaction on the device, including denied ones. | Same as training, and a denied attempt is still evidence about the device. | Label feedback arriving: split into "all" and "confirmed good". |

The served model does not use the feature yet (`model_features.json` is unchanged), so this
code is live-safe: the API computes and stores the value, the model ignores it.

## 6. What it is worth, and the open ship decision

Better ranking is not automatically more money. Thresholds below were picked on validation
by minimum cost, then applied to the test period (30,223 transactions, 1,389 frauds, average
fraud $37, cost of doing nothing $51,395).

| Cost of acting on a legitimate order | Model | Acted on | Fraud caught | Wrongly acted on | Precision | Total cost |
|---|---|---|---|---|---|---|
| $10 (a declined good customer) | Served | 0.9% | 85 | 191 | 31% | $49,888 |
| $10 | **New** | 6.2% | 435 | 1,438 | 23% | $49,743 |
| $5 (a review or step-up check) | Served | 4.5% | 204 | 1,155 | 15% | $49,448 |
| $5 | **New** | 6.6% | 461 | 1,527 | 23% | **$42,098** |
| $3 | Served | 11.2% | 290 | 3,110 | 9% | $49,764 |
| $3 | **New** | 6.8% | 467 | 1,583 | 23% | **$38,966** |

(The served model at its live threshold of 0.843 declines 1.2%, catches 106 and wrongly
declines 259, for $49,714: the same cost again.)

Precision by score band on test (new model vs served): top 1% of traffic 29.5% vs 29.1%;
next 1% 26.8% vs 17.5%; next 3% 21.3% vs 7.4%; next 5% 8.8% vs 4.5%.

What this says:

- With an average fraud of $37 and a $10 cost per wrongly declined order, declining pays only
  above **27% precision** ($10 / $37). The new feature creates a large group at about 23%.
  So at the current settings the new model **declines five times more orders for the same
  total cost**. The same holds under the unit-cost convention used for rule backtests (missed
  fraud 3, false positive 1; break-even precision 25%): new 4,300, served 4,103, doing
  nothing 4,167.
- If the action costs $5 or less per legitimate order, the new model cuts total cost by 15 to
  22% where the served model saves almost nothing. A 23%-precision group is a textbook
  **review or step-up queue** (ask for 3-D Secure, an SMS code, or a manual look), not an
  auto-decline.
- The validation-chosen decline threshold lands in a flat part of the cost curve, so a small
  change in cost assumptions moves the decline rate from about 1% to about 6%. That
  instability is a reason not to ship a decline threshold from this run as it is.

**Recommendation: do not ship this model behind the current allow/deny decision.** Ship it
together with a third decision, `review`, for the 23%-precision band (roadmap item in
`docs/production_readiness.md`), and keep `deny` for scores above the 27% break-even. Until
then the served model stays `xgboost-20261109211048-5ba3793`.

## 7. Things worth looking up

- *Ablation study*: measuring a component by removing it.
- *Paired bootstrap*: why comparing two models on the same resampled rows detects much
  smaller differences than comparing their separate confidence intervals.
- *Break-even precision*: cost of a false positive divided by the value saved by a true
  positive; the threshold below which an action loses money.
- *Step-up authentication / 3-D Secure*: the standard "third option" between allow and deny.
