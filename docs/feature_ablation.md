# Feature study: all-time device history (`did_other_uid_all`, first built as `did_txn_cnt_all`)

Date: 2026-11-15. Status: **shipped** with a `review` decision band. The model served today
(`xgboost-20261115222918-a9903ee`) keeps this feature and the monotone constraint of
section 8, and has 33 inputs after the audit in [feature_audit.md](feature_audit.md); its
numbers are in the README. The sections below record each step as it was measured at the
time, so their feature counts and figures are those of that step. Read in order: sections 1 to 6 are the study as first written (the feature is
called `did_txn_cnt_all` there and the conclusion is "do not ship behind allow/deny");
section 7 adds the review band that made it worth shipping; section 8 is the flaw found by
live testing the day it shipped, and the corrected feature that is served now.

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
(`docs/evaluation/rolling_origin_served_features.json`, `docs/evaluation/train_did_txn_cnt_all.json`):

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

- With an average fraud of $37 and a $10 cost per wrongly declined order, declining breaks
  even at **21% precision** ($10 / ($10 + $37): at precision π a decline saves π × $37 and
  loses (1 − π) × $10). The new feature creates a large group at about 23%, only just above
  that line: each decline there earns about $0.90 on average, against about $3.60 for the
  old model's fewer declines at 29%. Five times as many declines, each worth a quarter as
  much, add up to the same total. So at the current settings the new model **declines five
  times more orders for the same total cost**. (An earlier version of this page gave the
  break-even as 27%, $10 / $37; that formula was wrong. Every cost in the tables was
  computed directly from the data and is unaffected.) The same holds under the unit-cost convention used for rule backtests (missed
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
`docs/production_readiness.md`), and keep `deny` for scores where a decline clearly beats a review. Until
then the served model stays `xgboost-20261109211048-5ba3793`.

## 7. Follow-up: the `review` band (2026-11-15, same day)

The API and the training script now support a third decision. `training/train.py` picks a
review threshold and a deny threshold together on validation by cost (`pick_thresholds`),
and the ship gate also requires the new policy to cost no more on test than the served one.
Report of this first run: `docs/evaluation/train_review_band.json`; the shipped run, after
the tie-break below was added: `docs/evaluation/train_shipped_20261115.json`. Cost model: a wrongly declined order
costs $10; every reviewed order costs $5; a reviewed fraud is stopped. This is stricter
than the table in section 6, which charged the review only on legitimate orders.

| Policy on the test period | Reviewed | Denied | Fraud reviewed / denied / missed | Legit reviewed / declined | Total cost |
|---|---|---|---|---|---|
| Do nothing | | | 0 / 0 / 1,389 | | $51,395 |
| Served model as served (deny ≥ 0.843) | 0% | 1.2% | 0 / 106 / 1,283 | 0 / 259 | $49,714 |
| Served model, its own best review/deny pair | 2.3% | 0.2% | 157 / 18 / 1,214 | 522 / 44 | $48,558 |
| **New model (review ≥ 0.784, deny ≥ 0.904)** | 6.6% | 0.02% | 458 / 3 / 928 | 1,524 / 3 | **$44,403** |

So with a review band the new model cuts total cost by 10.7% against what is served and by
13.6% against doing nothing, and it almost stops declining on the model score at all: at
these costs a decline beats a review only above 50% precision, which the model reaches for
six orders. Declines are left to the hard rules.

**How much this depends on the two assumptions about review** (thresholds re-picked on
validation for each row; test cost of the new model vs the served model given the same
freedom):

| Review cost | Review stops | New model | Served model's best | New model's policy |
|---|---|---|---|---|
| $3 | 100% | $40,439 | $49,254 | review 6.6% |
| $3 | 80% | $44,355 | $49,380 | review 5.5%, deny 1.1% |
| $3 | 60% | $47,239 | $49,330 | review 4.3%, deny 2.3% |
| $5 | 100% | $44,403 | $48,558 | review 6.6% |
| $5 | 80% | $48,042 | $49,839 | review 3.9%, deny 2.3% |
| $5 | 60% | $49,743 | $49,888 | deny 6.2%, no review |
| $8 | any | $49,743 | $49,888 | deny 6.2%, no review |

The gain is real when a review costs about $5 or less and stops at least 80% of the fraud
it sees. Outside that range the review band disappears and the result is the cost-neutral
"deny 6%" policy of section 6. Those two numbers are assumptions here, not measurements:
they are what a real deployment would have to measure first, and they are command-line
parameters (`--review-cost`, `--review-catch-rate`) for that reason.

One weakness in that first run: when costs are nearly tied (the last rows), the search
returned the policy that acts on *more* orders. **Fixed before shipping:** among policies
within 1% of the lowest validation cost, `pick_thresholds` now takes the one that touches
the fewest orders (`training/policy.py`). Shipped policy: review ≥ 0.801, no model declines;
on test 6.2% reviewed (435 of 1,389 frauds, 23% precision, 1,438 legitimate orders), total
cost **$44,728** against $49,714 before, a difference of −$4,986 with a 95% paired interval
of [−$6,348, −$3,778]. With the tie-break the bad-assumption rows also behave: at a $5
review that stops only 60%, or an $8 review, the policy falls back to declining about 2% of
orders (about $49.2k) instead of 6%.

## 8. The flaw live testing found, and the feature that is served (2026-11-15)

The first shipped version used `did_txn_cnt_all`: every earlier transaction on the device.
After the ship I ran the demo and a load test against the rebuilt stack and added one more
request by hand: the same account buying three times on its own device, 30 days apart.

| Purchase | `did_txn_cnt_all` | Score | Decision |
|---|---|---|---|
| 1st | 0 | 0.375 | allow |
| 2nd | 1 | 0.806 | **review** |
| 3rd | 2 | 0.805 | **review** |

Every returning customer would be sent to review. The offline evaluation could not see this:
in Fraud_Data each user buys exactly once, so "this device has history" and "another account
used this device" are the same rows. The model learned the second; the feature measured the
first.

**Fix.** The served feature is now `did_other_uid_all`: the number of *other* accounts ever
seen on the device, the buyer excluded.

- On Fraud_Data it is the same column, value for value (checked on all 151,112 rows), so the
  ablation, the retrained model, the thresholds and every number in sections 3 to 7 are
  unchanged: test PR-AUC 0.141, policy cost $44,728, both ship gates pass.
- On real traffic it differs exactly where it should: the loyal customer above now has
  `did_txn_cnt_all` = 2 and `did_other_uid_all` = 0, scores 0.35 and is allowed
  (`scripts/demo.py`, story 6).
- Serving: a Redis set `did:uids_all:{device_id}` (`SADD` after each decision; `SCARD` and
  `SISMEMBER` in the read pipeline, 41 commands, still one round trip). The Postgres fallback
  became one query that answers both features (`user_id, COUNT(*) ... GROUP BY user_id`) and
  re-seeds both Redis keys. `db_init/009` adds the audit column. `did_txn_cnt_all` is still
  computed and stored (cheap, and useful to monitor) but is no longer a model input.
- Tests: the synthetic parity data has repeat users, so it tells the two features apart on
  both paths; `test_loyal_customer_on_own_device_has_no_other_accounts` pins the fix.

**A second weakness, measured and left open.** The model's score is not monotonic in the
feature: 0 other accounts → 0.47, 1 → 0.81, 3 → 0.77, 5 or more → 0.52. In the training
period, devices with many earlier accounts were the January bot devices, whose later
purchases were mostly legitimate (759 rows, 4.9% fraud), so the model learned that a very
shared device is ordinary. A device farm adding accounts slowly (so the 1-hour hard rule
does not fire) would be scored as low risk. XGBoost can be told that more accounts must
never lower the score (`monotone_constraints`); measured:

| | Rolling mean PR-AUC | Test PR-AUC | Test policy cost | Score at 5+ other accounts |
|---|---|---|---|---|
| Unconstrained (served) | 0.124 | 0.141 | $44,728 | 0.52 (allowed) |
| Monotone in `did_other_uid_all` | 0.119 | 0.136 | $44,554 | 0.82 (review) |

Paired on test: PR-AUC −0.006 [−0.010, −0.001] for the constraint, cost difference
[−$605, +$200] (no difference). Only 9 test rows have four or more other accounts, and 3 of
them are fraud. So the constraint costs a little ranking quality on this data, costs no
money, and closes a hole the data barely tests. It was first left out because the standing
rule for an automatic re-ship was "numbers within noise of the approved model"; Xiaowan
then approved it and it **is shipped**: `monotone_constraints` on `did_other_uid_all`,
test PR-AUC 0.136 [0.121, 0.153], review ≥ 0.762 (6.4% of orders, 448 frauds, 1,495
legitimate), policy cost **$44,554**. A device with six other accounts spread over months
now scores 0.83 and is reviewed (`scripts/demo.py`, story 7), and
`tests/test_served_model.py` checks that the served score never falls as the count rises.
LightGBM and CatBoost stay unconstrained comparison candidates; over the rolling folds they
are within noise of the constrained XGBoost (−0.002 [−0.005, +0.002] and
−0.001 [−0.004, +0.003]).

**A model that trained but could not be served.** The first constrained model was shipped
with the constraint as a `{feature: direction}` dictionary. It trained, scored and passed
both gates. The API then failed to score with it: on load it sets `n_jobs=1`, and XGBoost
rejects the dictionary form when parameters are re-applied to a loaded model. The local
stack had not been rebuilt, so nothing live broke, but the file in the repository was
unusable and all 192 tests passed, because every test replaced the model with a mock. Fixes:
the constraint is converted to one number per column before fitting; `--ship` now saves to
a temporary file, reloads it the way the API does, and requires identical scores before
replacing the served model; and four tests score through the real `model.joblib`.

**What this section is worth as a lesson.** The offline numbers were right and the feature
was still wrong, because the dataset could not express the case that broke it. It was found
by running real requests through the live stack, not by any metric. Two habits follow:
write the feature's meaning in words that would still be true on other data ("other
accounts", not "transactions"), and keep a demo script with named stories that is run after
every ship.

## 9. Things worth looking up

- *Ablation study*: measuring a component by removing it.
- *Paired bootstrap*: why comparing two models on the same resampled rows detects much
  smaller differences than comparing their separate confidence intervals.
- *Break-even precision*: cost of a false positive divided by the value saved by a true
  positive; the threshold below which an action loses money.
- *Step-up authentication / 3-D Secure*: the standard "third option" between allow and deny.
- *Monotonic constraints* in gradient boosting: forcing the model's answer to move in one
  direction with a feature, used when you know the direction and the data is thin.
- *Dataset shift / out-of-distribution inputs*: why a model can be right on its test set and
  wrong on the first real request.
