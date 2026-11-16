# Feature audit: does each model input mean the same thing when customers come back?

Date: 2026-11-15. Result: the served model went from 42 inputs to 33 with no loss of
accuracy, and none of the remaining inputs depends on a property that only this dataset has.

## Why an audit was needed

Fraud_Data has 151,112 purchases by 151,112 users: **nobody buys twice.** A feature can
therefore be exactly right on this data and mean something different on real traffic, and
no offline metric will show it, because the case that breaks it does not exist in the data.
This happened three times in one week before the audit:

| Found by | Feature | What was wrong |
|---|---|---|
| A hand-made request on the live stack | `did_txn_cnt_all` | Counted every earlier purchase on a device, so a loyal customer's second purchase on their own device went to review |
| The score-drift alert, first live run | `amount_vs_user_avg_30d` | Equal to the purchase amount for anyone without history; a repeat customer's ratio of about 1 read as "a one-dollar purchase" |
| A PSI check on the unseen test months | `days_since_signup` | All signups are before September, so account age only grows; the model extrapolated and its score distribution drifted |

Three finds by three different methods said the rest should be checked on purpose instead of
waiting for the next accident.

## The question asked of every input

*Say in words what this feature means for a customer who comes back, and for a month the
model has not seen. Is that still the thing the model learned?*

## Result for the 42 inputs

| Inputs | Verdict | Evidence | Action |
|---|---|---|---|
| `amount_vs_user_avg_30d` | **Wrong.** With no history the average was taken as 1 | Identical to `purchase_value` on all 151,112 rows | Definition corrected on both sides: 1.0 ("their usual") without history. Now constant on this data, so it leaves the model by the constant-feature rule; correct on data with repeat users |
| `did_txn_cnt_2h`, `_24h`, `_7d`, `_30d` | **Wrong, same flaw as `did_txn_cnt_all`.** Count every earlier purchase on the device, the buyer's own included | Each equals `did_uid_<window> − 1` on every row, so also pure duplicates here | Removed from the model's inputs. Still computed, audited and available to rules |
| `country_txn_count_24h`, `_7d`, `_30d` | **Unstable.** A raw volume counter: its scale is the request rate | 30-day mean 1,719 in March, 3,433 in June, 1,331 in November; 30 on a short local replay (PSI 5.8) | Removed from the model's inputs |
| `days_since_signup` | **Drifts by construction in this dataset** (sound in meaning) | Mean 37 days in March, 103 in November; PSI against the reference 0.36, 2.3, 5.2 over the test months; with it the score PSI was 0.32, 0.39, 0.64 | Removed from the model's inputs, for the same reason `month_of_year` already was |
| `did_other_uid_all`, `did_uid_2h/24h/7d/30d/90d`, `ip_uid_24h/7d` | OK. Distinct accounts, so a returning buyer does not move them | Unit tests with repeat customers on both the training and serving side | Kept. `did_other_uid_all` is constrained so more accounts never lowers the score |
| `purchase_value`, `hour_of_day`, `day_of_week`, `source`, `browser`, 17 country columns | OK. Request fields and their encodings | PSI at most 0.022 on any unseen month | Kept |
| `sex`, `age_group`, `age_account_gap` | OK in meaning. `age_account_gap` contains account age but is dominated by the customer's age | PSI 0.008 at worst | Kept. Using sex and age is a fairness question, noted in `docs/production_readiness.md` |

**What it cost.** Tuned XGBoost on the seven rolling-origin folds, paired bootstrap against
the previous feature set: removing the eight flawed inputs changed mean PR-AUC by +0.002
[−0.001, +0.005]; removing `days_since_signup` as well changed test PR-AUC by −0.001
[−0.005, +0.004]. On the test period the served model scores 0.140 [0.124, 0.158] (0.136
before the audit) and its policy costs $44,305 ($44,554 before). Nine inputs carried no
information the others did not.

**What it bought.**

- A customer's own history cannot raise their score: `tests/test_served_model.py` changes
  only a repeat customer's own purchase counts and requires the identical score.
- The score distribution holds still on unseen data: PSI against the reference is 0.004 on
  the validation month and at most 0.002 on September, October and November. Before, the
  score-drift alert would have fired on ordinary traffic.
- Drift alerts are now worth reading: when they fire, something actually changed.

## Dormant candidates

28 candidate features are constant on Fraud_Data and so are not in the model, but would
enter automatically on data with repeat users. They were reviewed with the same question:

| Candidates | Verdict |
|---|---|
| `time_since_last_txn_uid` | **Fixed.** Used 0 for both "no earlier purchase" and "again this second"; now −1 without history |
| `purchase_success_cnt_*`, `purchase_success_amt_*` | **Caution.** Built from fraud labels in training and from the engine's own decisions in serving; these are different things. The Sparkov study reports them separately as an optimistic upper bound for that reason. Do not let them into a served model as they are |
| `purchase_cnt_*`, `purchase_amt_*`, `pur_median_amt_90d`, `high_amount_vs_hist`, `uid_did_*`, `did_uid_days_on_file`, `ip_uid_2h` | Sound: about the same user's own past, or distinct counts that include the buyer by design |

The Sparkov comparison (`training/compare_datasets.py`), which does have repeat users, was
re-run with the corrected definitions: unchanged (PR-AUC 0.878 static, 0.974 with velocity
features).

## The rule from now on

A new model input needs three things before it is added to `CANDIDATE_FEATURES`:

1. One sentence saying what it means for a returning customer and in a month outside the
   training window.
2. A test with a repeat customer (the synthetic parity data has them; Fraud_Data does not).
3. A place on the monitored list in `app/monitoring.py`, so a change in its distribution is
   seen.

Features that fail the first question go into `DROP_ALWAYS` in `training/train.py` with the
reason, which is copied into `model_meta.json` under `dropped_features`.

## Things worth looking up

- *Dataset shift*, *covariate shift* and *out-of-distribution inputs*.
- *Feature leakage* (the cousin of this problem: a feature that knows too much, where these
  knew something different).
- *Extrapolation in tree models*: a tree cannot extend a trend beyond the values it saw; it
  reuses the prediction of its last leaf.
