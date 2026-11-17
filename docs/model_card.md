# Model card: fraud risk score for e-commerce purchases

A model card is a short, standard description of a machine-learning model: what it is, what
it was built for, how well it works and for whom, and what it must not be used for
(Mitchell et al., "Model Cards for Model Reporting", 2019). This one describes the model
served by this repository on 2026-11-16.

Every number below comes from a file in the repository, named next to it, and can be
regenerated with the command given in the last section.

## 1. Model details

| | |
|---|---|
| Version | `xgboost-20261116202655-…` (algorithm, training time, git commit; exact value in `model_meta.json`) |
| Type | Gradient-boosted decision trees (XGBoost): 140 trees of depth 3, learning rate 0.05 |
| Output | A risk score between 0 and 1 for one purchase. **Not a probability** (see section 7) |
| Inputs | 30 features, listed in `app/decisions/model_features.json`. No personal attributes: sex and age are not inputs |
| Constraint | The score can never fall as `did_other_uid_all` rises ("more other accounts on this device is never less risky") |
| Decision use | Score ≥ 0.773 → `review`. The deny threshold is "never": the model declines nothing |
| Files | `app/decisions/model.joblib`, `model_features.json`, `model_meta.json` (thresholds, training window, evaluation, monitoring reference) |
| Trained by | `python -m training.train --ship` |

**The inputs, by group**

| Group | Inputs | Share of the model's importance |
|---|---|---|
| Device sharing | `did_other_uid_all` (other accounts ever seen on the device), `did_uid_2h/24h/7d/30d/90d` (distinct accounts on the device in a window) | about 70% (`did_other_uid_all` alone 47%) |
| IP sharing | `ip_uid_24h`, `ip_uid_7d` | small |
| The purchase | `purchase_value`, `hour_of_day`, `day_of_week`, `source`, `browser` | small |
| Where from | 17 yes/no columns for the 10 largest countries, "unknown" and 6 continents | small |

Twelve candidate inputs are deliberately excluded, with the reason recorded in
`model_meta.json` under `dropped_features` and explained in
[feature_audit.md](feature_audit.md): nine counted a customer's own purchases as risk,
measured raw traffic volume, or drift by construction in this dataset; three are personal
attributes (sex and two age features) that were measured to add nothing (section 6).

## 2. Intended use

- **What it is for:** inside this repository's decision engine, after the hard rules have
  run, to pick out purchases that should get a second look: a step-up check (3-D Secure, an
  SMS code) or a manual review.
- **Who uses the output:** the calling checkout system, through the API's `fraud_decision`
  field (`allow` / `review` / `deny`).
- **What it is, honestly:** a demonstration model on a public dataset, built to show an
  end-to-end approach to training, evaluating, serving and monitoring a fraud model.

## 3. Out-of-scope uses

- **Declining customers on the score alone.** At the costs assumed here no score is
  confident enough for a decline to beat a review. All declines come from rules.
- **Any real payment traffic without retraining.** The model has only seen Fraud_Data.
- **Reading the score as a probability of fraud**, or comparing scores across model
  versions. Thresholds are re-chosen with every retrain.
- **Decisions about a person beyond one purchase** (account closure, credit, eligibility).
- **Populations with repeat customers, without checking first.** The features were audited
  for this, but the training data contains no repeat customers to test it on.

## 4. Data

**Source.** `Fraud_Data.csv`, a public e-commerce fraud dataset: 151,112 purchases made in
2015, one per user, with signup time, purchase time, amount, device id, IP address, traffic
source, browser, sex, age, and a fraud label. Not included in the repository.

**Split by time**, never at random, because a fraud model is always used on the future:

| Part | Period | Rows | Used for |
|---|---|---|---|
| Train | Jan 1 to Aug 5 | 98,178 | fitting |
| Validation | Aug 5 to Aug 30 | 15,111 | tree count, thresholds, choice between algorithms |
| Test | Aug 30 to Dec 16 | 30,223 (1,389 frauds, 4.6%) | reporting, ship gates, monitoring reference |

The served model is refit on train + validation (113,289 rows) with the tree count fixed.

**Excluded from training:** 7,600 purchases made within 60 seconds of signup. They are a
bot attack in January (every one is fraud) and are denied by the `signup_burst` rule before
the model runs, so the model never scores such a purchase in production. Trained on them, a
model learns "January means fraud" and nothing that carries forward.

**Known properties of this data that limit what can be concluded:**

- Every user buys exactly once. Anything about a customer's own history cannot be learned or
  tested here.
- After January the fraud rate is flat (about 4.5%) across months, countries, sexes and
  browsers. Almost all the signal is in device sharing.
- The test period has been looked at many times during development. The rolling-origin
  results in section 5 are the less-used evidence.

## 5. Performance

Test period, 30,223 purchases. Source: `app/decisions/model_meta.json` (`evaluation`).

| Metric | Value | For comparison |
|---|---|---|
| PR-AUC | **0.139** [0.123, 0.157] | random guessing 0.046; the model before the feature work 0.092 |
| ROC-AUC | 0.639 [0.621, 0.656] | 0.5 is random |
| Precision in the riskiest 1% of purchases | 31% | fraud rate overall 4.6% |
| Precision / share of fraud found in the riskiest 5% | 23% / 25% | |

PR-AUC is the headline metric because fraud is rare: it asks how pure the flagged group is
as more fraud is caught, where ROC-AUC is flattered by the large number of easy legitimate
purchases. Intervals are 95% bootstrap intervals.

**As a decision policy** (review at score ≥ 0.773; a wrongly declined order costs $10, each
review $5, a reviewed fraud is stopped; `training/policy.py`):

| | Orders reviewed | Fraud reviewed | Legitimate orders reviewed | Total cost |
|---|---|---|---|---|
| Do nothing | 0 | 0 of 1,389 | 0 | $51,395 |
| Model before the feature work, declining at its threshold | 365 declined | 106 | 259 | $49,714 |
| **Served model** | 2,028 (6.7%) | 469 (34%) | 1,559 | **$44,209** |

Two of those three cost figures are assumptions, so here is how the result moves with them
(thresholds re-chosen for each row; test cost):

| Review costs | Review stops | Total cost | Policy chosen |
|---|---|---|---|
| $3 | 100% | $40,153 | review 6.7% |
| $5 | 100% | $44,209 | review 6.7% |
| $5 | 80% | $47,384 | review 6.2%, decline 0.4% |
| $5 | 60% | $49,338 | decline 2.5% |
| $8 | any | about $49,400 to $49,600 | mostly declines, up to 2.9% |

The model earns its keep when a review costs about $5 or less and stops at least 80% of the
fraud it sees. Beyond that it is worth about as much as the old one.

**Over time** (`docs/evaluation/rolling_origin_served_features.json`): trained on the past
and scored on the next month, seven times, mean PR-AUC is 0.121. The three boosting
algorithms tried are within 0.003 of each other (LightGBM +0.002 [−0.001, +0.005] and
CatBoost +0.002 [−0.001, +0.006] against the served XGBoost): a tie.

**Stability** (`docs/monitoring.md`): the score distribution on each unseen month differs
from the reference by a PSI of at most 0.004, where 0.10 is the level worth watching.

## 6. Performance by group, and ethical considerations

Source: `docs/evaluation/model_card_slices.json` (`python -m training.model_card_slices`).
Test period, served model and threshold. "Review rate" is the share of the group's orders
sent to review; "recall" the share of the group's fraud that was reviewed.

| Group | Orders | Fraud rate | Review rate [95% CI] | Precision | Recall [95% CI] |
|---|---|---|---|---|---|
| **All** | 30,223 | 4.6% | 6.7% [6.4, 7.0] | 23% | 34% [31, 36] |
| Sex: F | 12,600 | 4.6% | 6.9% [6.4, 7.3] | 22% | 33% [30, 37] |
| Sex: M | 17,623 | 4.6% | 6.6% [6.2, 7.0] | 24% | 34% [31, 37] |
| Age under 25 | 5,226 | 4.7% | 6.2% [5.5, 6.8] | 22% | 29% [23, 35] |
| Age 25 to 34 | 12,409 | 4.6% | 6.8% [6.4, 7.3] | 23% | 35% [31, 39] |
| Age 35 to 44 | 9,451 | 4.7% | 6.7% [6.2, 7.3] | 25% | 35% [31, 39] |
| Age 45 and over | 3,137 | 4.1% | 7.0% [6.1, 8.0] | 19% | 32% [24, 41] |
| Source: Ads | 12,046 | 4.5% | 6.6% [6.2, 7.0] | 19% | 28% [24, 32] |
| Source: Direct | 6,061 | 5.2% | 6.8% [6.2, 7.4] | 35% | 46% [41, 51] |
| Source: SEO | 12,116 | 4.4% | 6.8% [6.4, 7.3] | 21% | 33% [29, 37] |
| Browser (5 groups) | | 4.2% to 4.6% | 6.6% to 7.3% | 22% to 28% | 32% to 48% |
| United States | 11,491 | 4.9% | 7.1% [6.6, 7.6] | 23% | 33% [30, 37] |
| IP not in the country table | 4,420 | 4.0% | 6.1% [5.4, 6.7] | 21% | 31% [24, 38] |
| China | 2,492 | 4.9% | 6.5% [5.5, 7.5] | 25% | 34% [24, 42] |
| South America (excl. Brazil) | 472 | 6.4% | 11.2% [8.9, 14.0] | 30% | 53% [37, 71] |
| Other country groups (13) | 318 to 2,552 each | 3.1% to 5.4% | 5.3% to 8.6% | 9% to 35% | 13% to 45% |

How to read it:

- **Sex and age band:** review rates overlap within their intervals (women 6.9%, men 6.6%;
  age bands 6.2% to 7.0%), and so does recall. No group is reviewed measurably more than
  another. The model no longer sees either attribute (below), so any remaining difference
  comes through other inputs.
- **Traffic source:** review rates are equal, but the model finds far more of the fraud
  among Direct visitors (46%) than among Ads visitors (28%), and its reviews there are
  nearly twice as precise. Fraud arriving through ads is harder for this model to see.
- **Country:** one group stands apart: South America outside Brazil is reviewed at 11.2%
  against 6.7% overall. Its fraud rate is also the highest (6.4%), and the reviews there
  are more precise than average (30%), so the extra reviews are finding real fraud; but
  the group is small (472 orders) and a customer there is 1.7 times as likely to be asked
  for an extra check. The small country groups have intervals too wide to say more.

**Sex and age are not inputs.** Until 2026-11-16 the model used `sex`, `age_group` and
`age_account_gap`. In many jurisdictions and for many uses, deciding by sex or age is
restricted or needs justification, and a step-up check is friction imposed on a customer.
The data offered no justification: with the three inputs put back, test PR-AUC is 0.139
against 0.139 without them (paired difference +0.000, interval [−0.003, +0.004]) and policy
cost $44,438 against $44,209. They bought nothing measurable, so they were removed.
`tests/test_served_model.py` checks that changing a customer's sex and age cannot change
their score. Age is still a request field, used only by the rule that rejects impossible
ages.

**Other considerations.**

- Device and IP sharing is the model's main evidence. People who legitimately share devices
  (families, internet cafés, public libraries, corporate networks) will be reviewed more
  often. The data has no way to measure how many of the reviewed were such customers.
- A review is not an accusation, but repeated extra checks are a real cost to a customer.
  The engine has no memory of having cleared someone before.
- Country is derived from the IP address. 15% of test purchases come from addresses not in
  the lookup table.

## 7. Limitations

- **The data.** Public, one purchase per user, with a fraud rate that barely varies by
  anything except device sharing. Results say how this approach behaves on this dataset.
- **Scores are not probabilities.** Training up-weights fraud about 21 times, which inflates
  scores (the median is about 0.46 while the fraud rate is 4.6%). Only the ordering and the
  thresholds chosen on validation mean anything.
- **The cost model is assumed.** $10 per wrongly declined order, $5 per review, and a
  review that always stops the fraud are inputs, not measurements (table in section 5).
- **Live accuracy is not measured.** Live traffic has no labels. Action rates and drift are
  monitored instead; accuracy is re-measured on labelled history at each retrain.
- **Coverage.** The served policy reviews 34% of the fraud; 66% passes. This is a weak
  signal used carefully, not a strong detector.
- **Repeat customers.** Inputs were audited so that a customer's own history cannot raise
  their score, and this is unit-tested, but the training data has no repeat customers.
- **Adversaries adapt.** A fraudster who uses a fresh device per account is invisible to
  the model's main input.

## 8. Monitoring, maintenance, reproduction

- **Monitoring:** [monitoring.md](monitoring.md). The four golden signals, dependencies,
  action rates (allow / review / deny against the last day and the expected rate) and drift
  of the score and eight inputs, with 29 alerts and a runbook.
- **Ship gates** (`training/train.py --ship`): test PR-AUC not below the served model's
  interval; policy cost not above the served policy's; the saved file reloaded the way the
  API loads it and required to give identical scores.
- **When to retrain:** a score-drift or feature-drift alert that is not explained by the
  feature pipeline; a sustained change in action rates; any change to the features.
- **Reproduce every number here:**

```bash
pip install --group train
python -m training.train                 # model, thresholds, test metrics, cost, sensitivity
python -m training.rolling_eval          # seven-fold results over time
python -m training.model_card_slices     # the by-group table and the sex/age ablation
```

Related pages: [feature_ablation.md](feature_ablation.md) (how the main feature was found
and priced), [feature_audit.md](feature_audit.md) (why nine inputs were removed),
[model_candidates.md](model_candidates.md) (the four algorithms compared),
[production_readiness.md](production_readiness.md) (what is and is not done).
