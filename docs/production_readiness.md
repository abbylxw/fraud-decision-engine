# Production readiness

What "ready" means here, where the project stands, and what is deliberately out of scope.
This is a portfolio project on public data, so the standard is: the model lifecycle is
closed (train, evaluate, explain, monitor, get labels, retrain), failure modes are tested,
and every gap is written down rather than hidden. Updated at the end of each phase of work;
last update 2026-11-15.

Status key: **Done** = implemented, tested and documented. **Partial** = works, with a named
gap. **Open** = not started.

## 1. Checklist

| Area | Status | Evidence | Gap / next step |
|---|---|---|---|
| **Evaluation method** | Done | Time split, bootstrap CIs, cost-based threshold (`training/evaluate.py`); rolling-origin folds with paired bootstrap (`training/rolling_eval.py`) | Test period has been read many times; rolling-origin is now the primary evidence for comparisons |
| **Model selection** | Done | Four tuned candidates, selection rule in code and in `model_meta.json` (`docs/model_candidates.md`) | Grid search, not a budgeted search; fine at this signal level |
| **Early stopping correctness** | Done | XGBoost/CatBoost tree counts picked with scikit-learn's metric (`training/candidates.py`), after the built-in metrics were found to stop at tree 1 | `compare_datasets.py` (Sparkov) still on XGBoost's own stopping |
| **Feature engineering evidence** | Done | Ablation with paired CIs and recorded null results (`docs/feature_ablation.md`) | |
| **Training/serving parity** | Done | Shared encodings; event-time windows; parity replay on synthetic and real rows; contract tests | `did_txn_cnt_all` is arrival-order, not event-time (documented, pinned by a test) |
| **Best available model in production** | **Open** | Candidate with `did_txn_cnt_all` and a review band: test PR-AUC 0.141 vs served 0.092, test cost $44.4k vs $49.7k, both ship gates pass (`docs/feature_ablation.md` §7) | Waiting for explicit ship approval; then apply `db_init/008`, rebuild the image, smoke-test |
| **Decision policy** | Partial | Hard rules; allow / review / deny on the model score with both thresholds chosen by cost on validation; shadow rules; rule backtest | Review cost and catch rate are assumptions, not measurements; cost ties resolve towards the more intrusive policy; velocity cut-offs in `rate_limiter_control` are expert-set and cannot fire on this data |
| **Reproducible training** | Done | `python -m training.train`, fixed seed, pinned dependency groups, model version = algorithm + timestamp + git sha | No scheduled retraining; artifacts live in git, not a registry |
| **Ship gate** | Done | `--ship` requires test PR-AUC ≥ the previous model's CI lower bound **and** test cost of the new policy ≤ the served policy; run only on explicit approval | Cost comparison is a point estimate, no interval |
| **Auditability** | Done | Every decision stores features, `model_version`, `model_score`, `decision_rule` | No per-decision reason codes |
| **Explainability** | Open | Global feature importances in `model_meta.json` | Top-3 SHAP reason codes per scored decision |
| **Monitoring: system** | Done | Prometheus metrics, Grafana dashboards, per-step latency, per-worker counters | No alert rules committed |
| **Monitoring: model** | Partial | Score histogram, decisions by rule and country | No drift metric (PSI) on features or score; no alert on `device_many_users_1h` volume |
| **Label feedback** | Open | | `POST /api/labels`; without it live precision/recall cannot be measured and retraining has no new labels |
| **Idempotency / retries** | Partial | Redis `SET NX` claim, 409 on in-flight duplicates, concurrency tests | No unique `idempotency_key` in Postgres (crash-after-commit gap) |
| **Failure handling** | Partial | Redis-miss fallbacks to Postgres (purchase history, device count), tracked background tasks, drained on shutdown | Behaviour when Redis is fully down is not tested; `ZADD GT` for out-of-order set updates |
| **Tests** | Partial | 167 unit/API/parity/contract tests, CI runs ruff + pytest + Docker build | Every DB test mocks Postgres (a type bug shipped this way once): one real-Postgres integration test in CI |
| **Performance** | Partial | Load tests and investigation (`docs/performance_2026-11.md`) | New first-seen-device Postgres lookup not load-tested; `score()` builds a DataFrame per request |
| **Deployment** | Partial | Docker Compose, 2 GB image, health endpoint, DB init scripts | Migrations are hand-applied on existing databases; no rollback procedure for a bad model beyond `git revert` |
| **Documentation** | Partial | Design trade-offs, architecture, evaluation, model comparison, feature study, this page | No model card; README still opens with the backend, not the ML story |
| **Security** | Open (out of scope, see §3) | Secrets in `.env`, not in git | No authentication or edge rate limiting on the API |

## 2. Roadmap, in order

Each item gets its own plan before code. The order is by what unblocks the most.

1. **Ship the 42-feature model with its review band.** The band, the two-threshold cost
   search and the extended ship gate are built and tested (2026-11-15); what remains is the
   approved `--ship`, the database migration and a rebuild.
2. **Model card and README for an ML reader.** `docs/model_card.md` (data, metrics with
   intervals, intended use, limits, the January bot attack, what the model must not be used
   for); README leads with evaluation and the feature study. Then make the repository public.
3. **Reason codes.** Top-3 SHAP contributions per scored decision (XGBoost `pred_contribs`),
   stored in the audit row, latency measured.
4. **Close the loop.** Label endpoint; PSI drift job for the top features and the score with
   a Grafana panel and alert; alert on hard-rule volume; a retraining entry point that runs
   the ship gate.
5. **Hardening.** Real-Postgres integration test in CI; unique idempotency key column;
   Redis-down test; `ZADD GT`; model artifact checksum at startup; load test of the new
   lookup; `inplace_predict` for scoring.

## 3. Out of scope, stated as limits

- **Real data.** Fraud_Data is public and simulated-looking (one purchase per user, flat
  fraud rate by country). Every number in this repository is about this dataset. The Sparkov
  ablation shows the velocity layer works when users repeat, on another simulated dataset.
- **Calibration.** Scores are not probabilities (`scale_pos_weight` inflates them); all
  thresholds are chosen on validation by cost or quantile, never read as probabilities.
- **Authentication, edge rate limiting, secrets management, multi-region, autoscaling.**
  Needed for a real deployment, not demonstrated here.
- **Fairness review.** `sex` and `age_group` are model inputs. Acceptable for a public
  teaching dataset; in a real system this needs a legal and fairness review before use.
- **Label delay.** Training treats labels as known immediately. Real chargebacks arrive
  weeks later; the Sparkov study reports label-history features separately for that reason.
