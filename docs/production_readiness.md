# Production readiness

What "ready" means here, where the project stands, and what is deliberately out of scope.
This is a portfolio project on public data, so the standard is: the model lifecycle is
closed (train, evaluate, explain, monitor, retrain), failure modes are tested,
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
| **Training/serving parity** | Done | Shared encodings; event-time windows; parity replay on synthetic (repeat users) and real rows; contract tests | The two all-time device features are arrival-order, not event-time (documented, pinned by a test) |
| **Feature meaning beyond this dataset** | Done | All 42 inputs audited (`docs/feature_audit.md`): nine removed or corrected at no accuracy cost; tests with repeat customers; a rule and a drop list with reasons for future features | 28 dormant candidates reviewed on paper only: they are constant here, so nothing can be measured |
| **Best available model in production** | Done | `xgboost-20261115222918-a9903ee`: 33 audited inputs, monotone in the device feature, test PR-AUC 0.140 (was 0.092), policy cost $44.3k (was $49.7k); live stack rebuilt, 8 demo stories pass | CatBoost is marginally ahead on the rolling folds; a tie by the selection rule |
| **Decision policy** | Partial | Hard rules; allow / review / deny on the model score with both thresholds chosen by cost on validation; shadow rules; rule backtest | Review cost and catch rate are assumptions, not measurements; velocity cut-offs in `rate_limiter_control` are expert-set and cannot fire on this data |
| **Ship safety** | Done | `--ship` checks two gates, then reloads the saved file the way the API does and requires identical scores before replacing the served model | Added after a model that trained but could not be served reached the repository |
| **Reproducible training** | Done | `python -m training.train`, fixed seed, pinned dependency groups, model version = algorithm + timestamp + git sha | No scheduled retraining; artifacts live in git, not a registry |
| **Ship gate** | Done | `--ship` requires test PR-AUC ≥ the previous model's CI lower bound **and** test cost of the new policy ≤ the served policy; run only on explicit approval | The cost rule uses the point estimate; the paired interval is reported next to it |
| **Auditability** | Done | Every decision stores features, `model_version`, `model_score`, `decision_rule` | No per-decision reason codes |
| **Explainability** | Open | Global feature importances in `model_meta.json` | Top-3 SHAP reason codes per scored decision |
| **Monitoring: system** | Done | The four golden signals: latency, traffic against its 1-day average, errors by status code, saturation (worker CPU, event-loop lag, memory, Postgres pool and connections, Redis memory); dependency probes; Redis key count; Postgres-fallback rate | Measured from inside the API; no host-level exporter |
| **Alerting** | Done | 29 alerts in four layers (golden signals, dependencies, action rates, drift), each with severity, traffic guard and a runbook (`docs/monitoring.md`); 19 promtool scenarios in CI; Alertmanager routing and inhibition; end-to-end drill passes | No notification channel configured (no secrets in git); thresholds are starting points, not tuned on production traffic |
| **Monitoring: model** | Done | Action rates (allow / review / deny against the 1-day normal and the expected rate, share per rule) as the proxy for decision quality; drift of the score and nine inputs (PSI at most 0.004 on unseen months); model-version skew | Live decision accuracy is out of scope by decision: live traffic has no labels. Accuracy is measured offline at each retrain |
| **Demo** | Done | `scripts/demo.py`: eight labelled stories against the live stack plus a traffic replay; `scripts/alert_drill.py`; all pass on the rebuilt stack; a shareable project-tour page | Not run in CI |
| **Idempotency / retries** | Partial | Redis `SET NX` claim, 409 on in-flight duplicates, concurrency tests | No unique `idempotency_key` in Postgres (crash-after-commit gap) |
| **Failure handling** | Partial | Redis-miss fallbacks to Postgres (purchase history, device count), tracked background tasks, drained on shutdown | Behaviour when Redis is fully down is not tested; `ZADD GT` for out-of-order set updates |
| **Tests** | Partial | 288 unit/API/parity/contract/policy/monitoring tests, six of them against the real model file; CI runs ruff + pytest + alert-rule tests + Docker build | Every DB test mocks Postgres (a type bug shipped this way once): one real-Postgres integration test in CI |
| **Performance** | Partial | Load tests and investigation (`docs/performance_2026-11.md`); before/after run for the new device lookups shows no regression (p99 about 20 ms at 200 RPS) | That run was on a database full of replayed rows; a clean-database run with first-seen devices is owed; `score()` builds a DataFrame per request |
| **Deployment** | Partial | Docker Compose, 2 GB image, health endpoint, DB init scripts | Migrations are hand-applied on existing databases; no rollback procedure for a bad model beyond `git revert` |
| **Documentation** | Partial | Design trade-offs, architecture, evaluation, model comparison, feature study, this page | No model card; README still opens with the backend, not the ML story |
| **Security** | Open (out of scope, see §3) | Secrets in `.env`, not in git | No authentication or edge rate limiting on the API |

## 2. Roadmap, in order

Each item gets its own plan before code. The order is by what unblocks the most.

Done so far (2026-11-15): the better model shipped with a review band; the feature audit
(33 inputs); monitoring and alerting in four layers.

1. **Model card and README for an ML reader.** `docs/model_card.md` (data, metrics with
   intervals, intended use, limits, the January bot attack, what the model must not be used
   for); README leads with evaluation and the feature study. Then make the repository public.
2. **Reason codes.** Top-3 SHAP contributions per scored decision (XGBoost `pred_contribs`),
   stored in the audit row, latency measured.
3. **A retraining entry point** that runs the ship gates and the saved-model check on a
   schedule or on a drift alert.
4. **Hardening.** Real-Postgres integration test in CI; unique idempotency key column;
   Redis-down test; `ZADD GT`; model artifact checksum at startup; balance connections
   across workers; `inplace_predict` for scoring; the demo and the alert drill in CI.

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
- **Live decision accuracy.** Not monitored, by decision: live traffic has no labels. Action
  rates and drift are watched instead (`docs/monitoring.md`); accuracy is measured offline.
- **Label delay.** Training treats labels as known immediately. Real chargebacks arrive
  weeks later; the Sparkov study reports label-history features separately for that reason.
