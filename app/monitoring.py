"""
Drift monitoring: compare live traffic with what the model saw when it was trained.

The idea, in three steps:
  1. At training time (training/train.py) the served model's scores on the test period
     (the most recent data it has not seen) and a few important features are cut into
     buckets, and the share of traffic in each bucket is written to model_meta.json under
     "monitoring_reference". That is what "normal" looks like.
  2. At serving time (app/main.py) every scored request adds 1 to the counter of the bucket
     its score and each monitored feature falls in.
  3. Prometheus turns the counters into live shares and compares them with the reference
     using PSI (prometheus/rules/fraud_engine.yml). Alerts fire when PSI stays high.

PSI (Population Stability Index) = sum over buckets of (live - expected) * ln(live / expected).
0 means identical distributions; by convention 0.10 is "watch", 0.25 is "act".

Pure functions only (no Prometheus, no model libraries) so both training and the API can
import this module and the arithmetic can be unit-tested.
"""
import bisect
import math
from collections.abc import Sequence

# Model inputs worth watching: the ones the model leans on (model_meta.json top_features)
# plus two request fields whose distribution says "the traffic itself changed".
# Only features that are inputs of the served model get a reference (train.py filters this
# list). History: on the first live run the score drifted (PSI 0.63) while none of the
# monitored inputs did; the cause was two unmonitored inputs, country_txn_count_30d and
# amount_vs_user_avg_30d. Both were flawed and are no longer model inputs
# (docs/feature_audit.md), but the lesson stands: monitor every kind of input the model
# has, not only the ones that look important.
MONITORED_FEATURES = ("did_other_uid_all", "did_uid_90d", "did_uid_30d", "did_uid_7d",
                      "ip_uid_7d", "purchase_value", "age_account_gap", "hour_of_day",
                      "day_of_week")
SCORE_BUCKETS = 10      # deciles of the validation score: each expected share is ~10%
FEATURE_BUCKETS = 5     # quintiles, before small buckets are merged
MIN_SHARE = 0.01        # a bucket expected to hold < 1% of traffic makes PSI jumpy: merge it
PSI_EPSILON = 1e-4      # floor for a share, so an empty live bucket does not give ln(0)


def bucket_index(edges: Sequence[float], x: float) -> int:
    """Bucket of x for ascending cut points `edges`: the number of edges <= x.
    edges [1, 2] -> x < 1 is bucket 0, 1 <= x < 2 is bucket 1, x >= 2 is bucket 2."""
    return bisect.bisect_right(edges, x)


def bucket_shares(edges: Sequence[float], values) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bucket_index(edges, v)] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def reference_buckets(values, n_buckets: int) -> dict:
    """Cut points at the quantiles of `values`, then merge any bucket expected to hold less
    than MIN_SHARE of traffic into its neighbour. Returns {"edges": [...], "shares": [...]}.

    A feature like did_other_uid_all is 0 for 95% of traffic, so its quantiles collapse to
    one or two distinct cut points; that is fine, the buckets are then {0, 1, 2+}."""
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    edges = {ordered[min(n - 1, (n * k) // n_buckets)] for k in range(1, n_buckets)}
    # A count that is 0 for most traffic has all its quantiles at 0. Its interesting values
    # are the few distinct ones above the minimum, so offer those as cut points too.
    if len(edges) < n_buckets - 1:
        edges |= set(sorted(set(ordered))[1:n_buckets])
    # An edge equal to the minimum would leave bucket 0 empty.
    edges = sorted(e for e in edges if e > ordered[0])
    while edges:
        shares = bucket_shares(edges, ordered)
        smallest = min(range(len(shares)), key=shares.__getitem__)
        if shares[smallest] >= MIN_SHARE:
            break
        # Bucket i sits between edges[i-1] and edges[i]; drop the edge towards the
        # smaller neighbour so the tiny bucket joins it.
        if smallest == 0:
            edges.pop(0)
        elif smallest == len(shares) - 1 or shares[smallest - 1] <= shares[smallest + 1]:
            edges.pop(smallest - 1)
        else:
            edges.pop(smallest)
    return {"edges": [round(e, 6) for e in edges],
            "shares": [round(s, 6) for s in bucket_shares(edges, ordered)]}


def psi(expected: Sequence[float], live: Sequence[float], epsilon: float = PSI_EPSILON) -> float:
    """Population Stability Index between two lists of bucket shares."""
    total = 0.0
    for e, a in zip(expected, live, strict=True):
        e, a = max(e, epsilon), max(a, epsilon)
        total += (a - e) * math.log(a / e)
    return total
