"""
The allow / review / deny policy: what it costs and how its two thresholds are chosen.

Kept apart from train.py and free of model libraries (numpy only) so the cost arithmetic
can be unit-tested in CI, which does not install the training dependencies.

Cost of a policy on a set of orders, in dollars:
    fraud that is allowed            its purchase value is lost
    legitimate order that is denied  false_decline_cost (lost margin + an angry customer)
    any order sent to review         review_cost (a step-up check or a minute of an analyst)
    fraud sent to review             stopped with probability catch_rate, otherwise lost
A denied fraud costs nothing. Review is cheaper than a wrong decline, so it is the right
action for scores that are risky but not risky enough for a decline to pay for itself.
"""
import numpy as np

NEVER = 1.01   # a threshold no score reaches: "never deny" / "never review"


def policy_cost(y, p, value, review_t, deny_t, false_decline_cost, review_cost, catch_rate=1.0):
    deny = p >= deny_t
    review = (p >= review_t) & ~deny
    fraud = y == 1
    return float(value[fraud & ~deny & ~review].sum()
                 + (1 - catch_rate) * value[fraud & review].sum()
                 + false_decline_cost * (deny & ~fraud).sum()
                 + review_cost * review.sum())


def pick_thresholds(y, p, value, false_decline_cost, review_cost, catch_rate=1.0,
                    tolerance=0.01):
    """(review_threshold, deny_threshold), review <= deny, chosen on these rows.

    Candidates: 401 quantiles of the scores plus NEVER. The cost of a pair splits into a
    part that depends only on the review threshold and a part that depends only on the
    deny threshold, so all ~80,000 valid pairs are costed with one array addition.

    Least-intrusive rule: the cost curve is flat near its minimum, and the exact minimum
    can sit on a policy that acts on far more orders for a few dollars. So among the pairs
    whose cost is within `tolerance` (1%) of the best, the one that acts on the FEWEST
    orders wins; then the fewest declines; then the lowest cost. tolerance=0 gives the
    plain minimum.
    """
    grid = np.append(np.unique(np.quantile(p, np.linspace(0, 1, 401))), NEVER)
    fraud = y == 1
    above = [p >= t for t in grid]
    fraud_value = np.array([value[a & fraud].sum() for a in above])   # fraud $ at or above t
    legit = np.array([(a & ~fraud).sum() for a in above])             # legit orders at or above t
    count = np.array([a.sum() for a in above])                        # all orders at or above t
    review_part = -catch_rate * fraud_value + review_cost * count
    deny_part = -(1 - catch_rate) * fraud_value + false_decline_cost * legit - review_cost * count
    cost = value[fraud].sum() + review_part[:, None] + deny_part[None, :]   # [review, deny]
    valid = np.arange(len(grid))[:, None] <= np.arange(len(grid))[None, :]
    cost = np.where(valid, cost, np.inf)

    near_best = cost <= cost.min() * (1 + tolerance) + 1e-9
    r_idx, d_idx = np.nonzero(near_best)
    # count[r] = orders reviewed or denied; count[d] = orders denied
    order = np.lexsort((cost[r_idx, d_idx], count[d_idx], count[r_idx]))
    r, d = r_idx[order[0]], d_idx[order[0]]
    return float(grid[r]), float(grid[d])


def policy_outcome(y, p, value, review_t, deny_t, false_decline_cost, review_cost,
                   catch_rate=1.0):
    deny = p >= deny_t
    review = (p >= review_t) & ~deny
    fraud = y == 1
    return {
        "review_threshold": round(review_t, 4), "deny_threshold": round(deny_t, 4),
        "denied": int(deny.sum()), "deny_rate": round(float(deny.mean()), 4),
        "reviewed": int(review.sum()), "review_rate": round(float(review.mean()), 4),
        "fraud_total": int(fraud.sum()),
        "fraud_denied": int((deny & fraud).sum()),
        "fraud_reviewed": int((review & fraud).sum()),
        "fraud_allowed": int((~deny & ~review & fraud).sum()),
        "false_declines": int((deny & ~fraud).sum()),
        "legit_reviewed": int((review & ~fraud).sum()),
        "review_precision": round(float(y[review].mean()), 4) if review.any() else None,
        "deny_precision": round(float(y[deny].mean()), 4) if deny.any() else None,
        "total_cost": round(policy_cost(y, p, value, review_t, deny_t, false_decline_cost,
                                        review_cost, catch_rate), 2),
    }


def cost_difference_ci(y, value, p_new, new_thresholds, p_old, old_thresholds,
                       false_decline_cost, review_cost, catch_rate=1.0, n_boot=1000, seed=42):
    """95% interval of (new policy cost - old policy cost) on the same resampled orders.
    Thresholds stay fixed; only the orders are resampled. Negative = the new policy is
    cheaper. Returned in dollars for a sample the size of the input."""
    rng = np.random.default_rng(seed)
    c = (false_decline_cost, review_cost, catch_rate)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(policy_cost(y[i], p_new[i], value[i], *new_thresholds, *c)
                     - policy_cost(y[i], p_old[i], value[i], *old_thresholds, *c))
    return [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]
