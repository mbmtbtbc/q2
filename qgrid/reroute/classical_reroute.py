"""
Classical baselines for the self-healing reroute QUBO built by
`reroute.qaoa_reroute.build_qubo` (imported, not modified).

The tie-switch candidate set is small (a handful of open switches per feeder,
per the QAOA module's own docstring), so the QUBO
    minimize  sum_i h_i x_i + sum_{i<j} J_ij x_i x_j ,   x_i in {0,1}
is exactly solvable by brute force in microseconds. That means QAOA's output
should always be reported *relative to the exact optimum* (an "optimality gap"),
not presented on its own -- otherwise there is no way to tell whether a QAOA
result (e.g. "close nothing") reflects the true optimum or an optimizer/formulation
failure. This module adds:

  - exact_solve       : brute-force enumeration over all 2^K assignments (ground truth)
  - greedy_solve       : literature-standard greedy heuristic for restoration switching
                          (repeatedly close the single switch with the best marginal
                          cost reduction until no closing improves the objective)
  - simulated_annealing_solve : classical SA baseline on the same QUBO cost
  - random_solve        : uniform-random baseline (mean/std/best-of-N), the floor
                           "buses recovered" results should sit above
  - compare_reroute_methods : runs all of the above plus (optionally) a supplied
                           QAOA decision, and reports each method's cost and its
                           optimality gap vs. the exact solution.
"""
from __future__ import annotations
import itertools
import numpy as np


def qubo_cost(x, h, J):
    """QUBO objective sum_i h_i x_i + sum_{i<j} J_ij x_i x_j for x in {0,1}^K."""
    x = np.asarray(x)
    cost = float(np.dot(h, x))
    for (i, j), Jij in J.items():
        cost += Jij * x[i] * x[j]
    return cost


def exact_solve(h, J, max_k=24):
    """
    Brute-force ground truth. K = len(h) candidates -> 2^K assignments. Realistic
    tie-switch candidate sets are O(5), so this is exact and near-instant; max_k
    guards against accidentally being called on a much larger candidate set.
    """
    K = len(h)
    if K == 0:
        return {}, 0.0
    if K > max_k:
        raise ValueError(
            f"exact_solve got K={K} candidates (> max_k={max_k}); brute force is "
            f"only appropriate for the small tie-switch candidate sets this pipeline "
            f"produces. Use simulated_annealing_solve or greedy_solve instead."
        )
    best_x, best_cost = None, np.inf
    for bits in itertools.product([0, 1], repeat=K):
        c = qubo_cost(bits, h, J)
        if c < best_cost:
            best_cost, best_x = c, bits
    return {i: int(b) for i, b in enumerate(best_x)}, float(best_cost)


def greedy_solve(h, J):
    """
    Standard greedy switching heuristic: starting from all-open (x=0), repeatedly
    close whichever remaining switch gives the largest cost *decrease* (accounting
    for pairwise penalties already-closed switches induce), stopping when no
    remaining switch improves the objective. This is the natural classical
    competitor in distribution-restoration literature (rank candidates by
    reward-to-penalty ratio, apply greedily), not brute force.
    """
    K = len(h)
    if K == 0:
        return {}, 0.0
    x = np.zeros(K, dtype=int)
    closed = set()
    improved = True
    while improved:
        improved = False
        best_i, best_delta = None, 0.0
        for i in range(K):
            if i in closed:
                continue
            x[i] = 1
            new_cost = qubo_cost(x, h, J)
            x[i] = 0
            old_cost = qubo_cost(x, h, J)
            delta = new_cost - old_cost
            if delta < best_delta - 1e-12:
                best_delta, best_i = delta, i
        if best_i is not None:
            x[best_i] = 1
            closed.add(best_i)
            improved = True
    return {i: int(b) for i, b in enumerate(x)}, float(qubo_cost(x, h, J))


def random_solve(h, J, n_trials: int = 500, seed: int = 42):
    """
    Uniform-random baseline. Reports the best sample found (the "you could get
    lucky" ceiling to compare QAOA/greedy against) as well as the mean/std cost
    across all trials (the real floor -- what an uninformed switching policy
    achieves on average).
    """
    K = len(h)
    if K == 0:
        return {}, 0.0, {"mean_cost": 0.0, "std_cost": 0.0, "n_trials": 0}
    rng = np.random.default_rng(seed)
    costs = np.empty(n_trials)
    best_x, best_cost = None, np.inf
    for t in range(n_trials):
        x = rng.integers(0, 2, size=K)
        c = qubo_cost(x, h, J)
        costs[t] = c
        if c < best_cost:
            best_cost, best_x = c, x.copy()
    stats = {"mean_cost": float(costs.mean()), "std_cost": float(costs.std()), "n_trials": n_trials}
    return {i: int(b) for i, b in enumerate(best_x)}, float(best_cost), stats


def simulated_annealing_solve(h, J, n_iter: int = 3000, t_start: float = 2.0,
                               t_end: float = 0.01, seed: int = 42):
    """
    Simple bit-flip simulated annealing on the same QUBO cost (geometric cooling
    schedule). Standard classical metaheuristic baseline for combinatorial
    reroute/restoration problems -- the real competitor QAOA is meant to beat,
    as opposed to brute force which is only a ground-truth reference.
    """
    K = len(h)
    if K == 0:
        return {}, 0.0
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=K)
    cost = qubo_cost(x, h, J)
    best_x, best_cost = x.copy(), cost
    cooling = np.geomspace(t_start, t_end, n_iter)
    for T in cooling:
        i = rng.integers(0, K)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        new_cost = qubo_cost(x_new, h, J)
        delta = new_cost - cost
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            x, cost = x_new, new_cost
            if cost < best_cost:
                best_x, best_cost = x.copy(), cost
    return {i: int(b) for i, b in enumerate(best_x)}, float(best_cost)


def compare_reroute_methods(h, J, tie_candidates, qaoa_decision=None, qaoa_cost=None,
                             n_random_trials: int = 500, sa_iter: int = 3000, seed: int = 42):
    """
    Runs exact / greedy / simulated-annealing / random on the same QUBO, and
    (if supplied) reports the QAOA decision's cost alongside them -- all expressed
    as an absolute cost and as an optimality gap relative to the exact optimum:
        gap = method_cost - exact_cost   (0 = optimal; the QUBO is a minimization,
                                           so gap is always >= 0 by construction of
                                           exact_solve, up to numerical noise)
    """
    K = len(h)
    exact_decision, exact_cost = exact_solve(h, J)
    greedy_decision, greedy_cost = greedy_solve(h, J)
    sa_decision, sa_cost = simulated_annealing_solve(h, J, n_iter=sa_iter, seed=seed)
    rand_decision, rand_best_cost, rand_stats = random_solve(h, J, n_trials=n_random_trials, seed=seed)

    def gap(c):
        return float(c - exact_cost)

    def decisions_to_switches(decision):
        return [list(tie_candidates[i]) for i, closed in decision.items() if closed]

    out = {
        "n_candidates": K,
        "tie_candidates": [list(t) for t in tie_candidates],
        "exact": {"decision": exact_decision, "closed_switches": decisions_to_switches(exact_decision),
                  "cost": exact_cost, "gap": 0.0},
        "greedy": {"decision": greedy_decision, "closed_switches": decisions_to_switches(greedy_decision),
                   "cost": greedy_cost, "gap": gap(greedy_cost)},
        "simulated_annealing": {"decision": sa_decision, "closed_switches": decisions_to_switches(sa_decision),
                                 "cost": sa_cost, "gap": gap(sa_cost)},
        "random_baseline": {"decision": rand_decision, "closed_switches": decisions_to_switches(rand_decision),
                             "best_of_n_cost": rand_best_cost, "best_of_n_gap": gap(rand_best_cost),
                             **rand_stats,
                             "mean_gap": gap(rand_stats["mean_cost"])},
    }
    if qaoa_decision is not None:
        q_cost = qaoa_cost if qaoa_cost is not None else qubo_cost(
            [qaoa_decision.get(i, 0) for i in range(K)], h, J)
        out["qaoa"] = {"decision": qaoa_decision, "closed_switches": decisions_to_switches(qaoa_decision),
                        "cost": float(q_cost), "gap": gap(q_cost)}
    return out