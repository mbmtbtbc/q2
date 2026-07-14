"""
Paired significance testing for the quantum-vs-classical claims.

Addresses: "run a paired significance test (Wilcoxon signed-rank across the
N-1 trials, since it's the same graph under different contingencies) rather
than eyeballing one TVD number."

Consumes qgrid/experiments/results/n1_sweep.json (run n1_sweep.py first) and
runs scipy.stats.wilcoxon on each MATCHED pair of per-trial measurements --
matched because every pair comes from the identical N-1 contingency (the
same seed_edge / same damaged graph), which is exactly the paired-samples
setup Wilcoxon signed-rank assumes, instead of an unpaired/independent-samples
test that would throw away that structure.

Tests run:
  1. Quantum vs classical IPR (final)               -- localization claim
  2. Quantum vs classical hitting time               -- speed-to-target claim
     (paired only on trials where BOTH are finite -- see note in output)
  3. Quantum vs classical spread distance at t~3.0   -- ballistic-vs-diffusive claim
  4. QAOA-best vs exact-optimum buses recovered      -- "QAOA is near-optimal"
     sanity check (expect small/non-significant difference)
  5. QAOA-best vs random-baseline buses recovered    -- "QAOA beats naive
     switching" claim
  6. QAOA-best vs greedy buses recovered             -- "QAOA is competitive
     with a standard restoration heuristic" claim

Wilcoxon signed-rank requires the paired differences to be non-zero for at
least one pair and drops exact zero-differences internally; if ALL
differences are zero (e.g. two methods always agree) it is reported as
"degenerate: no non-zero differences" rather than a crash or a fabricated
p-value.

Run:
    python3 -m qgrid.experiments.n1_sweep      # if results/n1_sweep.json missing
    python3 -m qgrid.experiments.significance

Writes qgrid/experiments/results/significance.json
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
from scipy import stats

from qgrid.experiments.common import RESULTS_DIR, npify


def _load_trials(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["trials"], payload["meta"]


def _paired_from(trials, get_a, get_b, require_both_finite=True):
    """
    Extracts matched (a, b) pairs from trials via accessor callables, one pair
    per trial that has both values defined. Returns (a_arr, b_arr, n_dropped,
    n_total) so callers can report how many trials were usable.
    """
    a_vals, b_vals = [], []
    n_total = len(trials)
    for t in trials:
        a, b = get_a(t), get_b(t)
        if require_both_finite:
            if a is None or b is None:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
        a_vals.append(a)
        b_vals.append(b)
    return np.array(a_vals, dtype=float), np.array(b_vals, dtype=float), n_total - len(a_vals), n_total


def wilcoxon_paired(a, b, label, alternative="two-sided"):
    """
    Runs scipy.stats.wilcoxon(a, b). Handles the degenerate all-zero-difference
    case explicitly instead of letting scipy raise, and reports the median
    paired difference (a - b) alongside the test statistic, since a p-value
    alone doesn't convey direction or effect size.
    """
    n = len(a)
    if n < 1:
        return {"label": label, "n_pairs": 0, "status": "no_data"}
    diff = a - b
    if np.all(diff == 0):
        return {"label": label, "n_pairs": n, "status": "degenerate_all_zero_diff",
                 "median_diff": 0.0}
    n_nonzero = int(np.sum(diff != 0))
    if n_nonzero < 1:
        return {"label": label, "n_pairs": n, "status": "degenerate_all_zero_diff",
                 "median_diff": float(np.median(diff))}
    try:
        res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        out = {
            "label": label, "n_pairs": n, "n_nonzero_diff": n_nonzero, "status": "ok",
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "alternative": alternative,
            "median_diff_a_minus_b": float(np.median(diff)),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "significant_at_0.05": bool(res.pvalue < 0.05),
        }
    except ValueError as e:
        out = {"label": label, "n_pairs": n, "n_nonzero_diff": n_nonzero,
                "status": f"wilcoxon_failed: {e}"}
    return out


def spread_at_time(trial, times, t_query):
    """Nearest-timestep spread-curve value for a trial at t_query."""
    idx = int(np.argmin(np.abs(np.array(times) - t_query)))
    return trial["benchmark"]["quantum_spread_curve"][idx], trial["benchmark"]["classical_spread_curve"][idx]


def run(sweep_path=None):
    if sweep_path is None:
        sweep_path = os.path.join(RESULTS_DIR, "n1_sweep.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"{sweep_path} not found -- run `python3 -m qgrid.experiments.n1_sweep` first.")
    trials, meta = _load_trials(sweep_path)
    trials = [t for t in trials if not t.get("is_tie_switch_seed")]

    # times grid used inside run_trial()'s default (np.linspace(0, 6, 41)) --
    # only correct if n1_sweep.py was run with default --qaoa/no custom `times`
    # override; recorded here explicitly rather than silently assumed elsewhere.
    times = np.linspace(0, 6, 41)
    t_query = 3.0

    results = {}

    # 1. IPR: quantum vs classical (lower IPR = more localized/ballistic)
    a, b, dropped, total = _paired_from(
        trials, lambda t: t["benchmark"]["quantum_ipr_final"],
        lambda t: t["benchmark"]["classical_ipr_final"])
    results["ipr_quantum_vs_classical"] = wilcoxon_paired(
        a, b, "Quantum IPR vs Classical IPR (final t)", alternative="less")
    results["ipr_quantum_vs_classical"]["n_dropped_nonfinite"] = dropped
    results["ipr_quantum_vs_classical"]["n_total_trials"] = total

    # 2. Hitting time: quantum vs classical (only trials where BOTH finite)
    a, b, dropped, total = _paired_from(
        trials, lambda t: t["benchmark"]["quantum_hitting_time"],
        lambda t: t["benchmark"]["classical_hitting_time"])
    results["hitting_time_quantum_vs_classical"] = wilcoxon_paired(
        a, b, "Quantum hitting time vs Classical hitting time", alternative="two-sided")
    results["hitting_time_quantum_vs_classical"]["n_dropped_nonfinite"] = dropped
    results["hitting_time_quantum_vs_classical"]["n_total_trials"] = total
    results["hitting_time_quantum_vs_classical"]["note"] = (
        "Only trials where both quantum and classical hitting time were finite "
        "are included; see n1_sweep.py docstring re: weak-coupling-induced "
        "non-finite hitting times on this radial feeder."
    )

    # 3. Spread distance at t~3.0 (ballistic vs diffusive)
    spreads = [spread_at_time(t, times, t_query) for t in trials]
    a = np.array([s[0] for s in spreads])
    b = np.array([s[1] for s in spreads])
    results["spread_at_t3_quantum_vs_classical"] = wilcoxon_paired(
        a, b, f"Quantum vs classical mean spread distance at t={t_query}",
        alternative="two-sided")
    results["spread_at_t3_quantum_vs_classical"]["n_dropped_nonfinite"] = 0
    results["spread_at_t3_quantum_vs_classical"]["n_total_trials"] = len(trials)

    # 4. QAOA-best vs exact-optimum buses recovered (near-optimality sanity check)
    a = np.array([t["qaoa_recovered_best"] for t in trials], dtype=float)
    b = np.array([t["exact_recovered"] for t in trials], dtype=float)
    results["recovered_qaoa_vs_exact"] = wilcoxon_paired(
        a, b, "Buses recovered: QAOA(best-of-restarts) vs exact optimum",
        alternative="two-sided")
    results["recovered_qaoa_vs_exact"]["n_dropped_nonfinite"] = 0
    results["recovered_qaoa_vs_exact"]["n_total_trials"] = len(trials)

    # 5. QAOA-best vs random-baseline buses recovered
    a = np.array([t["qaoa_recovered_best"] for t in trials], dtype=float)
    b = np.array([t["random_baseline_recovered"] for t in trials], dtype=float)
    results["recovered_qaoa_vs_random"] = wilcoxon_paired(
        a, b, "Buses recovered: QAOA(best-of-restarts) vs random baseline",
        alternative="greater")
    results["recovered_qaoa_vs_random"]["n_dropped_nonfinite"] = 0
    results["recovered_qaoa_vs_random"]["n_total_trials"] = len(trials)

    # 6. QAOA-best vs greedy buses recovered
    a = np.array([t["qaoa_recovered_best"] for t in trials], dtype=float)
    b = np.array([t["greedy_recovered"] for t in trials], dtype=float)
    results["recovered_qaoa_vs_greedy"] = wilcoxon_paired(
        a, b, "Buses recovered: QAOA(best-of-restarts) vs greedy heuristic",
        alternative="two-sided")
    results["recovered_qaoa_vs_greedy"]["n_dropped_nonfinite"] = 0
    results["recovered_qaoa_vs_greedy"]["n_total_trials"] = len(trials)

    return npify({"source_file": sweep_path, "source_meta": meta, "tests": results})


def _print_summary(payload):
    print("=" * 92)
    print("Paired Wilcoxon signed-rank tests across N-1 trials "
          f"(source: {payload['source_file']})")
    print("-" * 92)
    for key, r in payload["tests"].items():
        if r.get("status") == "no_data":
            print(f"  {r['label']:60s}  no data")
            continue
        if r.get("status", "").startswith("degenerate"):
            print(f"  {r['label']:60s}  degenerate (all paired diffs = 0), n={r['n_pairs']}")
            continue
        if r.get("status", "").startswith("wilcoxon_failed"):
            print(f"  {r['label']:60s}  FAILED: {r['status']}")
            continue
        sig = "***" if r["significant_at_0.05"] else "   "
        print(f"  {r['label']:60s}  W={r['statistic']:.2f}  p={r['p_value']:.4g} {sig}  "
              f"median(a-b)={r['median_diff_a_minus_b']:+.4f}  "
              f"n={r['n_pairs']}/{r['n_total_trials']}"
              + (f"  ({r['n_dropped_nonfinite']} dropped)" if r.get("n_dropped_nonfinite") else ""))
    print("-" * 92)
    print("*** = significant at alpha=0.05. 'a' is always the FIRST-named quantity in the")
    print("label; median_diff = median(a - b) across paired trials.")
    print("=" * 92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-file", type=str, default=None)
    args = ap.parse_args()

    payload = run(sweep_path=args.sweep_file)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "significance.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}\n")
    _print_summary(payload)