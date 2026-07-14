"""
Hyperparameter sensitivity / ablation.

Addresses: "the alpha/beta/gamma weight split (0.5/0.2/0.3), the
dephasing/diffusion rates (0.05/0.01), QAOA depth p and shot count -- sweep
at least alpha/beta/gamma and p."

Two sweeps, each reusing existing, unmodified qgrid code:

1. alpha/beta/gamma weight-mix sweep (`sweep_weights`)
   For each (alpha, beta, gamma) on the simplex (alpha+beta+gamma=1):
     - rebuild the graph's quantum weights with graph.weights.encode_quantum_weights
     - recompute the CTQW-occupation-vs-N-1-severity correlation via the
       EXISTING benchmark.classical_baselines.benchmark_criticality_baselines
       (this is the pipeline's own headline "the weight encoding carries real
       criticality signal, not an arbitrary choice" claim -- exactly what
       should NOT be an artifact of the hardcoded 0.5/0.2/0.3 default)
     - run a representative SUBSET of N-1 trials (not the full 32, to keep
       the sweep tractable -- see --n-trial-subset) and aggregate IPR / TVD /
       buses-recovered the same way n1_sweep.py does

2. QAOA depth (p) x shots sweep (`sweep_qaoa`)
   On a single representative post-cascade scenario (default seed_edge=(7,8),
   which produces a non-trivial 11-bus outage in the default topology), sweep
   p in {1,2,3,4} and shots in {256,1024,4096}, each with multiple restarts
   (see qaoa_with_restarts in common.py), and report best/mean/std cost and
   optimality gap vs the exact QUBO solution for every (p, shots) cell -- so
   a claim like "QAOA recovers N buses" can be checked for whether it depends
   on the arbitrary p=2, shots=1024 defaults in reroute/qaoa_reroute.py.

Run:
    python3 -m qgrid.experiments.sensitivity
    python3 -m qgrid.experiments.sensitivity --n-trial-subset 10 --qaoa-restarts 10

Writes qgrid/experiments/results/sensitivity.json
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np

from qgrid.experiments.common import (
    build_base_graph, active_non_tie_lines, run_trial, qaoa_with_restarts,
    summarize, npify, RESULTS_DIR,
)
from qgrid.walk.ctqw import ContinuousTimeQuantumWalk
from qgrid.failures.topology import criticality_ranking
from qgrid.failures.cascade import simulate_cascade
from qgrid.benchmark.classical_baselines import benchmark_criticality_baselines
from qgrid.reroute.qaoa_reroute import build_qubo
from qgrid.reroute.classical_reroute import exact_solve


DEFAULT_ABG_GRID = [
    (0.50, 0.20, 0.30),  # repo default
    (1.00, 0.00, 0.00),  # admittance-only
    (0.00, 1.00, 0.00),  # capacity-only
    (0.00, 0.00, 1.00),  # headroom-only
    (0.34, 0.33, 0.33),  # uniform
    (0.20, 0.20, 0.60),  # headroom-heavy (opposite skew from default)
    (0.70, 0.15, 0.15),  # admittance-heavy
]


def _pick_subset(lines, n):
    """Evenly-spaced subset of the N-1 line list, not just the first n, so the
    sample isn't biased toward lines nearest the slack bus."""
    if n >= len(lines):
        return list(lines)
    idxs = np.linspace(0, len(lines) - 1, n).round().astype(int)
    idxs = sorted(set(idxs.tolist()))
    return [lines[i] for i in idxs]


def sweep_weights(abg_grid=None, n_trial_subset=8, qaoa_restarts=8,
                   occupation_source_bus=1, occupation_t=6.0, verbose=True):
    abg_grid = abg_grid or DEFAULT_ABG_GRID
    rows = []
    for (alpha, beta, gamma) in abg_grid:
        if verbose:
            print(f"  alpha/beta/gamma = {alpha:.2f}/{beta:.2f}/{gamma:.2f}", flush=True)
        G0, meta = build_base_graph(alpha=alpha, beta=beta, gamma=gamma)
        lines = active_non_tie_lines(G0)
        subset = _pick_subset(lines, n_trial_subset)

        # (a) criticality-signal correlation (existing pipeline code, unmodified)
        qw = ContinuousTimeQuantumWalk(G0)
        occ = qw.occupation_at(occupation_source_bus, t=occupation_t)
        ranking = criticality_ranking(G0)
        crit_report = benchmark_criticality_baselines(G0, ctqw_occupation=occ,
                                                        existing_criticality_ranking=ranking)
        n1_sev_corr = crit_report["correlations"]["n1_severity_load_lost_kw"]["vs_ctqw_occupation"]
        degree_corr = crit_report["correlations"]["degree_centrality"]["vs_ctqw_occupation"]

        # (b) N-1 trial subset: IPR / TVD / buses recovered
        trial_results = [run_trial(edge, G0=G0, alpha=alpha, beta=beta, gamma=gamma,
                                    qaoa_restarts=qaoa_restarts) for edge in subset]
        row = {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "n1_severity_spearman_rho_vs_ctqw_occupation":
                n1_sev_corr["spearman_rho"] if n1_sev_corr else None,
            "n1_severity_spearman_p":
                n1_sev_corr["spearman_p"] if n1_sev_corr else None,
            "degree_centrality_spearman_rho_vs_ctqw_occupation":
                degree_corr["spearman_rho"] if degree_corr else None,
            "n_trials_in_subset": len(subset),
            "quantum_ipr_final": summarize([t["benchmark"]["quantum_ipr_final"] for t in trial_results]),
            "final_tvd_quantum_vs_classical":
                summarize([t["benchmark"]["final_tvd_quantum_vs_classical"] for t in trial_results]),
            "qaoa_recovered_best": summarize([t["qaoa_recovered_best"] for t in trial_results]),
        }
        rows.append(row)
    return npify({"grid": abg_grid, "n_trial_subset": n_trial_subset,
                   "qaoa_restarts": qaoa_restarts, "rows": rows})


def sweep_qaoa(seed_edge=(7, 8), p_grid=(1, 2, 3, 4), shots_grid=(256, 1024, 4096),
               qaoa_restarts=10, qaoa_maxiter=40, verbose=True):
    G0, meta = build_base_graph()
    Gf, snaps, log = simulate_cascade(
        G0, seed_edges=[seed_edge], overload_threshold=1.0,
        alpha=0.5, beta=0.2, gamma=0.3,
    )
    slack = G0.graph["slack_bus"]
    qw_post = ContinuousTimeQuantumWalk(Gf)
    occ = qw_post.occupation_at(slack, t=3.0)
    tie_candidates = [(u, v) for u, v, d in Gf.edges(data=True) if d.get("is_tie_switch")]
    h_qubo, J_qubo, cands = build_qubo(Gf, tie_candidates, occ)
    exact_decision, exact_cost = exact_solve(h_qubo, J_qubo) if len(h_qubo) else ({}, 0.0)

    rows = []
    for p in p_grid:
        for shots in shots_grid:
            if verbose:
                print(f"  p={p}  shots={shots}", flush=True)
            stats_ = qaoa_with_restarts(h_qubo, J_qubo, p=p, shots=shots,
                                         maxiter=qaoa_maxiter, n_restarts=qaoa_restarts,
                                         base_seed=2000 + p * 100 + shots)
            rows.append({
                "p": p, "shots": shots, "n_restarts": qaoa_restarts,
                "exact_cost": exact_cost,
                "best_cost": stats_["best_cost"], "mean_cost": stats_["mean_cost"],
                "std_cost": stats_["std_cost"], "worst_cost": stats_["worst_cost"],
                "best_gap_vs_exact": stats_["best_cost"] - exact_cost,
                "mean_gap_vs_exact": stats_["mean_cost"] - exact_cost,
            })
    return npify({"seed_edge": list(seed_edge), "n_tie_candidates": len(tie_candidates),
                   "exact_cost": exact_cost, "p_grid": list(p_grid), "shots_grid": list(shots_grid),
                   "qaoa_restarts": qaoa_restarts, "rows": rows})


def run(n_trial_subset=8, weight_qaoa_restarts=8, qaoa_seed_edge=(7, 8),
        p_grid=(1, 2, 3, 4), shots_grid=(256, 1024, 4096), qaoa_hyperparam_restarts=10,
        verbose=True):
    if verbose:
        print("=== alpha/beta/gamma weight-mix sweep ===")
    weights_result = sweep_weights(n_trial_subset=n_trial_subset,
                                    qaoa_restarts=weight_qaoa_restarts, verbose=verbose)
    if verbose:
        print("\n=== QAOA depth (p) x shots sweep ===")
    qaoa_result = sweep_qaoa(seed_edge=qaoa_seed_edge, p_grid=p_grid, shots_grid=shots_grid,
                              qaoa_restarts=qaoa_hyperparam_restarts, verbose=verbose)
    return npify({"weight_sweep": weights_result, "qaoa_hyperparam_sweep": qaoa_result})


def _print_summary(payload):
    print("\n" + "=" * 100)
    print("WEIGHT MIX (alpha/beta/gamma) SENSITIVITY")
    print("-" * 100)
    print(f"{'alpha':>6} {'beta':>6} {'gamma':>6} | {'N1sev rho':>10} {'(p)':>9} | "
          f"{'deg rho':>8} | {'IPR mean+/-std':>16} | {'TVD mean+/-std':>16} | {'recov mean+/-std':>18}")
    for r in payload["weight_sweep"]["rows"]:
        n1 = r["n1_severity_spearman_rho_vs_ctqw_occupation"]
        n1p = r["n1_severity_spearman_p"]
        deg = r["degree_centrality_spearman_rho_vs_ctqw_occupation"]
        ipr = r["quantum_ipr_final"]
        tvd = r["final_tvd_quantum_vs_classical"]
        rec = r["qaoa_recovered_best"]
        n1_str = f"{n1:+.3f}" if n1 is not None else "n/a"
        n1p_str = f"({n1p:.3g})" if n1p is not None else ""
        deg_str = f"{deg:+.3f}" if deg is not None else "n/a"
        ipr_str = f"{ipr['mean']:.3f}+/-{ipr['std']:.3f}" if ipr["n_valid"] else "n/a"
        tvd_str = f"{tvd['mean']:.3f}+/-{tvd['std']:.3f}" if tvd["n_valid"] else "n/a"
        rec_str = f"{rec['mean']:.2f}+/-{rec['std']:.2f}" if rec["n_valid"] else "n/a"
        print(f"{r['alpha']:6.2f} {r['beta']:6.2f} {r['gamma']:6.2f} | {n1_str:>10} {n1p_str:>9} | "
              f"{deg_str:>8} | {ipr_str:>16} | {tvd_str:>16} | {rec_str:>18}")
    print(f"(each row's IPR/TVD/recovered computed over an {payload['weight_sweep']['n_trial_subset']}"
          f"-trial N-1 subset, QAOA restarts={payload['weight_sweep']['qaoa_restarts']} per trial)")

    print("\n" + "=" * 100)
    q = payload["qaoa_hyperparam_sweep"]
    print(f"QAOA DEPTH (p) x SHOTS SENSITIVITY  (seed_edge={tuple(q['seed_edge'])}, "
          f"K={q['n_tie_candidates']} candidates, exact_cost={q['exact_cost']:.4f}, "
          f"restarts={q['qaoa_restarts']})")
    print("-" * 100)
    print(f"{'p':>3} {'shots':>7} | {'best cost':>10} {'mean cost':>10} {'std':>8} | "
          f"{'best gap':>10} {'mean gap':>10}")
    for r in q["rows"]:
        print(f"{r['p']:3d} {r['shots']:7d} | {r['best_cost']:10.4f} {r['mean_cost']:10.4f} "
              f"{r['std_cost']:8.4f} | {r['best_gap_vs_exact']:10.4f} {r['mean_gap_vs_exact']:10.4f}")
    print("=" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trial-subset", type=int, default=8,
                     help="How many N-1 lines to test per (alpha,beta,gamma) point (default 8 of 32).")
    ap.add_argument("--weight-qaoa-restarts", type=int, default=8)
    ap.add_argument("--qaoa-seed-edge", type=int, nargs=2, default=(7, 8))
    ap.add_argument("--p-grid", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--shots-grid", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--qaoa-hyperparam-restarts", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = run(n_trial_subset=args.n_trial_subset,
                  weight_qaoa_restarts=args.weight_qaoa_restarts,
                  qaoa_seed_edge=tuple(args.qaoa_seed_edge),
                  p_grid=tuple(args.p_grid), shots_grid=tuple(args.shots_grid),
                  qaoa_hyperparam_restarts=args.qaoa_hyperparam_restarts,
                  verbose=not args.quiet)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sensitivity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")
    _print_summary(payload)