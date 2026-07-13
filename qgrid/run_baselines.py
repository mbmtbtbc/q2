"""
Runs the classical-baseline comparisons requested for the paper's results
section, on top of the existing pipeline (nothing in run_pipeline.py, app.py,
or any walk/failures/reroute module is modified):

  1. Criticality/vulnerability: degree/closeness/PageRank/N-1 severity baselines
     vs. CTQW occupation and vs. the existing composite criticality_ranking,
     with Spearman + Pearson correlation (and p-values).
  2. Self-healing/QAOA: exact brute-force optimum, greedy heuristic, simulated
     annealing, and random baseline on the same post-cascade reroute QUBO QAOA
     solved, reported as absolute cost + optimality gap vs. the exact solution.
  3. Power-flow approximation: AC power flow (pandapower Newton-Raphson) vs.
     approximate_power_flow's loading%, on the baseline (pre-cascade) topology.

Run:  python3 -m qgrid.run_baselines
Writes qgrid/baseline_results.json. Does not touch qgrid_data.json or the dashboard.
"""
from __future__ import annotations
import os
import json
import numpy as np
import networkx as nx

from qgrid.data.loaders import load_topology
from qgrid.graph.build import build_graph, approximate_power_flow
from qgrid.graph.weights import encode_quantum_weights
from qgrid.walk.ctqw import ContinuousTimeQuantumWalk
from qgrid.failures.topology import criticality_ranking
from qgrid.failures.cascade import simulate_cascade
from qgrid.reroute.qaoa_reroute import build_qubo, solve_qaoa_reroute

from qgrid.benchmark.classical_baselines import benchmark_criticality_baselines
from qgrid.reroute.classical_reroute import compare_reroute_methods
from qgrid.graph.powerflow_validation import validate_against_pandapower, _HAS_PANDAPOWER


def npify(o):
    if isinstance(o, dict):
        return {str(k): npify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [npify(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (np.isinf(o) or np.isnan(o)):
        return None
    return o


def run(source="powercascade", topology_csv=None, load_csv=None,
        source_bus=1, seed_edge=(7, 8), occupation_time=6.0,
        qaoa_p=2, qaoa_shots=1024, qaoa_maxiter=40, qaoa_seed=42):

    topo, load, meta = load_topology(source, topology_csv=topology_csv, load_csv=load_csv)
    G0 = build_graph(topo, load, meta)
    G0 = approximate_power_flow(G0)
    G0 = encode_quantum_weights(G0)
    slack = G0.graph["slack_bus"]

    # ---------- 1. criticality baselines vs CTQW occupation ----------
    qw = ContinuousTimeQuantumWalk(G0)
    occ_at_t = qw.occupation_at(source_bus, t=occupation_time)
    ranking = criticality_ranking(G0)
    criticality_report = benchmark_criticality_baselines(
        G0, ctqw_occupation=occ_at_t, existing_criticality_ranking=ranking,
    )

    # ---------- 2. reroute baselines vs QAOA (same scenario as run_pipeline.py) ----------
    Gf, snaps, log = simulate_cascade(G0, seed_edges=[seed_edge], overload_threshold=1.0)
    qw_post = ContinuousTimeQuantumWalk(Gf)
    occ_post = qw_post.occupation_at(slack, t=3.0)
    tie_candidates = [(u, v) for u, v, d in Gf.edges(data=True) if d.get("is_tie_switch")]
    h_qubo, J_qubo, cands = build_qubo(Gf, tie_candidates, occ_post)
    qaoa_decision, qaoa_info = solve_qaoa_reroute(
        h_qubo, J_qubo, p=qaoa_p, shots=qaoa_shots, maxiter=qaoa_maxiter, seed=qaoa_seed,
    )
    reroute_report = compare_reroute_methods(
        h_qubo, J_qubo, cands, qaoa_decision=qaoa_decision,
        qaoa_cost=float(qaoa_info["optimal_cost"]),
    )

    # ---------- 3. power-flow validation (baseline topology) ----------
    # Run once on G0 as-loaded (source may carry a non-physical "capacity" column,
    # e.g. the powercascade-augmented CSV -- see capacity_kva_warning in the result),
    # and once on the built-in ieee33 topology whose capacity_kva is a stated,
    # physically-grounded thermal rating (1200 kVA/line), so loading%-based error
    # has at least one trustworthy reference point regardless of which source is used.
    if _HAS_PANDAPOWER:
        try:
            powerflow_report = validate_against_pandapower(G0)
        except Exception as e:  # keep the rest of the report usable even if this fails
            powerflow_report = {"error": str(e)}
        try:
            topo_b, load_b, meta_b = load_topology("ieee33")
            G0_builtin = encode_quantum_weights(approximate_power_flow(build_graph(topo_b, load_b, meta_b)))
            powerflow_report_builtin_ieee33 = validate_against_pandapower(G0_builtin)
        except Exception as e:
            powerflow_report_builtin_ieee33 = {"error": str(e)}
    else:
        powerflow_report = {"error": "pandapower not installed; "
                                      "pip install pandapower --break-system-packages"}
        powerflow_report_builtin_ieee33 = powerflow_report

    payload = {
        "meta": {"grid_name": meta["name"], "n_buses": G0.number_of_nodes(),
                  "n_lines": G0.number_of_edges(), "slack_bus": slack,
                  "source_bus": source_bus, "seed_edge": list(seed_edge)},
        "criticality_baselines": criticality_report,
        "reroute_baselines": reroute_report,
        "powerflow_validation": powerflow_report,
        "powerflow_validation_builtin_ieee33_capacities": powerflow_report_builtin_ieee33,
    }
    return npify(payload)


def _print_summary(payload):
    print("=" * 78)
    print(f"Grid: {payload['meta']['grid_name']}  "
          f"({payload['meta']['n_buses']} buses, {payload['meta']['n_lines']} lines)")

    print("\n--- Criticality baselines vs CTQW occupation (Spearman rho [p], Pearson r [p]) ---")
    for name, corr in payload["criticality_baselines"]["correlations"].items():
        vs_q = corr.get("vs_ctqw_occupation")
        if vs_q is None:
            print(f"  {name:32s}  n/a (degenerate or insufficient overlap)")
        else:
            print(f"  {name:32s}  rho={vs_q['spearman_rho']:+.3f} [p={vs_q['spearman_p']:.3g}]   "
                  f"r={vs_q['pearson_r']:+.3f} [p={vs_q['pearson_p']:.3g}]   n={vs_q['n']}")

    print("\n--- Reroute QUBO: method costs and optimality gap vs exact ---")
    rr = payload["reroute_baselines"]
    print(f"  candidates: {rr['n_candidates']}  tie switches: {rr['tie_candidates']}")
    for method in ["exact", "greedy", "simulated_annealing", "random_baseline", "qaoa"]:
        if method not in rr:
            continue
        m = rr[method]
        cost = m.get("cost", m.get("best_of_n_cost"))
        gap = m.get("gap", m.get("best_of_n_gap"))
        print(f"  {method:20s}  cost={cost:+.4f}  gap={gap:.4f}  closed={m['closed_switches']}")

    def _fmt(x, spec=".3f"):
        return format(x, spec) if x is not None else "n/a"

    def _print_pf_block(label, pf):
        print(f"\n--- Power-flow validation: {label} ---")
        if "error" in pf:
            print(f"  skipped: {pf['error']}")
            return
        agg = pf["aggregate_error"]
        print(f"  lines compared: {pf['n_lines_compared']}  converged: {pf['converged']}  "
              f"unsupplied buses: {pf['unsupplied_buses']}")
        print(f"  flow_kva  MAE={_fmt(agg['flow_kva_mae'])}  RMSE={_fmt(agg['flow_kva_rmse'])}  "
              f"MAPE={_fmt(agg['flow_kva_mean_abs_pct_error'])}%  "
              f"r={_fmt(agg['flow_kva_pearson_r'])} [p={_fmt(agg['flow_kva_pearson_p'], '.3g')}]")
        if pf.get("capacity_kva_warning"):
            print(f"  NOTE: {pf['capacity_kva_warning']}")
        else:
            print(f"  loading%  MAE={_fmt(agg['loading_pct_mae_points'])} pts  "
                  f"RMSE={_fmt(agg['loading_pct_rmse_points'])} pts  "
                  f"max={_fmt(agg['loading_pct_max_abs_error_points'])} pts")

    _print_pf_block("as-loaded (source's own capacity_kva)", payload["powerflow_validation"])
    _print_pf_block("built-in ieee33 physically-grounded capacities",
                     payload["powerflow_validation_builtin_ieee33_capacities"])
    print("=" * 78)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    topology_csv = os.path.join(project_root, "ieee33_augmented.csv")
    load_csv = os.path.join(project_root, "ieee33_load.csv")

    payload = run(source="powercascade", topology_csv=topology_csv, load_csv=load_csv)

    out_path = os.path.join(project_root, "baseline_results.json")
    payload_json = json.dumps(payload, indent=2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload_json)
    print(f"wrote {out_path}, {len(payload_json)} bytes\n")

    _print_summary(payload)