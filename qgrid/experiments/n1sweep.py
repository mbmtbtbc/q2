"""
N-1 statistical sweep.

Addresses: "currently the whole result is N=1" -- runs the CTQW-vs-CTRW
benchmark + cascading-failure + QAOA self-healing pipeline once per line in
the N-1 contingency set (every currently-active, non-tie line), instead of
just the single hardcoded seed_edge=(7, 8) used in run_pipeline.py's __main__
block, and reports mean +/- std and median +/- IQR across trials for the key
metrics (IPR, TVD, hitting time, buses recovered, ...).

>>> IMPORTANT NOTE ON TRIAL COUNT (35 vs 32) <<<
The request asked for "every line as an N-1 seed (35 lines -> 35 trials)".
The graph has 35 edges total, but 3 of them are tie switches that are
normally OPEN (no load flow). "Tripping" an already-open switch is a no-op,
not a contingency -- it produces a trivially-identical trial. This script
therefore sweeps the 32 physically-active, non-tie lines instead. Use
--include-tie-switches to add the 3 tie-switch edges back in anyway (they
will show up as degenerate zero-impact trials, included separately in the
output for transparency rather than silently dropped).

>>> IMPORTANT NOTE ON WHAT VARIES <<<
run_pipeline.py computes the CTQW-vs-classical benchmark ONCE on the
fault-free graph -- it does not depend on the seed edge, so sweeping it as-is
would give std=0 by construction. This script instead evaluates CTQW vs CTRW
on the POST-CASCADE graph for each trial (see qgrid/experiments/common.py
docstring), so IPR/TVD/hitting-time genuinely vary with the contingency.
Lines directly incident to the slack bus (here: line (1,2)) isolate the
walk's source node entirely when tripped, which is a legitimate but
degenerate case (IPR=1, TVD=0 by construction, nothing to heal) -- these are
included in the aggregate statistics as real data points, not filtered out,
but are called out in the printed summary so they aren't mistaken for a bug.

Run:
    python3 -m qgrid.experiments.n1_sweep
    python3 -m qgrid.experiments.n1_sweep --qaoa-restarts 20 --include-tie-switches

Writes qgrid/experiments/results/n1_sweep.json
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np

from qgrid.experiments.common import (
    build_base_graph, active_non_tie_lines, run_trial, summarize, npify, RESULTS_DIR,
)


def run(qaoa_restarts=15, qaoa_shots=1024, qaoa_maxiter=40, qaoa_p=2,
        include_tie_switches=False, source_bus=1, target_bus=18, verbose=True):
    G0, meta = build_base_graph()
    lines = active_non_tie_lines(G0)
    tie_lines = [(u, v) for u, v, d in G0.edges(data=True) if d.get("is_tie_switch")]

    trial_edges = list(lines)
    if include_tie_switches:
        trial_edges += tie_lines

    trials = []
    for i, edge in enumerate(trial_edges):
        if verbose:
            print(f"  [{i+1}/{len(trial_edges)}] seed_edge={edge}", flush=True)
        res = run_trial(edge, G0=G0, source_bus=source_bus, target_bus=target_bus,
                         qaoa_p=qaoa_p, qaoa_shots=qaoa_shots, qaoa_maxiter=qaoa_maxiter,
                         qaoa_restarts=qaoa_restarts)
        res["is_tie_switch_seed"] = edge in tie_lines
        trials.append(res)

    # Metrics aggregated over the 32 physically-meaningful (non-tie) trials only.
    core_trials = [t for t in trials if not t["is_tie_switch_seed"]]

    def col(key_path):
        out = []
        for t in core_trials:
            v = t
            for k in key_path:
                v = v[k]
            out.append(v)
        return out

    summary = {
        "n_trials": len(core_trials),
        "n_slack_incident_degenerate_trials": sum(
            1 for t in core_trials if t["benchmark"]["final_tvd_quantum_vs_classical"] == 0.0
            and t["benchmark"]["quantum_ipr_final"] == 1.0
        ),
        "quantum_ipr_final": summarize(col(["benchmark", "quantum_ipr_final"])),
        "classical_ipr_final": summarize(col(["benchmark", "classical_ipr_final"])),
        "final_tvd_quantum_vs_classical": summarize(col(["benchmark", "final_tvd_quantum_vs_classical"])),
        "quantum_hitting_time": summarize(col(["benchmark", "quantum_hitting_time"])),
        "classical_hitting_time": summarize(col(["benchmark", "classical_hitting_time"])),
        "buses_de_energized": summarize(col(["impact_summary", "buses_de_energized"])),
        "pct_buses_lost": summarize(col(["impact_summary", "pct_buses_lost"])),
        "load_lost_kw": summarize(col(["impact_summary", "load_lost_kw"])),
        "qaoa_recovered_best": summarize(col(["qaoa_recovered_best"])),
        "exact_recovered": summarize(col(["exact_recovered"])),
        "qaoa_cost_std_within_trial": summarize(col(["qaoa", "std_cost"])),
        "qaoa_gap_vs_exact": summarize([t["reroute_baselines"]["qaoa"]["gap"] for t in core_trials
                                          if "qaoa" in t["reroute_baselines"]]),
    }

    payload = {
        "meta": {"grid_name": meta["name"], "n_buses": G0.number_of_nodes(),
                  "n_lines_total": G0.number_of_edges(), "n_core_trials": len(core_trials),
                  "n_tie_switch_trials": len(trials) - len(core_trials),
                  "qaoa_restarts": qaoa_restarts, "qaoa_shots": qaoa_shots,
                  "qaoa_p": qaoa_p, "qaoa_maxiter": qaoa_maxiter,
                  "source_bus": source_bus, "target_bus": target_bus},
        "summary": summary,
        "trials": trials,
    }
    return npify(payload)


def _print_summary(payload):
    s = payload["summary"]
    m = payload["meta"]
    print("=" * 88)
    print(f"N-1 sweep: {m['n_core_trials']} core trials "
          f"(+{m['n_tie_switch_trials']} tie-switch trials)  "
          f"QAOA: p={m['qaoa_p']}, shots={m['qaoa_shots']}, restarts={m['qaoa_restarts']}")
    print(f"({s['n_slack_incident_degenerate_trials']} trial(s) are slack-incident-line "
          f"degenerate cases: IPR=1, TVD=0 by construction -- see script docstring)")
    print("-" * 88)

    def _row(label, d, unit=""):
        if d["n_valid"] == 0:
            print(f"  {label:34s}  no valid samples")
            return
        excl = f"  ({d['n_excluded']} excluded/non-finite)" if d["n_excluded"] else ""
        print(f"  {label:34s}  mean={d['mean']:.4f}{unit} +/- {d['std']:.4f}   "
              f"median={d['median']:.4f}{unit}  IQR=[{d['iqr_low']:.4f}, {d['iqr_high']:.4f}]"
              f"  (n={d['n_valid']}/{d['n_total']}){excl}")

    _row("Quantum IPR (final)", s["quantum_ipr_final"])
    _row("Classical IPR (final)", s["classical_ipr_final"])
    _row("TVD (quantum vs classical)", s["final_tvd_quantum_vs_classical"])
    _row("Quantum hitting time", s["quantum_hitting_time"])
    _row("Classical hitting time", s["classical_hitting_time"])
    _row("Buses de-energized", s["buses_de_energized"])
    _row("% buses lost", s["pct_buses_lost"], "%")
    _row("Load lost", s["load_lost_kw"], " kW")
    _row("QAOA buses recovered (best of restarts)", s["qaoa_recovered_best"])
    _row("Exact-optimum buses recovered", s["exact_recovered"])
    _row("QAOA cost std within a trial (across restarts)", s["qaoa_cost_std_within_trial"])
    _row("QAOA optimality gap vs exact", s["qaoa_gap_vs_exact"])
    print("=" * 88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qaoa-restarts", type=int, default=15)
    ap.add_argument("--qaoa-shots", type=int, default=1024)
    ap.add_argument("--qaoa-maxiter", type=int, default=40)
    ap.add_argument("--qaoa-p", type=int, default=2)
    ap.add_argument("--include-tie-switches", action="store_true")
    ap.add_argument("--source-bus", type=int, default=1)
    ap.add_argument("--target-bus", type=int, default=18)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    payload = run(qaoa_restarts=args.qaoa_restarts, qaoa_shots=args.qaoa_shots,
                  qaoa_maxiter=args.qaoa_maxiter, qaoa_p=args.qaoa_p,
                  include_tie_switches=args.include_tie_switches,
                  source_bus=args.source_bus, target_bus=args.target_bus,
                  verbose=not args.quiet)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "n1_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")
    _print_summary(payload)