"""
Validates `graph.build.approximate_power_flow` (a radial downstream-demand-
aggregation heuristic, documented in build.py as *not* a real load-flow solver)
against an actual AC power flow solved by pandapower, on the identical topology
and loads. This is the check flagged in the README's "known simplifications"
section: reviewers in a power-systems-adjacent venue will ask for it.

Only imported when called -- pandapower is an optional/heavier dependency not
listed in the base requirements.txt, so the import is guarded and the rest of
the pipeline works with or without it installed.

Usage
-----
    from qgrid.graph.powerflow_validation import validate_against_pandapower
    report = validate_against_pandapower(G0)   # G0 = build_graph(...) output,
                                                #      already run through
                                                #      approximate_power_flow
"""
from __future__ import annotations
import numpy as np
import networkx as nx

try:
    import pandapower as pp
    _HAS_PANDAPOWER = True
except ImportError:
    _HAS_PANDAPOWER = False


def _require_pandapower():
    if not _HAS_PANDAPOWER:
        raise ImportError(
            "pandapower is required for power-flow validation but is not installed. "
            "Install it with: pip install pandapower --break-system-packages"
        )


def build_pandapower_network(G: nx.Graph, vn_kv: float = 12.66):
    """
    Builds a pandapower network from the *active* subgraph of G, reusing G's own
    node ids as pandapower bus indices so results can be mapped straight back.
    Only radial/meshed-but-connected active topology reachable from the slack bus
    is included; buses outside that component are reported as unsupplied rather
    than silently dropped.
    """
    _require_pandapower()
    slack = G.graph["slack_bus"]

    active_nodes = [n for n, d in G.nodes(data=True) if d["status"] == "active"]
    active_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("status") == "active"]
    sub = nx.Graph()
    sub.add_nodes_from((n, G.nodes[n]) for n in active_nodes)
    sub.add_edges_from((u, v, G[u][v]) for u, v in active_edges if u in active_nodes and v in active_nodes)

    if slack not in sub:
        raise ValueError(f"slack bus {slack} is not in the active subgraph -- nothing to validate")
    reachable = nx.node_connected_component(sub, slack)
    unsupplied = sorted(set(active_nodes) - reachable)
    sub = sub.subgraph(reachable).copy()

    net = pp.create_empty_network()
    for n in sub.nodes():
        pp.create_bus(net, vn_kv=vn_kv, index=n, name=str(n))
    pp.create_ext_grid(net, bus=slack, vm_pu=1.0)

    line_id_to_edge = {}
    for u, v, d in sub.edges(data=True):
        capacity_kva = max(d.get("capacity_kva", 1.0), 1e-6)
        max_i_ka = capacity_kva / (np.sqrt(3) * vn_kv * 1000.0)
        lid = pp.create_line_from_parameters(
            net, from_bus=u, to_bus=v, length_km=1.0,
            r_ohm_per_km=d["r"], x_ohm_per_km=d["x"], c_nf_per_km=0.0,
            max_i_ka=max_i_ka, name=f"{u}-{v}",
        )
        line_id_to_edge[lid] = (u, v)

    for n in sub.nodes():
        p_kw = sub.nodes[n].get("p_kw", 0.0)
        q_kvar = sub.nodes[n].get("q_kvar", 0.0)
        if n != slack and (p_kw or q_kvar):
            pp.create_load(net, bus=n, p_mw=p_kw / 1000.0, q_mvar=q_kvar / 1000.0, name=str(n))

    return net, line_id_to_edge, unsupplied


def validate_against_pandapower(G: nx.Graph, vn_kv: float = 12.66):
    """
    Runs a real AC power flow (pandapower Newton-Raphson) on G's active topology
    and compares, per active line, the apparent power flow and loading% against
    what `approximate_power_flow` had already stored on G. G is expected to have
    already been run through approximate_power_flow (as the pipeline always does)
    so `flow_kva`/`loading_pct` are present for comparison.

    Returns per-line comparisons plus aggregate error metrics (MAE, RMSE, max
    abs error in loading percentage points) and Pearson/Spearman correlation
    between the approximate and AC-exact loading percentages.
    """
    _require_pandapower()
    from scipy import stats

    net, line_id_to_edge, unsupplied = build_pandapower_network(G, vn_kv=vn_kv)
    pp.runpp(net, algorithm="nr")

    rows = []
    for lid, (u, v) in line_id_to_edge.items():
        row = net.res_line.loc[lid]
        ac_s_kva = 1000.0 * float(np.hypot(row["p_from_mw"], row["q_from_mvar"]))
        capacity_kva = max(G[u][v].get("capacity_kva", 1.0), 1e-6)
        ac_loading_pct = 100.0 * ac_s_kva / capacity_kva

        approx_s_kva = float(G[u][v].get("flow_kva", 0.0))
        approx_loading_pct = 100.0 * float(G[u][v].get("loading_pct", 0.0))

        rows.append({
            "u": u, "v": v,
            "approx_flow_kva": approx_s_kva, "ac_flow_kva": ac_s_kva,
            "approx_loading_pct": approx_loading_pct, "ac_loading_pct": ac_loading_pct,
            "abs_error_pct_points": abs(approx_loading_pct - ac_loading_pct),
            "ac_line_loading_percent_pandapower": float(row["loading_percent"]),
        })

    def _corr(a, b):
        if np.std(a) > 0 and np.std(b) > 0:
            r, p_r = stats.pearsonr(a, b)
            rho, p_rho = stats.spearmanr(a, b)
            return float(r), float(p_r), float(rho), float(p_rho)
        return None, None, None, None

    if rows:
        approx_load_pct = np.array([r["approx_loading_pct"] for r in rows])
        ac_load_pct = np.array([r["ac_loading_pct"] for r in rows])
        approx_kva = np.array([r["approx_flow_kva"] for r in rows])
        ac_kva = np.array([r["ac_flow_kva"] for r in rows])

        # Loading%-based error: only physically meaningful if capacity_kva reflects
        # real thermal ratings. Flag rather than silently trust it.
        load_err = np.abs(approx_load_pct - ac_load_pct)
        mae_load = float(load_err.mean())
        rmse_load = float(np.sqrt((load_err ** 2).mean()))
        max_err_load = float(load_err.max())
        pearson_r_load, pearson_p_load, spearman_rho_load, spearman_p_load = _corr(approx_load_pct, ac_load_pct)

        # Flow-magnitude error (kVA): capacity-independent, the real test of the
        # approximate_power_flow algorithm itself (downstream-demand aggregation
        # vs actual AC |S| flow).
        flow_err_kva = np.abs(approx_kva - ac_kva)
        mae_flow_kva = float(flow_err_kva.mean())
        rmse_flow_kva = float(np.sqrt((flow_err_kva ** 2).mean()))
        max_err_flow_kva = float(flow_err_kva.max())
        nz = ac_kva > 1e-9
        mape_flow_pct = float((flow_err_kva[nz] / ac_kva[nz]).mean() * 100.0) if nz.any() else None
        pearson_r_flow, pearson_p_flow, spearman_rho_flow, spearman_p_flow = _corr(approx_kva, ac_kva)

        # Sanity flag: if the graph's stored capacity_kva is far smaller than the
        # actual AC flow on most lines, "capacity_kva" is not a physical kVA rating
        # in this dataset and loading%-based error above should not be trusted/reported
        # as a physical-engineering accuracy claim.
        capacities = np.array([max(G[u][v].get("capacity_kva", 1.0), 1e-6) for (u, v) in
                                [(r["u"], r["v"]) for r in rows]])
        frac_over_capacity = float(np.mean(ac_kva > capacities))
        capacity_looks_synthetic = frac_over_capacity > 0.5

        max_bus_vm_pu_deviation = float(np.abs(net.res_bus["vm_pu"] - 1.0).max())
        converged = bool(net["converged"])
    else:
        mae_load = rmse_load = max_err_load = None
        pearson_r_load = pearson_p_load = spearman_rho_load = spearman_p_load = None
        mae_flow_kva = rmse_flow_kva = max_err_flow_kva = mape_flow_pct = None
        pearson_r_flow = pearson_p_flow = spearman_rho_flow = spearman_p_flow = None
        capacity_looks_synthetic = None
        max_bus_vm_pu_deviation = None
        converged = False

    return {
        "n_lines_compared": len(rows),
        "unsupplied_buses": unsupplied,
        "converged": converged,
        "per_line": rows,
        "capacity_kva_looks_synthetic": capacity_looks_synthetic,
        "capacity_kva_warning": (
            "capacity_kva on this graph is far smaller than actual AC power flow on "
            "most lines -- it is not a physical kVA thermal rating in this dataset, "
            "so loading%-based error below is not a meaningful physical-accuracy claim. "
            "Use aggregate_error.flow_kva_* instead, or re-run validation on a graph "
            "loaded with physically-grounded capacities (e.g. source='ieee33')."
        ) if capacity_looks_synthetic else None,
        "aggregate_error": {
            "loading_pct_mae_points": mae_load,
            "loading_pct_rmse_points": rmse_load,
            "loading_pct_max_abs_error_points": max_err_load,
            "loading_pct_pearson_r": pearson_r_load, "loading_pct_pearson_p": pearson_p_load,
            "loading_pct_spearman_rho": spearman_rho_load, "loading_pct_spearman_p": spearman_p_load,
            "flow_kva_mae": mae_flow_kva,
            "flow_kva_rmse": rmse_flow_kva,
            "flow_kva_max_abs_error": max_err_flow_kva,
            "flow_kva_mean_abs_pct_error": mape_flow_pct,
            "flow_kva_pearson_r": pearson_r_flow, "flow_kva_pearson_p": pearson_p_flow,
            "flow_kva_spearman_rho": spearman_rho_flow, "flow_kva_spearman_p": spearman_p_flow,
        },
        "max_bus_voltage_deviation_pu": max_bus_vm_pu_deviation,
    }