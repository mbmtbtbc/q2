"""
Classical baselines for the criticality/vulnerability claim.

`failures.topology.criticality_ranking()` already exists in the pipeline and is
used internally to seed cascade contingencies / sanity-check the CTQW. This module
adds the literature-standard classical measures it should be benchmarked against,
plus a simplified N-1 severity index (single-contingency, power-flow-redistribution
based -- the same spirit as a Line Outage Distribution Factor severity score, without
requiring a full linearized-DC-power-flow LODF matrix), and the correlation
statistics (Spearman + Pearson, with p-values) needed to make a defensible
quantitative claim instead of an eyeballed one.

Nothing here modifies existing modules -- it only imports and composes them.
"""
from __future__ import annotations
import copy
import numpy as np
import networkx as nx
from scipy import stats

from qgrid.failures.topology import active_subgraph
from qgrid.graph.build import approximate_power_flow


# ---------------------------------------------------------------------------
# Classical node-centrality baselines (weighted graph, weight_key="q_weight")
# ---------------------------------------------------------------------------

def degree_centrality_ranking(G: nx.Graph, weight_key: str = "q_weight"):
    """Weighted degree centrality (sum of incident edge weights)."""
    sub = active_subgraph(G)
    if sub.number_of_nodes() == 0:
        return {}
    return dict(sub.degree(weight=weight_key))


def closeness_centrality_ranking(G: nx.Graph, weight_key: str = "q_weight"):
    """
    Closeness centrality on the active subgraph. `q_weight` is a coupling
    *strength* (higher = stronger/closer), so distance = 1/weight, matching the
    convention already used for betweenness in failures.topology.criticality_ranking.
    """
    sub = active_subgraph(G)
    if sub.number_of_nodes() == 0:
        return {}
    sub = sub.copy()
    for u, v, d in sub.edges(data=True):
        d["_dist"] = 1.0 / max(d.get(weight_key, 1e-6), 1e-6)
    return nx.closeness_centrality(sub, distance="_dist")


def pagerank_ranking(G: nx.Graph, weight_key: str = "q_weight", alpha: float = 0.85):
    """PageRank on the active subgraph, using q_weight as edge weight."""
    sub = active_subgraph(G)
    if sub.number_of_nodes() == 0:
        return {}
    return nx.pagerank(sub, alpha=alpha, weight=weight_key)


# ---------------------------------------------------------------------------
# Simplified N-1 severity index (LODF-style): single-line-outage power-flow
# redistribution stress, mapped from lines to buses.
# ---------------------------------------------------------------------------

def n1_line_severity(G: nx.Graph):
    """
    For every currently-active, non-tie line, simulate a single-line (N-1) outage:
    remove it, re-run the pipeline's own approximate_power_flow to redistribute
    downstream demand, and score the outage's severity as:
        - buses_islanded   : # buses disconnected from the slack by this outage alone
        - load_lost_kw     : downstream demand stranded by this outage alone
        - max_post_loading : worst post-outage loading_pct on any surviving line
                              (the redistribution-stress signal an LODF severity
                              index is meant to capture -- how close the single
                              worst remaining corridor gets pushed to its thermal limit)
    Returns a dict keyed by the outaged edge (u, v).
    """
    slack = G.graph["slack_bus"]
    severity = {}
    for u, v, d in G.edges(data=True):
        if d.get("status") != "active" or d.get("is_tie_switch"):
            continue
        H = copy.deepcopy(G)
        H[u][v]["status"] = "failed"
        H = approximate_power_flow(H)

        active_edges = [(a, b) for a, b, dd in H.edges(data=True) if dd["status"] == "active"]
        sub = H.edge_subgraph(active_edges).copy() if active_edges else nx.Graph()
        energized = nx.node_connected_component(sub, slack) if slack in sub else {slack}
        islanded = [n for n in H.nodes() if n not in energized]
        load_lost = sum(H.nodes[n]["p_kw"] for n in islanded)
        surviving_loadings = [dd.get("loading_pct", 0.0) for a, b, dd in H.edges(data=True)
                               if dd["status"] == "active"]
        max_post_loading = max(surviving_loadings) if surviving_loadings else 0.0

        severity[(u, v)] = {
            "buses_islanded": len(islanded),
            "load_lost_kw": float(load_lost),
            "max_post_loading": float(max_post_loading),
        }
    return severity


def n1_bus_severity(G: nx.Graph):
    """
    Maps the per-line N-1 severity index onto buses so it is comparable with
    CTQW occupation / the existing bus-level criticality_ranking: a bus's severity
    is the worst (max) severity among the lines incident to it, using load_lost_kw
    as the primary scalar (buses_islanded and max_post_loading are also returned
    per-bus for completeness).
    """
    line_sev = n1_line_severity(G)
    bus_sev = {}
    for (u, v), s in line_sev.items():
        for n in (u, v):
            cur = bus_sev.get(n)
            if cur is None or s["load_lost_kw"] > cur["load_lost_kw"]:
                bus_sev[n] = dict(s)
    # buses with no incident non-tie active line (shouldn't normally happen) get 0
    for n in G.nodes():
        bus_sev.setdefault(n, {"buses_islanded": 0, "load_lost_kw": 0.0, "max_post_loading": 0.0})
    return bus_sev


# ---------------------------------------------------------------------------
# Correlation of the CTQW occupation ranking against each classical baseline
# ---------------------------------------------------------------------------

def rank_correlation(score_a: dict, score_b: dict):
    """
    Spearman and Pearson correlation (with p-values) between two node->score dicts,
    aligned on their common keys. Returns None if fewer than 3 common nodes or if
    either series is constant (correlation undefined).
    """
    common = sorted(set(score_a) & set(score_b))
    if len(common) < 3:
        return None
    a = np.array([float(score_a[n]) for n in common])
    b = np.array([float(score_b[n]) for n in common])
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    rho, p_spearman = stats.spearmanr(a, b)
    r, p_pearson = stats.pearsonr(a, b)
    return {
        "n": len(common),
        "spearman_rho": float(rho), "spearman_p": float(p_spearman),
        "pearson_r": float(r), "pearson_p": float(p_pearson),
    }


def benchmark_criticality_baselines(G: nx.Graph, ctqw_occupation: dict,
                                     existing_criticality_ranking=None,
                                     weight_key: str = "q_weight"):
    """
    Computes every classical baseline above and correlates each against:
      (a) the CTQW occupation distribution (the quantum-walk criticality signal), and
      (b) the pipeline's existing composite criticality_ranking(), if supplied,
          so the new quantum-vs-classical comparison and the existing internal
          ranking can both be reported in the same table.

    ctqw_occupation: dict node -> probability (e.g. from
        ContinuousTimeQuantumWalk.occupation_at(slack, t=<long-time horizon>),
        or a time-averaged occupation series -- caller's choice of horizon).
    """
    baselines = {
        "degree_centrality": degree_centrality_ranking(G, weight_key),
        "closeness_centrality": closeness_centrality_ranking(G, weight_key),
        "pagerank": pagerank_ranking(G, weight_key),
    }
    bus_sev = n1_bus_severity(G)
    baselines["n1_severity_load_lost_kw"] = {n: s["load_lost_kw"] for n, s in bus_sev.items()}
    baselines["n1_severity_buses_islanded"] = {n: s["buses_islanded"] for n, s in bus_sev.items()}

    existing_score = None
    if existing_criticality_ranking is not None:
        existing_score = {r["bus"]: r["criticality_score"] for r in existing_criticality_ranking}
        baselines["existing_composite_criticality"] = existing_score

    results = {}
    for name, score in baselines.items():
        results[name] = {
            "vs_ctqw_occupation": rank_correlation(score, ctqw_occupation),
        }
        if existing_score is not None and name != "existing_composite_criticality":
            results[name]["vs_existing_criticality_ranking"] = rank_correlation(score, existing_score)

    return {
        "baseline_scores": baselines,
        "n1_line_severity": {f"{u}-{v}": s for (u, v), s in n1_line_severity(G).items()},
        "correlations": results,
    }