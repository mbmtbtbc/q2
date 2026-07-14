"""
Cascading failure engine: N-1 (or N-k) contingency seeding + overload-triggered
propagation, the classic cascading-failure model used in power-systems literature
(remove line -> redistribute flow -> overloaded lines trip -> repeat).
"""
from __future__ import annotations
import networkx as nx
import copy
from qgrid.graph.build import approximate_power_flow
from qgrid.graph.weights import encode_quantum_weights


def trip_line(G, u, v):
    G[u][v]["status"] = "failed"


def trip_node(G, n):
    G.nodes[n]["status"] = "failed"
    for nbr in list(G.neighbors(n)):
        G[n][nbr]["status"] = "failed"


def simulate_cascade(G: nx.Graph, seed_edges=None, seed_nodes=None,
                      overload_threshold=1.0, max_rounds=20, reencode_weights=True,
                      alpha=None, beta=None, gamma=None):
    """
    Returns a list of "snapshots" (one per round) each = deep-copied graph state,
    plus a trip log [(round, kind, element)] for the animation/dashboard.

    If `reencode_weights` is True, the optional alpha/beta/gamma weights are used
    so cascade-weight updates preserve the same quantum-weight mix as the input
    graph. If False, failed edges still get q_weight=0 but existing weights are
    otherwise preserved.
    """
    H = copy.deepcopy(G)
    log = []
    snapshots = []

    for u, v in (seed_edges or []):
        trip_line(H, u, v)
        log.append((0, "line", (u, v)))
    for n in (seed_nodes or []):
        trip_node(H, n)
        log.append((0, "node", n))

    H = approximate_power_flow(H)
    if reencode_weights:
        H = encode_quantum_weights(
            H,
            alpha=0.5 if alpha is None else alpha,
            beta=0.2 if beta is None else beta,
            gamma=0.3 if gamma is None else gamma,
        )
    else:
        for u, v, d in H.edges(data=True):
            if d["status"] == "failed":
                d["q_weight"] = 0.0
    snapshots.append(copy.deepcopy(H))

    for rnd in range(1, max_rounds + 1):
        overloaded = [(u, v) for u, v, d in H.edges(data=True)
                      if d["status"] == "active" and d.get("loading_pct", 0) > overload_threshold]
        if not overloaded:
            break
        for u, v in overloaded:
            trip_line(H, u, v)
            log.append((rnd, "line", (u, v)))
        H = approximate_power_flow(H)
        if reencode_weights:
            H = encode_quantum_weights(
                H,
                alpha=0.5 if alpha is None else alpha,
                beta=0.2 if beta is None else beta,
                gamma=0.3 if gamma is None else gamma,
            )
        else:
            for u, v, d in H.edges(data=True):
                if d["status"] == "failed":
                    d["q_weight"] = 0.0
        snapshots.append(copy.deepcopy(H))

    # mark isolated (de-energized) buses: not connected to slack in active subgraph
    active_edges = [(u, v) for u, v, d in H.edges(data=True) if d["status"] == "active"]
    active_sub = H.edge_subgraph(active_edges).copy() if active_edges else nx.Graph()
    slack = H.graph["slack_bus"]
    energized = nx.node_connected_component(active_sub, slack) if slack in active_sub else {slack}
    for n in H.nodes():
        H.nodes[n]["energized"] = n in energized

    return H, snapshots, log


def cascade_impact_summary(G_final: nx.Graph):
    total = G_final.number_of_nodes()
    de_energized = sum(1 for n, d in G_final.nodes(data=True) if not d.get("energized", True))
    failed_lines = sum(1 for u, v, d in G_final.edges(data=True) if d["status"] == "failed")
    total_lines = G_final.number_of_edges()
    load_lost_kw = sum(d["p_kw"] for n, d in G_final.nodes(data=True) if not d.get("energized", True))
    return {
        "buses_de_energized": de_energized,
        "buses_total": total,
        "pct_buses_lost": 100 * de_energized / total,
        "lines_failed": failed_lines,
        "lines_total": total_lines,
        "load_lost_kw": load_lost_kw,
    }
