"""
Builds a networkx graph from the common topology/load schema. This is the
only module that touches raw dataframes -- everything downstream (walk,
failures, reroute) only ever sees a networkx.Graph, so scaling from IEEE-33
to IEEE-123 (or any other feeder) requires zero changes past this point.
"""
from __future__ import annotations
import networkx as nx
import numpy as np


def build_graph(topo_df, load_df, meta) -> nx.Graph:
    G = nx.Graph(name=meta.get("name", "grid"))
    for _, row in topo_df.iterrows():
        z = complex(row.r_ohm, row.x_ohm)
        is_tie = bool(row.is_tie_switch)
        G.add_edge(
            int(row.from_bus), int(row.to_bus),
            r=row.r_ohm, x=row.x_ohm, z_mag=abs(z),
            capacity_kva=row.capacity_kva,
            is_tie_switch=is_tie,
            status="open" if is_tie else "active",  # tie switches are normally OPEN
            flow_kva=0.0,
        )
    loads = load_df.groupby("bus")[["p_kw", "q_kvar"]].mean().to_dict("index")
    for n in G.nodes:
        ld = loads.get(n, {"p_kw": 0.0, "q_kvar": 0.0})
        G.nodes[n]["p_kw"] = ld["p_kw"]
        G.nodes[n]["q_kvar"] = ld["q_kvar"]
        G.nodes[n]["is_slack"] = (n == meta["slack_bus"])
        G.nodes[n]["status"] = "active"
    G.graph["slack_bus"] = meta["slack_bus"]
    return G


def approximate_power_flow(G: nx.Graph):
    """
    Lightweight radial/near-radial load-flow approximation (not a full Newton-Raphson
    solver -- sufficient for capacity-relative loading% used in weight encoding &
    cascade triggers). For each edge, flow_kva ~= downstream aggregated demand seen
    from the slack via BFS trees on the *active* subgraph. Works per connected component.
    """
    slack = G.graph["slack_bus"]
    active_nodes = [n for n, d in G.nodes(data=True) if d["status"] == "active"]
    active_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("status") == "active"]
    # Build subgraph with only active nodes and active edges
    sub = nx.Graph()
    sub.add_nodes_from((n, G.nodes[n]) for n in active_nodes)
    sub.add_edges_from((u, v, G[u][v]) for u, v in active_edges if u in active_nodes and v in active_nodes)
    for comp in nx.connected_components(sub):
        comp_sub = sub.subgraph(comp)
        root = slack if slack in comp else next(iter(comp))
        if not nx.is_connected(comp_sub):
            continue
        tree = nx.bfs_tree(comp_sub, root)
        # downstream demand aggregation (leaves -> root)
        order = list(nx.dfs_postorder_nodes(tree, root))
        downstream_s = {n: np.hypot(G.nodes[n]["p_kw"], G.nodes[n]["q_kvar"]) for n in comp}
        for n in order:
            for child in tree.successors(n):
                downstream_s[n] += downstream_s[child]
        for u, v in tree.edges():
            s_kva = downstream_s[v]
            if G.has_edge(u, v):
                G[u][v]["flow_kva"] = s_kva
                G[u][v]["loading_pct"] = min(1.0, s_kva / max(G[u][v]["capacity_kva"], 1e-6))
    # inactive/removed edges: no flow
    for u, v in G.edges():
        if G.nodes[u]["status"] != "active" or G.nodes[v]["status"] != "active":
            G[u][v]["flow_kva"] = 0.0
            G[u][v]["loading_pct"] = 0.0
        elif "loading_pct" not in G[u][v]:
            G[u][v]["loading_pct"] = 0.0
    return G
