"""
Structural vulnerability analysis: articulation points (cut vertices) and
bridges (cut edges) on the *active* subgraph. These are the buses/lines whose
loss disconnects part of the grid -- prime candidates both for cascade seeding
(worst-case contingency selection) and for where self-healing reroute (QAOA)
needs to find an alternative path via tie switches.
"""
from __future__ import annotations
import networkx as nx


def active_subgraph(G: nx.Graph):
    active_nodes = [n for n, d in G.nodes(data=True) if d["status"] == "active"]
    active_edges = [(u, v) for u, v, d in G.edges(data=True) if d["status"] == "active"]
    return G.subgraph(active_nodes).edge_subgraph(active_edges).copy() if active_edges else nx.Graph()


def find_articulation_points(G: nx.Graph):
    sub = active_subgraph(G)
    return sorted(nx.articulation_points(sub)) if sub.number_of_nodes() > 0 else []


def find_bridges(G: nx.Graph):
    sub = active_subgraph(G)
    return sorted(nx.bridges(sub)) if sub.number_of_nodes() > 0 else []


def criticality_ranking(G: nx.Graph, weight_key="q_weight"):
    """
    Ranks buses by a composite criticality score: articulation-point flag +
    betweenness centrality (on active, weighted subgraph) + downstream load served.
    Used to (a) choose realistic worst-case cascade seeds, (b) sanity-check that
    the CTQW's stationary occupation correlates with true structural criticality
    (a key benchmark claim for the paper).
    """
    sub = active_subgraph(G)
    if sub.number_of_nodes() == 0:
        return []
    ap = set(nx.articulation_points(sub))
    # distance-like weight for betweenness: inverse of quantum weight (stronger coupling = "shorter")
    for u, v, d in sub.edges(data=True):
        d["_dist"] = 1.0 / max(d.get(weight_key, 1e-6), 1e-6)
    bc = nx.betweenness_centrality(sub, weight="_dist", normalized=True)
    downstream_load = {n: sub.nodes[n].get("p_kw", 0.0) for n in sub.nodes()}

    ranking = []
    for n in sub.nodes():
        score = (2.0 if n in ap else 0.0) + bc.get(n, 0.0) + 1e-4 * downstream_load[n]
        ranking.append({"bus": n, "is_articulation": n in ap, "betweenness": bc.get(n, 0.0),
                         "load_kw": downstream_load[n], "criticality_score": score})
    return sorted(ranking, key=lambda r: -r["criticality_score"])
