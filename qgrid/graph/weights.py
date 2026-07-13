"""
Quantum weight encoding: turns physical line properties into the coupling
strengths w_ij used in the CTQW Hamiltonian H = sum_ij w_ij (|i><j| + |j><i|).

    w_ij = alpha * admittance_term + beta * capacity_term + gamma * headroom_term

- admittance_term : 1/|Z_ij| normalized to [0,1] across the graph
                     (physical coupling strength -- stronger lines hop faster)
- capacity_term   : C_ij / C_max  (thicker/higher-rated lines = stronger bridges)
- headroom_term   : 1 - loading_pct_ij (near-saturated lines act as weak links,
                     making the walk *avoid* congestion -- this is what lets the
                     later self-healing QAOA reroute preferentially through
                     under-utilized capacity)

alpha+beta+gamma = 1. Defaults chosen so admittance (topology-physics) dominates
at rest, while headroom dominates dynamically once loading data streams in --
tune via benchmark.calibrate_weights for a specific dataset.
"""
from __future__ import annotations
import numpy as np
import networkx as nx


def encode_quantum_weights(G: nx.Graph, alpha=0.5, beta=0.2, gamma=0.3, require_flow=True):
    if require_flow and not all("loading_pct" in G[u][v] for u, v in G.edges()):
        for u, v in G.edges():
            G[u][v].setdefault("loading_pct", 0.0)

    inv_z = np.array([1.0 / max(G[u][v]["z_mag"], 1e-6) for u, v in G.edges()])
    cap = np.array([G[u][v]["capacity_kva"] for u, v in G.edges()])
    load_pct = np.array([G[u][v]["loading_pct"] for u, v in G.edges()])

    def norm01(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.ones_like(x) * 0.5

    admittance_term = norm01(inv_z)
    capacity_term = norm01(cap)
    headroom_term = 1.0 - load_pct

    w = alpha * admittance_term + beta * capacity_term + gamma * headroom_term
    # inactive edges (failed lines) get zero quantum coupling -> removed from Hamiltonian support
    for (u, v), wij in zip(G.edges(), w):
        active_or_candidate = G[u][v]["status"] in ("active", "open")
        G[u][v]["q_weight"] = float(wij) if active_or_candidate else 0.0
    return G


def calibrate_weights(G: nx.Graph, target_metric_fn, alpha_grid=None, beta_grid=None, gamma_grid=None):
    """
    Small grid search over (alpha, beta, gamma) simplex maximizing target_metric_fn(G)
    (e.g. correlation of CTQW stationary occupation with ground-truth criticality from
    the PowerCascade cascade_events labels, once available). Returns best (a,b,g), score.
    """
    best = (None, -np.inf)
    grid = np.linspace(0, 1, 6)
    for a in grid:
        for b in grid:
            g = 1 - a - b
            if g < 0 or g > 1:
                continue
            encode_quantum_weights(G, alpha=a, beta=b, gamma=g)
            score = target_metric_fn(G)
            if score > best[1]:
                best = ((a, b, g), score)
    return best
