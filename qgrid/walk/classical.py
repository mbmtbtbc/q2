"""
Classical continuous-time random walk (CTRW) baseline on the same weighted graph:
dP/dt = -L P, generator L = weighted graph Laplacian, P(t) = exp(-Lt) P(0).
This is the classical analogue used to benchmark the quantum walk against
(same graph, same weights, same initial delta distribution) -- the standard
comparison in CTQW literature (ballistic vs diffusive spreading, hitting times).
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.linalg import expm


def build_generator(G: nx.Graph, weight_key="q_weight"):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    W = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        w = d.get(weight_key, 1.0)
        W[idx[u], idx[v]] = w
        W[idx[v], idx[u]] = w
    L = np.diag(W.sum(axis=1)) - W
    return L, nodes, idx


class ClassicalContinuousRandomWalk:
    def __init__(self, G: nx.Graph, weight_key="q_weight"):
        self.G = G
        self.L, self.nodes, self.idx = build_generator(G, weight_key)
        self.N = len(self.nodes)

    def initial_state(self, source_node):
        p0 = np.zeros(self.N)
        p0[self.idx[source_node]] = 1.0
        return p0

    def evolve_series(self, source_node, times):
        p0 = self.initial_state(source_node)
        probs = np.zeros((len(times), self.N))
        for k, t in enumerate(times):
            probs[k] = expm(-self.L * t) @ p0
        return probs

    def stationary_distribution(self):
        # for connected graph: proportional to weighted degree
        deg = np.diag(self.L) + 0  # since L = D - W, diag(L) = weighted degree
        return deg / deg.sum()
