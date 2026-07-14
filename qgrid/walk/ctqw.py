"""
Continuous-Time Quantum Walk (CTQW) on the weighted grid graph.

Why not a gate-based Qiskit circuit for the dynamics itself?
--------------------------------------------------------------
CTQW evolution is exact unitary propagation U(t) = exp(-i H t) on an N-dimensional
Hilbert space indexed by grid nodes (H = weighted Laplacian/adjacency). This is the
standard simulation approach in the CTQW literature (Farhi & Gutmann 1998; Childs 2009)
-- it is *not* a gate decomposition problem, it's Hamiltonian simulation, so we use
scipy's sparse matrix exponential-action (expm_multiply) for speed/exactness at
33-123 node scale. Qiskit is used downstream where it belongs: (a) the density-matrix/
noise-channel formulation via qiskit-aer's DensityMatrix + Kraus channels, and
(b) QAOA for combinatorial rerouting. This division is standard practice and is what
we report/justify in the methodology section of the paper.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm


def build_hamiltonian(G: nx.Graph, weight_key="q_weight", laplacian=False):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    H = np.zeros((N, N), dtype=complex)
    for u, v, d in G.edges(data=True):
        w = d.get(weight_key, 1.0)
        i, j = idx[u], idx[v]
        H[i, j] = -w
        H[j, i] = -w
    if laplacian:
        deg = -H.sum(axis=1)
        H += np.diag(deg)
    return H, nodes, idx


class ContinuousTimeQuantumWalk:
    def __init__(self, G: nx.Graph, weight_key="q_weight", laplacian=False):
        self.G = G
        self.H, self.nodes, self.idx = build_hamiltonian(G, weight_key, laplacian)
        self.N = len(self.nodes)
        self.H_sparse = csr_matrix(self.H)

    def initial_state(self, source_node):
        psi0 = np.zeros(self.N, dtype=complex)
        psi0[self.idx[source_node]] = 1.0
        return psi0

    def evolve(self, psi0, t):
        """psi(t) = exp(-i H t) psi0, via sparse expm-action (exact, fast)."""
        return expm_multiply(-1j * self.H_sparse * t, psi0)

    def evolve_series(self, source_node, times, return_unitarity=False):
        psi0 = self.initial_state(source_node)
        probs = np.zeros((len(times), self.N))
        unitarity = np.zeros(len(times), dtype=float) if return_unitarity else None
        for k, t in enumerate(times):
            psi_t = self.evolve(psi0, t)
            probs[k] = np.abs(psi_t) ** 2
            if return_unitarity:
                unitarity[k] = float(np.sum(probs[k]))
        if return_unitarity:
            return probs, unitarity
        return probs  # shape (T, N), rows sum to 1

    def propagator(self, t):
        """Dense unitary exp(-iHt) -- used by the density-matrix module for small N."""
        return expm(-1j * self.H * t)

    def occupation_at(self, source_node, t):
        psi0 = self.initial_state(source_node)
        psi_t = self.evolve(psi0, t)
        return dict(zip(self.nodes, np.abs(psi_t) ** 2))
