"""
Open-system (mixed-state) formulation of the grid quantum walk.

dρ/dt = -i[H, ρ] + Σ_k ( L_k ρ L_k†  -  1/2 {L_k† L_k, ρ} )     (Lindblad/GKSL)

Two physically-motivated noise channels are provided (NOT tied to a specific
field/sensor -- pure diffusion/decoherence noise as requested):

1. Dephasing noise (decoherence): L_k = sqrt(gamma_deph) |k><k|
   Destroys *coherences* (off-diagonal terms) involving node k while leaving
   populations untouched. This is how we model **partial/uncertain information**
   at a bus -- e.g. missing or stale telemetry -> the walk "forgets" quantum
   phase relationships through that node, degrading interference-based routing
   advantages locally without removing the node's capacity outright.

2. Dissipative/diffusive noise: amplitude-damping-like generator that leaks
   population toward a uniform "environment" distribution at rate gamma_diff,
   modeling generic environmental diffusion (line losses / thermal noise),
   independent of any specific measured field.

We integrate the Lindblad equation via the vectorized superoperator
(standard technique: vec(rho) with column-stacking, superop is dim^2 x dim^2),
solved by matrix exponential -- exact for the small (33-123 node) systems here.
qiskit.quantum_info.DensityMatrix is used post-hoc to compute purity / von Neumann
entropy / fidelity in a standard, citably-correct way (Qiskit supports arbitrary
operator dimensions, not just qubit registers).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import DensityMatrix, Statevector, entropy, purity


def _vectorize_superoperator(H, lindblad_ops):
    N = H.shape[0]
    I = np.eye(N, dtype=complex)
    superH = -1j * (np.kron(I, H) - np.kron(H.T, I))
    superD = np.zeros_like(superH)
    for L in lindblad_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        superD += np.kron(L.conj(), L) - 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    return superH + superD


def dephasing_operators(N, gamma_deph):
    return [np.sqrt(gamma_deph) * np.diag([1.0 if i == k else 0.0 for i in range(N)]).astype(complex)
            for k in range(N)]


def diffusive_loss_operators(N, gamma_diff):
    """Generic environmental diffusion: population leaks pairwise between all node
    pairs at small uniform rate, mimicking unmodeled environmental coupling / thermal
    diffusion -- not tied to any specific measured field."""
    ops = []
    for i in range(N):
        for j in range(N):
            if i != j:
                L = np.zeros((N, N), dtype=complex)
                L[i, j] = np.sqrt(gamma_diff / (N - 1))
                ops.append(L)
    return ops


class DensityMatrixWalk:
    def __init__(self, H, gamma_deph=0.05, gamma_diff=0.01, partial_info_nodes=None,
                 partial_info_boost=3.0):
        """
        H : Hamiltonian (dense ndarray) from walk.ctqw.build_hamiltonian
        partial_info_nodes : indices with degraded telemetry -> extra local dephasing,
                              modeling missing/partial information as decoherence.
        """
        self.H = H
        self.N = H.shape[0]
        ops = dephasing_operators(self.N, gamma_deph)
        if partial_info_nodes:
            boosted = dephasing_operators(self.N, gamma_deph * partial_info_boost)
            for k in partial_info_nodes:
                ops[k] = boosted[k]
        ops += diffusive_loss_operators(self.N, gamma_diff)
        self.superop = _vectorize_superoperator(H, ops)

    def initial_density_matrix(self, source_idx):
        rho0 = np.zeros((self.N, self.N), dtype=complex)
        rho0[source_idx, source_idx] = 1.0
        return rho0

    def evolve(self, rho0, t):
        vec_rho0 = rho0.reshape(-1, order="F")  # column-stack
        vec_rhot = expm(self.superop * t) @ vec_rho0
        return vec_rhot.reshape((self.N, self.N), order="F")

    def evolve_series(self, source_idx, times):
        rho0 = self.initial_density_matrix(source_idx)
        out = []
        for t in times:
            out.append(self.evolve(rho0, t))
        return out  # list of NxN density matrices

    @staticmethod
    def metrics(rho):
        dm = DensityMatrix(rho)
        return {
            "trace": float(np.real(np.trace(rho))),
            "purity": float(np.real(purity(dm))),
            "von_neumann_entropy": float(entropy(dm, base=2)),
            "populations": np.real(np.diag(rho)),
        }
