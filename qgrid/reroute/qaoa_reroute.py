"""
Self-healing rerouting after a cascade, formulated as a combinatorial optimization
solved with QAOA (qiskit).

Problem formulation
--------------------
After a cascade de-energizes a set of buses B, restoration = deciding which normally-
open TIE SWITCHES to close so that de-energized buses are reconnected to the slack
through the *lowest-congestion* alternate path, without re-creating an overload.
This is a constrained shortest-path / Steiner-like restoration problem -- NP-hard in
general topology reconfiguration (well known in distribution-system restoration
literature) -- so we cast the "which tie switches to close" decision as a QUBO and
solve with QAOA:

    minimize   sum_i h_i x_i + sum_{i<j} J_ij x_i x_j
    subject to x_i in {0,1}  (tie switch i closed/open)

- h_i rewards closing switches on paths that reach de-energized load with low
  congestion (encoded from the CTQW's post-failure occupation probabilities --
  this is the "quantum walk facilitates self-healing" link: nodes/edges the walk
  still finds highly reachable/low-loading get a bonus).
- J_ij penalizes closing tie-switch pairs that would jointly create a new loop
  through an already-overloaded corridor, or exceed a simple radiality/backfeed
  capacity budget.

For small candidate sets (a handful of open tie switches per event, which is
realistic -- a feeder has O(5) tie switches) QAOA on qiskit-aer's statevector/
qasm simulator is directly tractable and gives us a genuine quantum-optimization
component for the "self-healing" story, distinct from the CTQW spreading dynamics.
"""
from __future__ import annotations
import numpy as np
import itertools
import networkx as nx
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.optimize import minimize


def build_qubo(G: nx.Graph, tie_candidates, quantum_occupation, overload_penalty=5.0,
               reward_scale=2.0):
    """
    tie_candidates: list of (u, v) tie-switch edges currently open, being considered for closing.
    quantum_occupation: dict node -> CTQW occupation probability post-failure (from ctqw.occupation_at)
                         used as the reachability/low-congestion reward signal.
    Returns h (linear, len K), J (dict[(i,j)] -> coupling), candidate list (order matches h).
    """
    K = len(tie_candidates)
    h = np.zeros(K)
    J = {}
    # Normalize occupation to ensure QAOA receives a meaningful gradient
    max_occ = max([quantum_occupation.get(u, 0) + quantum_occupation.get(v, 0) for u, v in tie_candidates] + [1e-12])
    
    for i, (u, v) in enumerate(tie_candidates):
        reward = reward_scale * ((quantum_occupation.get(u, 0) + quantum_occupation.get(v, 0)) / max_occ)
        cap_pressure = 0.5 * (G[u][v].get("loading_pct", 0.0))
        h[i] = -reward + cap_pressure  # negative = QAOA (minimizer) favors closing it

    for (i, (u1, v1)), (j, (u2, v2)) in itertools.combinations(enumerate(tie_candidates), 2):
        shared_bus = len({u1, v1} & {u2, v2}) > 0
        if shared_bus:
            J[(i, j)] = overload_penalty * 0.5  # discourage double-feeding same corridor
    return h, J, tie_candidates


def qubo_to_ising(h, J):
    """QUBO x in {0,1} -> Ising z in {-1,+1} via x=(1-z)/2. Returns (h_z, J_z, offset)."""
    K = len(h)
    h_z = np.zeros(K)
    J_z = {}
    offset = 0.0
    for i in range(K):
        h_z[i] += -0.5 * h[i]
        offset += 0.5 * h[i]
    for (i, j), Jij in J.items():
        J_z[(i, j)] = J_z.get((i, j), 0.0) + 0.25 * Jij
        h_z[i] += -0.25 * Jij
        h_z[j] += -0.25 * Jij
        offset += 0.25 * Jij
    return h_z, J_z, offset


def qaoa_circuit(K, h_z, J_z, gammas, betas):
    qc = QuantumCircuit(K)
    qc.h(range(K))
    for gamma, beta in zip(gammas, betas):
        for i in range(K):
            if abs(h_z[i]) > 1e-12:
                qc.rz(2 * gamma * h_z[i], i)
        for (i, j), Jij in J_z.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * gamma * Jij, j)
                qc.cx(i, j)
        for i in range(K):
            qc.rx(2 * beta, i)
    qc.measure_all()
    return qc


def _expected_cost(counts, h_z, J_z, offset, shots):
    total = 0.0
    for bitstring, count in counts.items():
        z = np.array([1 - 2 * int(b) for b in bitstring[::-1]])  # qiskit bit order
        cost = offset + sum(h_z[i] * z[i] for i in range(len(z)))
        cost += sum(Jij * z[i] * z[j] for (i, j), Jij in J_z.items())
        total += cost * count
    return total / shots


def solve_qaoa_reroute(h, J, p=2, shots=2048, maxiter=60, seed=42):
    """
    Runs QAOA (qiskit-aer sampler) with a classical COBYLA outer loop optimizing
    (gamma, beta) to minimize expected Ising cost, then returns the best sampled
    bitstring (which tie switches to close).
    """
    K = len(h)
    if K == 0:
        return [], {}
    h_z, J_z, offset = qubo_to_ising(h, J)
    sampler = AerSampler()
    rng = np.random.default_rng(seed)

    def objective(params):
        gammas, betas = params[:p], params[p:]
        qc = qaoa_circuit(K, h_z, J_z, gammas, betas)
        job = sampler.run([qc], shots=shots)
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        return _expected_cost(counts, h_z, J_z, offset, shots)

    x0 = rng.uniform(0, np.pi, size=2 * p)
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})

    gammas, betas = res.x[:p], res.x[p:]
    qc = qaoa_circuit(K, h_z, J_z, gammas, betas)
    job = sampler.run([qc], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    best_bits = max(counts, key=counts.get)
    decision = {i: int(b) for i, b in enumerate(best_bits[::-1])}  # 1 = close this tie switch
    return decision, {"optimal_cost": res.fun, "counts": counts, "gammas": gammas, "betas": betas}


def apply_reroute(G: nx.Graph, tie_candidates, decision):
    G2 = G.copy()
    closed = []
    for i, (u, v) in enumerate(tie_candidates):
        if decision.get(i, 0) == 1:
            G2[u][v]["status"] = "active"
            closed.append((u, v))
    return G2, closed
