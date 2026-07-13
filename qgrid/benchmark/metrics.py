"""
Quantitative benchmark metrics: quantum CTQW vs classical CTRW baseline.
These are the numbers that go into the paper's results table/figures.
"""
from __future__ import annotations
import numpy as np


def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def inverse_participation_ratio(p):
    """IPR -> localization measure. Low IPR = spread out, high IPR = localized."""
    return 1.0 / np.sum(p ** 2)


def hitting_time(probs, times, target_idx, threshold=0.05):
    """First time the walker's probability at target node exceeds threshold."""
    hits = np.where(probs[:, target_idx] >= threshold)[0]
    return times[hits[0]] if len(hits) else np.inf


def mixing_time(probs, times, stationary, tol=0.05):
    tvd = np.array([total_variation_distance(probs[k], stationary) for k in range(len(times))])
    below = np.where(tvd <= tol)[0]
    return times[below[0]] if len(below) else np.inf


def spread_rate(probs, times, node_positions):
    """
    Average distance-from-source traveled per unit time (graph-distance weighted).
    node_positions: array of shortest-path distances from source, aligned with probs columns.
    Ballistic (quantum, O(t)) vs diffusive (classical, O(sqrt(t))) spreading is the
    headline qualitative benchmark result for CTQW papers.
    """
    mean_dist = probs @ node_positions
    return mean_dist


def benchmark_walks(quantum_probs, classical_probs, times, source_idx, target_idx,
                     stationary_classical, node_distances):
    return {
        "quantum_ipr_final": inverse_participation_ratio(quantum_probs[-1]),
        "classical_ipr_final": inverse_participation_ratio(classical_probs[-1]),
        "quantum_hitting_time": hitting_time(quantum_probs, times, target_idx),
        "classical_hitting_time": hitting_time(classical_probs, times, target_idx),
        "classical_mixing_time": mixing_time(classical_probs, times, stationary_classical),
        "quantum_spread_curve": spread_rate(quantum_probs, times, node_distances),
        "classical_spread_curve": spread_rate(classical_probs, times, node_distances),
        "final_tvd_quantum_vs_classical": total_variation_distance(quantum_probs[-1], classical_probs[-1]),
    }
