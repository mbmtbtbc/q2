"""
QGrid: Quantum-Walk-Based Cascading Failure Analysis & Self-Healing for Power Grids
=====================================================================================
Modular research pipeline:
    data       -> topology + load adapters (IEEE-33 built in, IEEE-123 / custom pluggable)
    graph      -> networkx construction + quantum weight encoding
    walk       -> CTQW (pure state) and density-matrix (mixed/open-system) formulations
    failures   -> cascading failure engine + articulation-point analysis
    reroute    -> QAOA-based self-healing rerouting
    benchmark  -> classical baseline + quantitative comparison metrics
    viz        -> interactive HTML/Plotly dashboard generation
"""
__version__ = "0.1.0"
