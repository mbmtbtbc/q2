import os
import time
from flask import Flask, render_template, request, jsonify
import networkx as nx
import numpy as np

# Import qgrid pipeline components
from qgrid.data.loaders import load_topology
from qgrid.graph.build import build_graph, approximate_power_flow
from qgrid.graph.weights import encode_quantum_weights
from qgrid.walk.ctqw import ContinuousTimeQuantumWalk, build_hamiltonian
from qgrid.walk.classical import ClassicalContinuousRandomWalk
from qgrid.walk.density import DensityMatrixWalk
from qgrid.benchmark.metrics import benchmark_walks
from qgrid.failures.topology import find_articulation_points, criticality_ranking
from qgrid.failures.cascade import simulate_cascade, cascade_impact_summary
from qgrid.reroute.qaoa_reroute import build_qubo, solve_qaoa_reroute, apply_reroute

# Use custom static and template folders to avoid moving existing viz files
app = Flask(__name__, template_folder='qgrid/viz', static_folder='qgrid/viz', static_url_path='/static')

# Global cache to hold the pre-computed grid state
GLOBAL_CACHE = {}

def npify(o):
    if isinstance(o, dict):
        return {str(k): npify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [npify(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (np.isinf(o) or np.isnan(o)):
        return None
    return o

def layout_positions(G):
    pos = nx.kamada_kawai_layout(G)
    return {str(n): [float(p[0]), float(p[1])] for n, p in pos.items()}

def initialize():
    if "G0" in GLOBAL_CACHE:
        return
    print("Initializing base graph and quantum walks...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    topology_csv = os.path.join(project_root, "qgrid", "ieee33_augmented.csv")
    load_csv = os.path.join(project_root, "qgrid", "ieee33_load.csv")
    
    topo, load, meta = load_topology("powercascade", topology_csv=topology_csv, load_csv=load_csv)
    G0 = build_graph(topo, load, meta)
    G0 = approximate_power_flow(G0)
    G0 = encode_quantum_weights(G0)
    slack = G0.graph["slack_bus"]
    source_bus = 1
    target_bus = 18
    pos = layout_positions(G0)

    # 1. Baseline Graph
    graph_payload = {
        "nodes": [{"id": n, "pos": pos[str(n)], "p_kw": G0.nodes[n]["p_kw"],
                   "is_slack": G0.nodes[n]["is_slack"]} for n in G0.nodes()],
        "edges": [{"u": u, "v": v, "q_weight": G0[u][v]["q_weight"],
                   "loading_pct": G0[u][v].get("loading_pct", 0.0),
                   "is_tie_switch": G0[u][v]["is_tie_switch"],
                   "status": G0[u][v]["status"]} for u, v in G0.edges()],
    }

    # 2. CTQW & CTRW
    qw = ContinuousTimeQuantumWalk(G0)
    cw = ClassicalContinuousRandomWalk(G0)
    times = np.linspace(0, 6, 21)
    probs_q, q_unitarity = qw.evolve_series(source_bus, times, return_unitarity=True)
    probs_c = cw.evolve_series(source_bus, times)
    nodes_sorted = qw.nodes
    dist_from_source = np.array([nx.shortest_path_length(G0, source_bus, n) for n in nodes_sorted], dtype=float)
    bench = benchmark_walks(probs_q, probs_c, times, qw.idx[source_bus], qw.idx[target_bus], cw.stationary_distribution(), dist_from_source)
    
    walk_payload = {
        "nodes_order": nodes_sorted,
        "times": times.tolist(),
        "quantum_probs": probs_q.tolist(),
        "quantum_unitarity": q_unitarity.tolist(),
        "classical_probs": probs_c.tolist(),
        "benchmark": npify(bench),
    }

    # 3. Density Matrix
    H, nodes_h, idx_h = build_hamiltonian(G0)
    partial_info_buses = (9, 15)
    partial_idx = [idx_h[b] for b in partial_info_buses if b in idx_h]
    dw = DensityMatrixWalk(H, gamma_deph=0.05, gamma_diff=0.01, partial_info_nodes=partial_idx)
    dtimes = np.linspace(0, 6, 5)
    series = dw.evolve_series(idx_h[source_bus], dtimes)
    dm_metrics = [dw.metrics(rho) for rho in series]
    density_payload = {
        "times": dtimes.tolist(),
        "trace": [m["trace"] for m in dm_metrics],
        "purity": [m["purity"] for m in dm_metrics],
        "entropy": [m["von_neumann_entropy"] for m in dm_metrics],
        "populations": [m["populations"].tolist() for m in dm_metrics],
        "partial_info_buses": list(partial_info_buses),
        "nodes_order": nodes_h,
    }

    # 4. Topology
    aps = find_articulation_points(G0)
    ranking = criticality_ranking(G0)
    
    base_payload = {
        "meta": {"grid_name": meta["name"], "n_buses": G0.number_of_nodes(),
                 "n_lines": G0.number_of_edges(), "slack_bus": slack,
                 "source_bus": source_bus, "target_bus": target_bus},
        "graph": graph_payload,
        "walk": walk_payload,
        "density": density_payload,
        "topology": {"articulation_points": aps, "criticality_ranking": npify(ranking[:10])},
    }
    
    GLOBAL_CACHE["G0"] = G0
    GLOBAL_CACHE["slack"] = slack
    GLOBAL_CACHE["base_payload"] = npify(base_payload)
    print("Initialization complete!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/base_data")
def base_data():
    return jsonify(GLOBAL_CACHE["base_payload"])

@app.route("/api/simulate")
def simulate():
    start_time = time.time()
    
    u = request.args.get("u")
    v = request.args.get("v")
    if u is None or v is None:
        return jsonify({"error": "Missing edge parameters u and v"}), 400
    try:
        # Note: nodes are integers in NetworkX
        u, v = int(u), int(v)
    except ValueError:
        return jsonify({"error": "Edge parameters must be integers"}), 400

    G0 = GLOBAL_CACHE["G0"]
    slack = GLOBAL_CACHE["slack"]
    
    if not G0.has_edge(u, v):
        return jsonify({"error": f"Edge ({u}, {v}) does not exist in graph"}), 404

    # 1. Cascade Simulation
    Gf, snaps, log = simulate_cascade(
        G0, seed_edges=[(u, v)], overload_threshold=1.0,
        alpha=0.5, beta=0.2, gamma=0.3,
    )
    impact = cascade_impact_summary(Gf)
    deenergized = [n for n, d in Gf.nodes(data=True) if not d.get("energized", True)]

    cascade_rounds = []
    for i, snap in enumerate(snaps):
        cascade_rounds.append({
            "round": i,
            "edges": [{"u": e_u, "v": e_v, "status": snap[e_u][e_v]["status"],
                       "loading_pct": snap[e_u][e_v].get("loading_pct", 0.0)} for e_u, e_v in snap.edges()],
        })

    # 2. Lightning-fast QAOA Reroute
    qw_post = ContinuousTimeQuantumWalk(Gf)
    occ = qw_post.occupation_at(slack, t=3.0)
    tie_candidates = []
    for n_u, n_v, d in Gf.edges(data=True):
        if d.get("is_tie_switch"):
            u_en = Gf.nodes[n_u].get("energized", True)
            v_en = Gf.nodes[n_v].get("energized", True)
            # Only consider tie switches that bridge an energized and de-energized bus
            if u_en != v_en:
                tie_candidates.append((n_u, n_v))
    h_qubo, J_qubo, cands = build_qubo(Gf, tie_candidates, occ)
    
    # Highly optimized QAOA for speed (seconds instead of minutes)
    decision, qaoa_info = solve_qaoa_reroute(h_qubo, J_qubo, p=1, shots=256, maxiter=15)
    
    G_healed, closed = apply_reroute(Gf, tie_candidates, decision)
    G_healed = approximate_power_flow(G_healed)
    active_edges = [(n_u, n_v) for n_u, n_v, d in G_healed.edges(data=True) if d["status"] == "active"]
    sub = G_healed.edge_subgraph(active_edges) if active_edges else nx.Graph()
    energized_now = nx.node_connected_component(sub, slack) if slack in sub else {slack}
    recovered = [n for n in deenergized if n in energized_now]

    healing_payload = {
        "seed_edge": [u, v],
        "de_energized_after_cascade": deenergized,
        "impact_summary": npify(impact),
        "tie_candidates": [list(t) for t in tie_candidates],
        "qaoa_occupation_signal": npify(occ),
        "qaoa_decision": npify(decision),
        "qaoa_optimal_cost": float(qaoa_info.get("optimal_cost", 0.0)),
        "qaoa_info": npify(qaoa_info),
        "closed_switches": [list(c) for c in closed],
        "recovered_buses": recovered,
        "still_down_buses": [n for n in deenergized if n not in energized_now],
        "healed_edges": [{"u": n_u, "v": n_v, "status": G_healed[n_u][n_v]["status"],
                           "loading_pct": G_healed[n_u][n_v].get("loading_pct", 0.0)} for n_u, n_v in G_healed.edges()],
    }

    elapsed = time.time() - start_time
    print(f"Simulation for edge ({u},{v}) took {elapsed:.2f}s")
    
    return jsonify(npify({
        "cascade_rounds": cascade_rounds,
        "healing": healing_payload
    }))

if __name__ == "__main__":
    print("Starting QGrid Server. This may take up to a minute...")
    initialize()
    app.run(debug=True, port=5000, use_reloader=False)
