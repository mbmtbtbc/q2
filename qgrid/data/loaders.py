"""
Adapters that map heterogeneous source datasets onto one common schema so the
rest of the pipeline never has to know where the numbers came from.

Common schema (what every adapter must return)
------------------------------------------------
topology_df : columns [from_bus, to_bus, r_ohm, x_ohm, capacity_kva, is_tie_switch]
load_df     : columns [bus, timestamp, p_kw, q_kvar]     (timestamp optional -> static)
meta        : dict with at least {"slack_bus": int, "name": str}

Usage
-----
    from qgrid.data.loaders import load_topology

    # zero external files -> built-in IEEE-33
    topo, load, meta = load_topology("ieee33")

    # once you've downloaded the PowerCascade dataset from IEEE DataPort:
    topo, load, meta = load_topology(
        "powercascade", topology_csv="powercascade_topology.csv",
        cascade_csv="powercascade_events.csv"
    )

    # once you've downloaded the Kaggle real-time load monitoring set:
    topo, load, meta = load_topology(
        "ieee33", load_csv="smart_grid_real_time_load_monitoring.csv"
    )

    # future scale-up, single call-site change:
    topo, load, meta = load_topology("ieee123")
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from qgrid.data import ieee33


def _built_in_ieee33():
    topo = pd.DataFrame(ieee33.BRANCHES_33, columns=["from_bus", "to_bus", "r_ohm", "x_ohm"])
    topo["capacity_kva"] = ieee33.LINE_CAPACITY_KVA_DEFAULT
    topo["is_tie_switch"] = topo.apply(
        lambda r: (int(r.from_bus), int(r.to_bus)) in ieee33.TIE_SWITCHES
        or (int(r.to_bus), int(r.from_bus)) in ieee33.TIE_SWITCHES,
        axis=1,
    )
    load = pd.DataFrame(ieee33.LOADS_33, columns=["bus", "p_kw", "q_kvar"])
    load["timestamp"] = 0
    meta = {"slack_bus": ieee33.SLACK_BUS, "name": "IEEE-33", "n_buses": 33}
    return topo, load, meta


def _load_kaggle_realtime_loads(csv_path: str, n_buses: int):
    """
    Adapter for: kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset
    Expected raw columns vary by export; this adapter is defensive: it looks for
    common column name patterns and reshapes into [bus, timestamp, p_kw, q_kvar].
    Adjust COLMAP below once you inspect the real header row.
    """
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.lower()
    colmap_candidates = {
        "bus": ["bus", "bus_id", "node", "meter_id"],
        "timestamp": ["timestamp", "time", "datetime"],
        "p_kw": ["load_kw", "active_power_kw", "p_kw", "power_kw"],
        "q_kvar": ["reactive_power_kvar", "q_kvar", "load_kvar", "q_kvar"],
    }
    resolved = {}
    for std_name, candidates in colmap_candidates.items():
        for c in candidates:
            if c in raw.columns:
                resolved[std_name] = c
                break
    if "bus" not in resolved:
        # fall back: distribute rows cyclically over buses 2..n_buses (bus 1 = slack)
        raw["bus"] = (np.arange(len(raw)) % (n_buses - 1)) + 2
        resolved["bus"] = "bus"
    out = pd.DataFrame()
    out["bus"] = raw[resolved["bus"]]
    out["timestamp"] = raw[resolved["timestamp"]] if "timestamp" in resolved else 0
    out["p_kw"] = raw[resolved["p_kw"]] if "p_kw" in resolved else raw.select_dtypes("number").iloc[:, 0]
    out["q_kvar"] = raw[resolved["q_kvar"]] if "q_kvar" in resolved else out["p_kw"] * 0.5
    return out


def _load_powercascade(topology_csv: str = None, cascade_csv: str = None):
    """
    Adapter for: ieee-dataport.org PowerCascade synthetic cascading-failure dataset.
    If a topology CSV is supplied it is used as-is (must contain from/to bus + impedance
    columns, mapped defensively below); otherwise falls back to built-in IEEE-33 topology
    and only augments with cascade *event* labels for benchmarking against ground truth.
    """
    if topology_csv is not None:
        raw = pd.read_csv(topology_csv)
        raw.columns = raw.columns.str.lower()
        colmap = {"from_bus": ["from_bus", "from", "src", "bus_i", "frombus"],
                  "to_bus": ["to_bus", "to", "dst", "bus_j", "tobus"],
                  "r_ohm": ["r_ohm", "resistance", "r"],
                  "x_ohm": ["x_ohm", "reactance", "x"],
                  "capacity_kva": ["capacity_kva", "rating", "thermal_limit", "capacity"]}
        resolved = {}
        for std, cands in colmap.items():
            for c in cands:
                if c in raw.columns:
                    resolved[std] = c
                    break
        topo = pd.DataFrame({
            "from_bus": raw[resolved["from_bus"]],
            "to_bus": raw[resolved["to_bus"]],
            "r_ohm": raw.get(resolved.get("r_ohm"), 0.4),
            "x_ohm": raw.get(resolved.get("x_ohm"), 0.3),
        })
        topo["capacity_kva"] = raw[resolved["capacity_kva"]] if "capacity_kva" in resolved else ieee33.LINE_CAPACITY_KVA_DEFAULT
        if "status" in raw.columns:
            topo["is_tie_switch"] = raw["status"] == 0
        elif "is_tie_switch" in raw.columns:
            topo["is_tie_switch"] = raw["is_tie_switch"].astype(bool)
        else:
            topo["is_tie_switch"] = False
    else:
        topo, _, _ = _built_in_ieee33()

    cascade_events = pd.read_csv(cascade_csv) if cascade_csv else None
    load = pd.DataFrame(ieee33.LOADS_33, columns=["bus", "p_kw", "q_kvar"])
    load["timestamp"] = 0
    meta = {"slack_bus": ieee33.SLACK_BUS, "name": "PowerCascade", "cascade_events": cascade_events}
    return topo, load, meta


def load_topology(source: str = "ieee33", topology_csv: str = None, load_csv: str = None,
                   cascade_csv: str = None, n_buses: int = 33):
    """
    Single entry point. `source` selects the built-in reference network; the optional
    *_csv args let you overlay/replace pieces with your real downloaded datasets without
    touching any other module in the pipeline (this is the modularity seam for IEEE-33 -> IEEE-123).
    """
    if source == "ieee33":
        topo, load, meta = _built_in_ieee33()
    elif source == "powercascade":
        topo, load, meta = _load_powercascade(topology_csv, cascade_csv)
    elif source == "ieee123":
        raise NotImplementedError(
            "Drop an ieee123.py in qgrid/data/ with the same BRANCHES_123/LOADS_123 schema "
            "as ieee33.py, then add a `_built_in_ieee123()` branch here. No other module changes."
        )
    else:
        raise ValueError(f"Unknown source '{source}'")

    if load_csv is not None:
        load = _load_kaggle_realtime_loads(load_csv, n_buses=n_buses)

    return topo, load, meta
