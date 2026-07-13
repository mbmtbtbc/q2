let DATA = null;
let SIMULATION = null;
let POS = {};

// ---------- shared helpers ----------
function nodePosMap(graph) {
  const m = {};
  graph.nodes.forEach(n => m[n.id] = n.pos);
  return m;
}

function edgeTraces(edges, colorFn, widthFn, opts = {}) {
  const traces = [];
  edges.forEach(e => {
    const p1 = POS[e.u], p2 = POS[e.v];
    if (!p1 || !p2) return;
    traces.push({
      x: [p1[0], p2[0], null], y: [p1[1], p2[1], null],
      mode: 'lines', type: 'scatter', hoverinfo: 'skip',
      line: { color: colorFn(e), width: widthFn(e) },
      showlegend: false
    });
  });
  return traces;
}

function nodeTrace(nodes, opts = {}) {
  return {
    x: nodes.map(n => POS[n.id][0]), y: nodes.map(n => POS[n.id][1]),
    mode: 'markers+text', type: 'scatter',
    text: nodes.map(n => String(n.id)),
    textposition: 'top center',
    textfont: { size: 9, color: '#8b93a7' },
    marker: {
      size: nodes.map(n => n.is_slack ? 20 : 11),
      color: nodes.map(n => n.is_slack ? '#fbbf24' : (opts.colorMap ? opts.colorMap(n) : '#5eead4')),
      line: { width: 1.5, color: '#0b0f19' },
      symbol: nodes.map(n => n.is_slack ? 'star' : 'circle')
    },
    hovertext: nodes.map(n => `Bus ${n.id}${n.is_slack ? ' (SLACK)' : ''}<br>Load: ${n.p_kw.toFixed(1)} kW`),
    hoverinfo: 'text',
    showlegend: false
  };
}

const LAYOUT_BASE = {
  paper_bgcolor: '#121826', plot_bgcolor: '#121826',
  font: { color: '#e6e9f2', size: 11 },
  xaxis: { visible: false }, yaxis: { visible: false },
  margin: { l: 10, r: 10, t: 10, b: 10 },
};

// ============================================================
// TAB DEFINITIONS
// ============================================================
const TABS = [
  { id: 'overview', label: '1 · Network & Quantum Weights' },
  { id: 'walk', label: '2 · CTQW vs Classical Baseline' },
  { id: 'density', label: '3 · Open-System Noise (Decoherence)' },
  { id: 'sandbox', label: '4 · Interactive Resilience Sandbox' },
];

function initTabs() {
  document.getElementById('meta-line').textContent =
    `${DATA.meta.grid_name} · ${DATA.meta.n_buses} buses · ${DATA.meta.n_lines} lines (incl. tie switches) · slack bus ${DATA.meta.slack_bus}`;
  document.getElementById('footer-grid').textContent = DATA.meta.grid_name;

  const tabsEl = document.getElementById('tabs');
  const pagesEl = document.getElementById('pages');
  TABS.forEach((t, i) => {
    const d = document.createElement('div');
    d.className = 'tab' + (i === 0 ? ' active' : '');
    d.textContent = t.label;
    d.onclick = () => selectTab(t.id);
    d.id = 'tabbtn-' + t.id;
    tabsEl.appendChild(d);
    const p = document.createElement('div');
    p.className = 'page' + (i === 0 ? ' active' : '');
    p.id = 'page-' + t.id;
    pagesEl.appendChild(p);
  });
}

function selectTab(id) {
  TABS.forEach(t => {
    document.getElementById('tabbtn-' + t.id).classList.toggle('active', t.id === id);
    document.getElementById('page-' + t.id).classList.toggle('active', t.id === id);
  });
  Plotly.Plots.resize(document.querySelectorAll(`#page-${id} .plot`)[0] || document.createElement('div'));
}

// ============================================================
// TAB 1: OVERVIEW
// ============================================================
function buildOverview() {
  const page = document.getElementById('page-overview');
  page.innerHTML = `
    <div class="grid2">
      <div class="card">
        <h3>Weighted Grid Graph</h3>
        <div class="sub">Edge color/width = encoded quantum coupling weight w_ij = α·admittance + β·capacity + γ·headroom. Dashed = normally-open tie switch.</div>
        <div id="plot-overview-graph" class="plot" style="height:480px;"></div>
      </div>
      <div class="card">
        <h3>Structural Criticality</h3>
        <div class="sub">Articulation points &amp; top-10 critical buses (betweenness + load + cut-vertex flag)</div>
        <div id="plot-criticality" class="plot" style="height:250px;"></div>
        <div style="margin-top:10px;">
          <div class="metric"><span>Articulation points</span><span class="vd">${DATA.topology.articulation_points.join(', ') || 'none'}</span></div>
        </div>
      </div>
    </div>
  `;
  const edges = DATA.graph.edges;
  const wVals = edges.map(e => e.q_weight);
  const wMax = Math.max(...wVals, 1e-6);
  const traces = edgeTraces(edges,
    e => e.is_tie_switch ? 'rgba(167,139,250,0.55)' : `rgba(94,234,212,${0.25 + 0.65 * (e.q_weight / wMax)})`,
    e => e.is_tie_switch ? 2 : 1.5 + 5 * (e.q_weight / wMax)
  );
  traces.push(nodeTrace(DATA.graph.nodes));
  Plotly.newPlot('plot-overview-graph', traces, LAYOUT_BASE, { displayModeBar: false, responsive: true });

  const rank = DATA.topology.criticality_ranking;
  Plotly.newPlot('plot-criticality', [{
    x: rank.map(r => r.criticality_score), y: rank.map(r => 'Bus ' + r.bus),
    type: 'bar', orientation: 'h', marker: { color: rank.map(r => r.is_articulation ? '#fb7185' : '#5eead4') }
  }], { ...LAYOUT_BASE, xaxis: { visible: true, color: '#8b93a7', gridcolor: '#2a3350' }, yaxis: { visible: true, color: '#8b93a7', autorange: 'reversed' }, margin: { l: 60, r: 10, t: 10, b: 30 } },
    { displayModeBar: false, responsive: true });
}

// ============================================================
// TAB 2: CTQW VS CLASSICAL
// ============================================================
function buildWalk() {
  const page = document.getElementById('page-walk');
  const b = DATA.walk.benchmark;
  page.innerHTML = `
    <div class="grid3" style="grid-template-columns: 1fr 1fr;">
      <div class="card">
        <h3>Quantum CTQW — Probability Distribution</h3>
        <div class="sub">exp(-iHt) from bus ${DATA.meta.source_bus}. Ballistic, interference-driven spreading.</div>
        <div id="plot-qwalk" class="plot" style="height:420px;"></div>
        <div class="controls"><label>t = <span id="walk-t-label">0.0</span></label>
          <input type="range" id="walk-slider" min="0" max="${DATA.walk.times.length - 1}" value="0"></div>
      </div>
      <div class="card">
        <h3>Classical CTRW — Probability Distribution</h3>
        <div class="sub">exp(-Lt) — same graph, same weights. Diffusive spreading (benchmark baseline).</div>
        <div id="plot-cwalk" class="plot" style="height:420px;"></div>
      </div>
    </div>
    <div class="grid2" style="margin-top:18px;">
      <div class="card">
        <h3>Spread Curve: Ballistic vs Diffusive</h3>
        <div class="sub">Mean graph-distance from source vs time — the headline CTQW benchmark result.</div>
        <div id="plot-spread" class="plot" style="height:280px;"></div>
      </div>
      <div class="card">
        <h3>Benchmark Metrics</h3>
        <div class="sub">Quantum vs classical walk, same initial condition &amp; weights</div>
        <div class="metric"><span>Final IPR (quantum)</span><span class="v">${b.quantum_ipr_final.toFixed(3)}</span></div>
        <div class="metric"><span>Final IPR (classical)</span><span class="v">${b.classical_ipr_final.toFixed(3)}</span></div>
        <div class="metric"><span>Final TVD (quantum vs classical)</span><span class="v">${b.final_tvd_quantum_vs_classical.toFixed(3)}</span></div>
        <div class="metric"><span>Classical mixing time (tol 0.05)</span><span class="v">${b.classical_mixing_time === null ? '> horizon' : b.classical_mixing_time.toFixed(2)}</span></div>
        <div class="flowtext" style="margin-top:10px;">Lower IPR ⇒ more delocalized. The quantum walk's IPR trajectory and TVD-from-classical quantify the coherent-transport advantage referenced in the paper's results section.</div>
      </div>
    </div>
  `;
  const nodes = DATA.graph.nodes;
  const nodesOrder = DATA.walk.nodes_order;
  const staticEdges = edgeTraces(DATA.graph.edges, e => 'rgba(120,130,160,0.25)', e => 1);

  function frameTraces(probs) {
    const maxp = Math.max(...probs, 1e-9);
    return [...staticEdges, {
      x: nodesOrder.map(n => POS[n][0]), y: nodesOrder.map(n => POS[n][1]),
      mode: 'markers', type: 'scatter',
      marker: {
        size: probs.map(p => 8 + 42 * Math.sqrt(p / maxp)), color: probs, colorscale: 'Viridis', cmin: 0, cmax: maxp,
        line: { width: 1, color: '#0b0f19' }, showscale: true, colorbar: { thickness: 10, len: 0.6, tickfont: { color: '#8b93a7', size: 9 } }
      },
      text: nodesOrder.map((n, i) => `Bus ${n}<br>P=${probs[i].toFixed(4)}`), hoverinfo: 'text', showlegend: false
    }];
  }
  Plotly.newPlot('plot-qwalk', frameTraces(DATA.walk.quantum_probs[0]), LAYOUT_BASE, { displayModeBar: false, responsive: true });
  Plotly.newPlot('plot-cwalk', frameTraces(DATA.walk.classical_probs[0]), LAYOUT_BASE, { displayModeBar: false, responsive: true });

  const slider = document.getElementById('walk-slider');
  slider.addEventListener('input', () => {
    const k = parseInt(slider.value);
    document.getElementById('walk-t-label').textContent = DATA.walk.times[k].toFixed(2);
    Plotly.react('plot-qwalk', frameTraces(DATA.walk.quantum_probs[k]), LAYOUT_BASE, { displayModeBar: false });
    Plotly.react('plot-cwalk', frameTraces(DATA.walk.classical_probs[k]), LAYOUT_BASE, { displayModeBar: false });
  });

  Plotly.newPlot('plot-spread', [
    { x: DATA.walk.times, y: b.quantum_spread_curve, name: 'Quantum (ballistic)', mode: 'lines', line: { color: '#5eead4', width: 3 } },
    { x: DATA.walk.times, y: b.classical_spread_curve, name: 'Classical (diffusive)', mode: 'lines', line: { color: '#a78bfa', width: 3, dash: 'dot' } }
  ], {
    ...LAYOUT_BASE, xaxis: { visible: true, title: 't', color: '#8b93a7', gridcolor: '#2a3350' }, yaxis: { visible: true, title: 'mean hop-distance', color: '#8b93a7', gridcolor: '#2a3350' },
    legend: { orientation: 'h', font: { color: '#e6e9f2' } }, margin: { l: 50, r: 10, t: 10, b: 40 }
  }, { displayModeBar: false, responsive: true });
}

// ============================================================
// TAB 3: DENSITY MATRIX / NOISE
// ============================================================
function buildDensity() {
  const page = document.getElementById('page-density');
  const d = DATA.density;
  page.innerHTML = `
    <div class="grid2">
      <div class="card">
        <h3>Purity &amp; Von Neumann Entropy</h3>
        <div class="sub">Lindblad evolution with dephasing (decoherence) + diffusive loss. Extra dephasing injected at buses ${d.partial_info_buses.join(', ')} = simulated missing/partial telemetry.</div>
        <div id="plot-purity" class="plot" style="height:320px;"></div>
      </div>
      <div class="card">
        <h3>Population Diffusion (Mixed State)</h3>
        <div class="sub">Diagonal of ρ(t) across buses over time — open-system analogue of the pure-state CTQW.</div>
        <div id="plot-populations" class="plot" style="height:320px;"></div>
      </div>
    </div>
    <div class="card" style="margin-top:18px;">
      <h3>Partial Information → Decoherence: Network View</h3>
      <div class="sub">Buses with degraded telemetry (highlighted) receive boosted local dephasing, modeling loss of quantum-coherent routing advantage where grid observability is poor.</div>
      <div class="controls"><label>t = <span id="dm-t-label">0.0</span></label>
        <input type="range" id="dm-slider" min="0" max="${d.times.length - 1}" value="0"></div>
      <div id="plot-dm-graph" class="plot" style="height:420px;"></div>
    </div>
  `;
  Plotly.newPlot('plot-purity', [
    { x: d.times, y: d.purity, name: 'Purity Tr(ρ²)', mode: 'lines+markers', line: { color: '#5eead4' } },
    { x: d.times, y: d.entropy, name: 'Von Neumann entropy (bits)', mode: 'lines+markers', line: { color: '#fbbf24' }, yaxis: 'y2' }
  ], {
    ...LAYOUT_BASE, xaxis: { visible: true, title: 't', color: '#8b93a7', gridcolor: '#2a3350' },
    yaxis: { visible: true, title: 'purity', range: [0, 1.05], color: '#8b93a7', gridcolor: '#2a3350' },
    yaxis2: { visible: true, overlaying: 'y', side: 'right', title: 'entropy', color: '#8b93a7' },
    legend: { orientation: 'h', font: { color: '#e6e9f2' } }, margin: { l: 50, r: 50, t: 10, b: 40 }
  }, { displayModeBar: false, responsive: true });

  const nOrder = d.nodes_order;
  const zPop = d.populations[0].length ? transposeTimeSeries(d.populations) : [];
  Plotly.newPlot('plot-populations', [{
    z: transposeTimeSeries(d.populations), x: d.times, y: nOrder.map(n => 'Bus ' + n),
    type: 'heatmap', colorscale: 'Viridis', showscale: true, colorbar: { thickness: 10, tickfont: { color: '#8b93a7', size: 9 } }
  }], { ...LAYOUT_BASE, xaxis: { visible: true, title: 't', color: '#8b93a7' }, yaxis: { visible: true, color: '#8b93a7', tickfont: { size: 8 } }, margin: { l: 60, r: 10, t: 10, b: 40 } },
    { displayModeBar: false, responsive: true });

  function transposeTimeSeries(pops) {
    const T = pops.length, N = pops[0].length;
    const z = [];
    for (let i = 0; i < N; i++) { z.push(pops.map(row => row[i])); }
    return z;
  }

  const staticEdges = edgeTraces(DATA.graph.edges, e => 'rgba(120,130,160,0.25)', e => 1);
  function dmFrame(k) {
    const pops = d.populations[k];
    const maxp = Math.max(...pops, 1e-9);
    const isPartial = n => d.partial_info_buses.includes(n);
    return [...staticEdges, {
      x: nOrder.map(n => POS[n][0]), y: nOrder.map(n => POS[n][1]),
      mode: 'markers', type: 'scatter',
      marker: {
        size: pops.map(p => 8 + 42 * Math.sqrt(p / maxp)),
        color: nOrder.map((n, i) => isPartial(n) ? '#fb7185' : '#5eead4'),
        opacity: pops.map(p => 0.35 + 0.65 * (p / maxp)),
        line: { width: nOrder.map(n => isPartial(n) ? 2 : 1), color: '#0b0f19' }
      },
      text: nOrder.map((n, i) => `Bus ${n}${isPartial(n) ? ' (partial info)' : ''}<br>ρ_ii=${pops[i].toFixed(4)}`), hoverinfo: 'text', showlegend: false
    }];
  }
  Plotly.newPlot('plot-dm-graph', dmFrame(0), LAYOUT_BASE, { displayModeBar: false, responsive: true });
  document.getElementById('dm-slider').addEventListener('input', function () {
    const k = parseInt(this.value);
    document.getElementById('dm-t-label').textContent = d.times[k].toFixed(2);
    Plotly.react('plot-dm-graph', dmFrame(k), LAYOUT_BASE, { displayModeBar: false });
  });
}

// ============================================================
// TAB 4: INTERACTIVE SANDBOX
// ============================================================
function buildSandbox() {
  const page = document.getElementById('page-sandbox');

  // Build a dropdown for edges
  let edgeOptions = DATA.graph.edges.filter(e => !e.is_tie_switch).map(e => `<option value="${e.u},${e.v}">Line ${e.u} → ${e.v}</option>`).join('');

  page.innerHTML = `
    <div class="card" style="margin-bottom:18px;">
      <h3>Interactive Failure Simulation</h3>
      <div class="sub">Select a line to fail. The server will instantly run a cascading failure simulation and solve the QAOA self-healing QUBO to restore power.</div>
      <div class="controls" style="margin: 10px 0 0 0;">
        <select id="fail-edge-select" style="padding:6px; background:#1e293b; color:#e6e9f2; border:1px solid #2a3350; border-radius:5px;">
          ${edgeOptions}
        </select>
        <button id="btn-simulate" class="btn primary">Simulate Failure</button>
      </div>
    </div>
    <div id="sandbox-results" style="display:none;">
      <div class="grid2">
        <div class="card">
          <h3>1. Cascading Failure</h3>
          <div class="sub" id="cascade-sub"></div>
          <div id="plot-cascade-graph" class="plot" style="height:440px;"></div>
        </div>
        <div class="card">
          <h3>2. QAOA Self-Healing Response</h3>
          <div class="sub" id="healing-sub"></div>
          <div id="plot-healed-graph" class="plot" style="height:440px;"></div>
        </div>
      </div>
      <div class="card" style="margin-top:18px;">
        <h3>Recovery Outcome</h3>
        <div class="metric"><span>QAOA optimal Ising cost</span><span class="v" id="qaoa-cost"></span></div>
        <div class="metric"><span>Buses recovered</span><span class="vo" id="buses-recovered"></span></div>
        <div id="bus-badges" style="margin-top:14px;"></div>
      </div>
    </div>
  `;

  document.getElementById('btn-simulate').addEventListener('click', () => {
    const edge = document.getElementById('fail-edge-select').value.split(',');
    document.getElementById('loading-overlay').style.display = 'flex';
    fetch(`/api/simulate?u=${edge[0]}&v=${edge[1]}`).then(r => r.json()).then(res => {
      SIMULATION = res;
      document.getElementById('loading-overlay').style.display = 'none';
      if (res.error) {
        alert("Error: " + res.error);
        return;
      }
      document.getElementById('sandbox-results').style.display = 'block';
      renderSandbox(res, edge);
    });
  });
}

function renderSandbox(sim, seedEdge) {
  const rounds = sim.cascade_rounds;
  const h = sim.healing;

  document.getElementById('cascade-sub').innerHTML = `Seed contingency: line (${seedEdge[0]}, ${seedEdge[1]}) tripped. Cascaded into ${rounds.length - 1} rounds.`;
  document.getElementById('healing-sub').innerHTML = `Tie switch(es) closed: <b style="color:#34d399">${h.closed_switches.map(c => '(' + c[0] + ',' + c[1] + ')').join(', ') || 'none needed'}</b>`;
  document.getElementById('qaoa-cost').textContent = h.qaoa_optimal_cost.toFixed(4);
  document.getElementById('buses-recovered').textContent = `${h.recovered_buses.length} / ${h.de_energized_after_cascade.length}`;

  document.getElementById('bus-badges').innerHTML = h.de_energized_after_cascade.map(b => `<span class="badge ${h.recovered_buses.includes(b) ? 'energized' : 'dead'}">Bus ${b} ${h.recovered_buses.includes(b) ? '✓ restored' : '✗ down'}</span>`).join(' ');

  const finalCascadeEdges = rounds[rounds.length - 1].edges;
  function graphFrame(edges) {
    return [...edgeTraces(edges,
      e => e.status === 'failed' ? '#fb7185' : (e.status === 'open' ? 'rgba(167,139,250,0.35)' : `rgba(94,234,212,${0.3 + 0.6 * e.loading_pct})`),
      e => e.status === 'failed' ? 3 : 2
    ), nodeTrace(DATA.graph.nodes, { colorMap: n => h.de_energized_after_cascade.includes(n.id) ? '#fb7185' : '#5eead4' })];
  }
  Plotly.react('plot-cascade-graph', graphFrame(finalCascadeEdges), LAYOUT_BASE, { displayModeBar: false });
  Plotly.react('plot-healed-graph', graphFrame(h.healed_edges), LAYOUT_BASE, { displayModeBar: false });
}

// ---------- init ----------
document.getElementById('meta-line').textContent = "Loading grid base state from server...";
fetch('/api/base_data').then(r => r.json()).then(d => {
  DATA = d;
  POS = nodePosMap(DATA.graph);

  initTabs();
  buildOverview();
  buildWalk();
  buildDensity();
  buildSandbox();
});
