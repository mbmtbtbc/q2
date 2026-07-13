"""
Rebuilds the self-contained interactive dashboard HTML from the current
qgrid.run_pipeline output. Run after changing datasets/parameters.

    python3 build_dashboard.py
"""
import json
from qgrid.run_pipeline import run

if __name__ == "__main__":
    payload = run()  # change source="ieee123" / seed_edge=... / etc. here
    data_json = json.dumps(payload)

    with open("qgrid/viz/dashboard_template.html") as f:
        tpl = f.read()
    html = tpl.replace("__DATA_JSON__", data_json)

    with open("qgrid/viz/dashboard.js") as f:
        js = f.read()
    html = html.replace('<script src="dashboard.js"></script>', f"<script>\n{js}\n</script>")

    out_path = "qgrid_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html)} bytes)")
