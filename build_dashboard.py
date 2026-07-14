"""
Rebuilds the self-contained interactive dashboard HTML from the current
qgrid.run_pipeline output. Run after changing datasets/parameters.

    python3 build_dashboard.py
"""
import json
import re
from qgrid.run_pipeline import run

if __name__ == "__main__":
    payload = run()  # change source="ieee123" / seed_edge=... / etc. here
    data_json = json.dumps(payload)

    with open("qgrid/viz/index.html", encoding="utf-8") as f:
        tpl = f.read()

    # Replace the embedded data block with fresh pipeline output
    html = re.sub(
        r"<script id=\"qgrid-data\" type=\"application/json\">[\s\S]*?<\/script>",
        f"<script id=\"qgrid-data\" type=\"application/json\">{data_json}</script>",
        tpl,
        flags=re.MULTILINE,
    )

    with open("qgrid/viz/dashboard.js", encoding="utf-8") as f:
        js = f.read()
    html = html.replace('<script src="/static/dashboard.js"></script>', f"<script>\n{js}\n</script>")

    out_path = "qgrid_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html)} bytes)")
