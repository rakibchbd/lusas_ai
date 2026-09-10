"""Generate a dependency-free local HTML control dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .config import Settings
from .monitor import report


def _cell(value: object) -> str:
    return html.escape(str(value if value is not None else "-"))


def render(settings: Settings) -> str:
    payload = report(settings, recent=20)
    policy = payload["policy"]
    web = payload["web_learning"]
    rows = []
    for event in payload["recent_upgrades"]:
        rows.append(
            "<tr>"
            f"<td>{_cell(event.get('time'))}</td>"
            f"<td>{_cell(event.get('version'))}</td>"
            f"<td>{_cell(event.get('status'))}</td>"
            f"<td>{_cell(event.get('score'))}</td>"
            f"<td>{_cell(event.get('improvement'))}</td>"
            "</tr>"
        )
    data = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LUSAS AI dashboard</title>
<style>
body {{ font: 15px system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
  padding: 0 1rem; color: #17202a; background: #f6f8fa; }}
h1 {{ margin-bottom: .25rem; }} .muted {{ color: #667; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 1rem; }}
.card {{ background: white; border: 1px solid #d8dee4; border-radius: 10px; padding: 1rem; }}
.value {{ font-size: 1.45rem; font-weight: 700; margin-top: .3rem; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border-bottom: 1px solid #d8dee4; padding: .6rem; text-align: left; }}
code {{ background: #eaeef2; padding: .1rem .25rem; border-radius: 4px; }}
</style></head><body>
<h1>LUSAS AI</h1><p class="muted">Local control dashboard. Generated from local audit records.</p>
<section class="grid">
<div class="card">Version<div class="value">{_cell(payload['current_version'])}</div></div>
<div class="card">Successful upgrades<div class="value">{_cell(payload['upgrade_count'])}</div></div>
<div class="card">Web articles cached<div class="value">{_cell(web['cached_articles'])}</div></div>
<div class="card">Lessons / regressions<div class="value">{_cell(payload['lessons_count'])} / {_cell(payload['regressions_count'])}</div></div>
</section>
<h2>Policy</h2><div class="card">Model upgrades: <b>{_cell(policy['auto_apply_upgrades'])}</b> ·
Code upgrades: <b>{_cell(policy['auto_code_upgrades'])}</b> ·
Web learning: <b>{_cell(web['enabled'])}</b> every <b>{_cell(web['interval_minutes'])} minutes</b></div>
<h2>Upgrade history</h2><table><thead><tr><th>Time</th><th>Version</th><th>Status</th><th>Score</th><th>Improvement</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="5">No upgrade records yet.</td></tr>'}</tbody></table>
<script type="application/json" id="lusas-report">{data}</script>
</body></html>
"""


def write(settings: Settings) -> Path:
    path = settings.root / ".lusas" / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(render(settings), encoding="utf-8")
    temporary.replace(path)
    return path
