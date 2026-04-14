"""AutoTrader Dashboard: real-time experiment visualization.

Serves a web UI showing:
- Live experiment log with keep/discard/crash status
- Composite score evolution chart
- Strategy lineage tree
- Safety status and rate limits
- Active strategy details

Run: python -m src.autotrader.dashboard
Or: uvicorn src.autotrader.dashboard:app --reload --port 8501
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


def create_app(
    results_tsv: str = "data/autotrader/experiment_results.tsv",
    strategies_dir: str = "data/autotrader/strategies",
    audit_log: str = "data/autotrader/audit.jsonl",
) -> Any:
    """Create the FastAPI dashboard app."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from .experiment_log import ExperimentLog

    app = FastAPI(title="AutoTrader Dashboard", version="1.0.0")
    log = ExperimentLog(path=results_tsv)
    strategies_path = Path(strategies_dir)
    audit_path = Path(audit_log)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _render_dashboard_html()

    @app.get("/api/experiments")
    async def get_experiments(limit: int = 100):
        results = log.recent(limit)
        return [
            {
                "experiment_id": r.experiment_id,
                "thesis_id": r.thesis_id,
                "spec_id": r.spec_id,
                "composite_score": r.composite_score,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "hit_rate": r.hit_rate,
                "cumulative_return": r.cumulative_return,
                "num_trades": r.num_trades,
                "duration": r.backtest_duration_seconds,
                "status": r.status,
                "description": r.description,
            }
            for r in results
        ]

    @app.get("/api/summary")
    async def get_summary():
        results = log.read_all()
        kept = [r for r in results if r.status == "keep"]
        discarded = [r for r in results if r.status == "discard"]
        crashed = [r for r in results if r.status in ("crash", "timeout")]
        best = max(kept, key=lambda r: r.composite_score) if kept else None

        return {
            "total": len(results),
            "kept": len(kept),
            "discarded": len(discarded),
            "crashed": len(crashed),
            "best_score": best.composite_score if best else 0,
            "best_sharpe": best.sharpe_ratio if best else 0,
            "best_description": best.description if best else "none",
            "score_history": [
                {"idx": i, "score": r.composite_score, "status": r.status}
                for i, r in enumerate(results)
            ],
        }

    @app.get("/api/strategies")
    async def get_strategies():
        strategies = []
        for path in sorted(strategies_path.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                strategies.append({
                    "spec_id": data.get("spec_id"),
                    "parent_id": data.get("parent_id"),
                    "description": data.get("description"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
        return strategies

    @app.get("/api/strategy/{spec_id}")
    async def get_strategy(spec_id: str):
        path = strategies_path / f"{spec_id}.json"
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(path.read_text())

    @app.get("/api/audit")
    async def get_audit(limit: int = 50):
        if not audit_path.exists():
            return []
        lines = audit_path.read_text().strip().split("\n")
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records

    return app


def _render_dashboard_html() -> str:
    """Render the single-page dashboard HTML with embedded JS charts."""
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoTrader Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0f; color: #e0e0e0; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
           padding: 20px 30px; border-bottom: 1px solid #333; }
.header h1 { font-size: 1.4em; color: #00d4ff; }
.header .subtitle { font-size: 0.85em; color: #888; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
         padding: 20px 30px; }
.card { background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px;
         padding: 16px; }
.card h3 { font-size: 0.75em; color: #888; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 8px; }
.card .value { font-size: 1.8em; font-weight: bold; }
.green { color: #00e676; } .red { color: #ff5252; }
.yellow { color: #ffd740; } .blue { color: #448aff; }
.chart-area { padding: 20px 30px; }
.chart-area canvas { width: 100%; height: 300px; background: #12121a;
                      border: 1px solid #2a2a3a; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; margin: 20px 30px;
         background: #12121a; border: 1px solid #2a2a3a; border-radius: 8px;
         overflow: hidden; }
th { background: #1a1a2e; padding: 10px 12px; text-align: left;
     font-size: 0.75em; color: #888; text-transform: uppercase; }
td { padding: 8px 12px; border-top: 1px solid #1a1a2e; font-size: 0.85em; }
tr:hover { background: #1a1a2e; }
.status-keep { color: #00e676; } .status-discard { color: #ffd740; }
.status-crash { color: #ff5252; }
#score-chart { width: calc(100% - 60px); margin: 0 30px; }
.refresh-btn { background: #1a1a2e; border: 1px solid #333; color: #00d4ff;
                padding: 6px 16px; border-radius: 4px; cursor: pointer;
                font-family: inherit; float: right; margin-top: -30px; }
.refresh-btn:hover { background: #2a2a3e; }
</style>
</head>
<body>
<div class="header">
    <h1>AutoTrader &mdash; Self-Improving Quant</h1>
    <div class="subtitle">Autonomous Strategy Experimentation Loop</div>
</div>

<div class="grid" id="summary-cards">
    <div class="card"><h3>Total Experiments</h3><div class="value blue" id="total">-</div></div>
    <div class="card"><h3>Kept</h3><div class="value green" id="kept">-</div></div>
    <div class="card"><h3>Best Score</h3><div class="value green" id="best-score">-</div></div>
    <div class="card"><h3>Best Sharpe</h3><div class="value blue" id="best-sharpe">-</div></div>
</div>

<div class="chart-area">
    <h3 style="color:#888;font-size:0.8em;margin-bottom:8px;padding-left:4px;">COMPOSITE SCORE EVOLUTION</h3>
    <canvas id="score-chart"></canvas>
</div>

<button class="refresh-btn" onclick="loadData()">Refresh</button>
<h3 style="color:#888;font-size:0.8em;margin:20px 30px 8px;text-transform:uppercase;letter-spacing:1px;">Recent Experiments</h3>
<table>
<thead><tr>
    <th>ID</th><th>Score</th><th>Sharpe</th><th>Return</th><th>DD</th>
    <th>Trades</th><th>Time</th><th>Status</th><th>Description</th>
</tr></thead>
<tbody id="experiments-table"></tbody>
</table>

<script>
async function loadData() {
    const [summary, experiments] = await Promise.all([
        fetch('/api/summary').then(r => r.json()),
        fetch('/api/experiments?limit=50').then(r => r.json()),
    ]);

    document.getElementById('total').textContent = summary.total;
    document.getElementById('kept').textContent = summary.kept;
    document.getElementById('best-score').textContent = summary.best_score.toFixed(4);
    document.getElementById('best-sharpe').textContent = summary.best_sharpe.toFixed(2);

    // Render table
    const tbody = document.getElementById('experiments-table');
    tbody.innerHTML = experiments.reverse().map(e => `<tr>
        <td>${e.experiment_id}</td>
        <td>${e.composite_score.toFixed(4)}</td>
        <td>${e.sharpe_ratio.toFixed(3)}</td>
        <td>${(e.cumulative_return * 100).toFixed(2)}%</td>
        <td>${(e.max_drawdown * 100).toFixed(2)}%</td>
        <td>${e.num_trades}</td>
        <td>${e.duration.toFixed(1)}s</td>
        <td class="status-${e.status}">${e.status}</td>
        <td>${e.description.slice(0, 50)}</td>
    </tr>`).join('');

    // Render chart
    drawChart(summary.score_history);
}

function drawChart(history) {
    const canvas = document.getElementById('score-chart');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 600;
    ctx.scale(2, 2);

    const w = canvas.offsetWidth, h = 300;
    ctx.clearRect(0, 0, w, h);

    if (!history.length) return;

    const scores = history.map(h => h.score);
    const maxScore = Math.max(...scores) * 1.1 || 1;
    const minScore = Math.min(...scores) * 0.9;

    // Grid
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
        const y = 20 + (h - 40) * i / 4;
        ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(w - 10, y); ctx.stroke();
        ctx.fillStyle = '#555';
        ctx.font = '10px monospace';
        ctx.fillText((maxScore - (maxScore - minScore) * i / 4).toFixed(3), 0, y + 4);
    }

    // Line
    const xStep = (w - 50) / Math.max(history.length - 1, 1);
    ctx.beginPath();
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.5;
    history.forEach((pt, i) => {
        const x = 40 + i * xStep;
        const y = 20 + (h - 40) * (1 - (pt.score - minScore) / (maxScore - minScore || 1));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    history.forEach((pt, i) => {
        const x = 40 + i * xStep;
        const y = 20 + (h - 40) * (1 - (pt.score - minScore) / (maxScore - minScore || 1));
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = pt.status === 'keep' ? '#00e676' : pt.status === 'discard' ? '#ffd740' : '#ff5252';
        ctx.fill();
    });
}

loadData();
setInterval(loadData, 15000);
</script>
</body>
</html>"""


# Allow running as module: python -m src.autotrader.dashboard
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8501)
