"""AutoTrader Dashboard: real-time monitoring for the autonomous strategy loop.

Full-featured web UI with:
- Server-Sent Events (SSE) for real-time push updates
- Composite score evolution chart with keep/discard markers
- Live activity feed (what the loop is doing right now)
- Active strategy code viewer with syntax highlighting
- Strategy lineage tree (parent -> child evolution)
- Safety gauges (rate limits, crash count, halt status)
- Swap audit trail timeline
- Experiment detail modal on click
- Per-component metric sparklines (Sharpe, return, drawdown, hit rate)

Run:
    rlaif autotrader dashboard
    # or: uvicorn src.autotrader.dashboard:app --reload --port 8501
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

_HAS_FASTAPI = False
_HAS_SSE = False

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    _HAS_FASTAPI = True
    try:
        from sse_starlette.sse import EventSourceResponse
        _HAS_SSE = True
    except ImportError:
        pass
except ImportError:
    pass

# ── Shared state for SSE push ───────────────────────────────────────────

_orchestrator_ref: Any = None  # set by wire_orchestrator()
_last_event_id = 0


def wire_orchestrator(orchestrator: Any) -> None:
    """Connect a running orchestrator so the dashboard can read live state."""
    global _orchestrator_ref
    _orchestrator_ref = orchestrator


def create_app(
    results_tsv: str = "data/autotrader/experiment_results.tsv",
    strategies_dir: str = "data/autotrader/strategies",
    audit_log: str = "data/autotrader/audit.jsonl",
) -> Any:
    """Create the FastAPI dashboard application."""
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
    from .experiment_log import ExperimentLog

    app = FastAPI(title="AutoTrader Dashboard", version="2.0.0")
    log = ExperimentLog(path=results_tsv)
    strat_path = Path(strategies_dir)
    audit_path = Path(audit_log)

    # ── HTML ────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _DASHBOARD_HTML

    # ── REST API ────────────────────────────────────────────────────────

    @app.get("/api/experiments")
    async def api_experiments(limit: int = 200):
        return [
            {
                "experiment_id": r.experiment_id,
                "thesis_id": r.thesis_id,
                "spec_id": r.spec_id,
                "composite_score": r.composite_score,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "max_drawdown": r.max_drawdown,
                "hit_rate": r.hit_rate,
                "cumulative_return": r.cumulative_return,
                "num_trades": r.num_trades,
                "duration": r.backtest_duration_seconds,
                "status": r.status,
                "description": r.description,
            }
            for r in log.recent(limit)
        ]

    @app.get("/api/summary")
    async def api_summary():
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
            "keep_rate": round(len(kept) / max(len(results), 1) * 100, 1),
            "best_score": best.composite_score if best else 0,
            "best_sharpe": best.sharpe_ratio if best else 0,
            "best_return": best.cumulative_return if best else 0,
            "best_description": best.description if best else "none",
            "score_history": [
                {
                    "idx": i,
                    "score": r.composite_score,
                    "sharpe": r.sharpe_ratio,
                    "ret": r.cumulative_return,
                    "dd": r.max_drawdown,
                    "hit": r.hit_rate,
                    "status": r.status,
                    "desc": r.description,
                }
                for i, r in enumerate(results)
            ],
        }

    @app.get("/api/strategies")
    async def api_strategies():
        strategies = []
        for p in sorted(strat_path.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                strategies.append({
                    "spec_id": d.get("spec_id"),
                    "parent_id": d.get("parent_id"),
                    "description": d.get("description"),
                    "timestamp": d.get("timestamp"),
                    "has_code": bool(d.get("signal_code")),
                })
            except Exception:
                continue
        return strategies

    @app.get("/api/strategy/{spec_id}")
    async def api_strategy_detail(spec_id: str):
        p = strat_path / f"{spec_id}.json"
        if not p.exists():
            return JSONResponse({"error": "not found"}, 404)
        return json.loads(p.read_text())

    @app.get("/api/audit")
    async def api_audit(limit: int = 100):
        if not audit_path.exists():
            return []
        lines = audit_path.read_text().strip().split("\n")
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    @app.get("/api/live")
    async def api_live():
        """Return live orchestrator state (if wired)."""
        if _orchestrator_ref is None:
            return {"connected": False}
        try:
            return {"connected": True, **_orchestrator_ref.status()}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    # ── Settings API ───────────────────────────────────────────────────

    from .settings_manager import SettingsManager
    settings_mgr = SettingsManager()

    @app.get("/api/settings")
    async def api_settings():
        """Return all settings grouped, with secrets masked."""
        return {
            "groups": settings_mgr.get_grouped(),
            "issues": settings_mgr.validate(),
        }

    @app.post("/api/settings")
    async def api_settings_save(request: Request):
        """Save settings to .env.local. Body: {key: value, ...}"""
        body = await request.json()
        updates = {}
        for key, value in body.items():
            # Don't save masked values (the user didn't change them)
            if value and not value.startswith("*"):
                updates[key] = value
        if updates:
            settings_mgr.set_many(updates)
        return {"saved": list(updates.keys()), "issues": settings_mgr.validate()}

    @app.get("/api/settings/validate")
    async def api_settings_validate():
        return {"issues": settings_mgr.validate()}

    # ── SSE stream ──────────────────────────────────────────────────────

    if _HAS_SSE:
        @app.get("/api/stream")
        async def api_stream(request: Request):
            async def event_generator():
                last_count = log.total_count()
                while True:
                    if await request.is_disconnected():
                        break
                    current = log.total_count()
                    if current != last_count:
                        last_count = current
                        results = log.recent(1)
                        if results:
                            r = results[-1]
                            yield {
                                "event": "experiment",
                                "data": json.dumps({
                                    "experiment_id": r.experiment_id,
                                    "composite_score": r.composite_score,
                                    "sharpe_ratio": r.sharpe_ratio,
                                    "cumulative_return": r.cumulative_return,
                                    "max_drawdown": r.max_drawdown,
                                    "hit_rate": r.hit_rate,
                                    "status": r.status,
                                    "description": r.description,
                                    "total": current,
                                }),
                            }
                    await asyncio.sleep(2)

            return EventSourceResponse(event_generator())

    return app


# ── Dashboard HTML ──────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoTrader Dashboard</title>
<style>
:root {
  --bg:       #07080d;
  --surface:  #0e1018;
  --border:   #1e2030;
  --border-h: #2a2e44;
  --text:     #c8cdd8;
  --text-dim: #6b7084;
  --cyan:     #00d4ff;
  --green:    #00e676;
  --red:      #ff5252;
  --yellow:   #ffd740;
  --blue:     #448aff;
  --purple:   #b388ff;
  --font:     'SF Mono', 'Fira Code', 'Cascadia Code', 'JetBrains Mono', monospace;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text);
       font-size: 13px; overflow-x:hidden; }

/* Header */
.hdr { background: linear-gradient(135deg, #0f1225 0%, #0d1520 100%);
       padding: 16px 24px; border-bottom: 1px solid var(--border);
       display: flex; align-items: center; justify-content: space-between; }
.hdr h1 { font-size: 1.15em; color: var(--cyan); letter-spacing: 0.5px; }
.hdr .sub { font-size: 0.78em; color: var(--text-dim); margin-top: 2px; }
.hdr .live-dot { width:8px;height:8px;border-radius:50%;background:var(--green);
                  display:inline-block;animation:pulse 2s infinite;margin-right:6px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.hdr-right { display:flex; align-items:center; gap:12px; }
.hdr-right .ts { font-size:0.75em; color:var(--text-dim); }

/* Summary cards */
.cards { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; padding:14px 24px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:12px 14px; }
.card:hover { border-color:var(--border-h); }
.card h3 { font-size:0.65em; color:var(--text-dim); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:6px; }
.card .val { font-size:1.55em; font-weight:700; }
.card .delta { font-size:0.72em; margin-top:2px; }
.card .bar { height:3px; border-radius:2px; margin-top:8px; background:var(--border); }
.card .bar-fill { height:100%; border-radius:2px; transition:width 0.5s; }

/* Main grid */
.main { display:grid; grid-template-columns:1fr 340px; gap:0; min-height:calc(100vh - 180px); }

/* Chart panel */
.chart-panel { padding:14px 24px; border-right:1px solid var(--border); }
.chart-panel h2 { font-size:0.72em; color:var(--text-dim); text-transform:uppercase;
                   letter-spacing:1px; margin-bottom:8px; }
.chart-panel canvas { width:100%; height:260px; border-radius:6px; }

/* Side panel */
.side { background:var(--surface); overflow-y:auto; max-height:calc(100vh - 180px); }
.side-section { padding:14px 16px; border-bottom:1px solid var(--border); }
.side-section h2 { font-size:0.7em; color:var(--cyan); text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:10px; }

/* Activity feed */
.feed-item { padding:6px 0; border-bottom:1px solid var(--border); font-size:0.82em; }
.feed-item:last-child { border-bottom:none; }
.feed-item .fi-status { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.feed-item .fi-score { float:right; font-weight:600; }
.fi-keep { background:var(--green); } .fi-discard { background:var(--yellow); } .fi-crash { background:var(--red); }

/* Safety gauges */
.gauge { margin-bottom:10px; }
.gauge-label { font-size:0.72em; color:var(--text-dim); margin-bottom:3px;
               display:flex; justify-content:space-between; }
.gauge-track { height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
.gauge-fill { height:100%; border-radius:3px; transition:width 0.5s; }
.gauge-ok  .gauge-fill { background:var(--green); }
.gauge-warn .gauge-fill { background:var(--yellow); }
.gauge-crit .gauge-fill { background:var(--red); }

/* Code viewer */
.code-box { background:#0a0b10; border:1px solid var(--border); border-radius:6px;
            padding:10px 12px; font-size:0.78em; line-height:1.5; overflow-x:auto;
            max-height:300px; overflow-y:auto; white-space:pre; color:#90b0d0; }
.code-box .kw { color:var(--purple); } .code-box .fn { color:var(--cyan); }
.code-box .str { color:var(--green); } .code-box .cm { color:#555; }
.code-box .op { color:var(--yellow); }

/* Lineage */
.lineage-node { display:flex; align-items:center; padding:4px 0; font-size:0.8em; }
.lineage-node .ln-dot { width:10px;height:10px;border-radius:50%;border:2px solid var(--cyan);
                         background:var(--bg);margin-right:8px;flex-shrink:0; }
.lineage-node.ln-active .ln-dot { background:var(--cyan); }
.lineage-line { width:1px;height:12px;background:var(--border);margin-left:4px; }

/* Table */
.tbl-wrap { padding:0 24px 24px; overflow-x:auto; }
.tbl-wrap h2 { font-size:0.72em; color:var(--text-dim); text-transform:uppercase;
               letter-spacing:1px; margin:14px 0 8px; }
table { width:100%; border-collapse:collapse; background:var(--surface);
        border:1px solid var(--border); border-radius:6px; overflow:hidden; }
th { background:#0d0f18; padding:8px 10px; text-align:left; font-size:0.68em;
     color:var(--text-dim); text-transform:uppercase; letter-spacing:0.5px;
     position:sticky; top:0; }
td { padding:6px 10px; border-top:1px solid var(--border); font-size:0.8em; }
tr:hover td { background:#12141f; }
tr.row-keep td:first-child { border-left:3px solid var(--green); }
tr.row-discard td:first-child { border-left:3px solid var(--yellow); }
tr.row-crash td:first-child { border-left:3px solid var(--red); }
.clickable { cursor:pointer; }

/* Modal */
.modal-bg { display:none; position:fixed; top:0;left:0;right:0;bottom:0;
            background:rgba(0,0,0,0.7); z-index:100; align-items:center; justify-content:center; }
.modal-bg.open { display:flex; }
.modal { background:var(--surface); border:1px solid var(--border-h); border-radius:10px;
         padding:24px; max-width:700px; width:90%; max-height:80vh; overflow-y:auto; }
.modal h2 { color:var(--cyan); font-size:1em; margin-bottom:12px; }
.modal .close-btn { float:right; background:none; border:none; color:var(--text-dim);
                    cursor:pointer; font-size:1.3em; }
.modal .close-btn:hover { color:var(--text); }

/* Nav buttons */
.nav-btn { background:var(--surface); border:1px solid var(--border); color:var(--text-dim);
           padding:5px 14px; border-radius:4px; cursor:pointer; font-family:var(--font);
           font-size:0.78em; opacity:0.6; transition:all 0.2s; }
.nav-btn:hover { opacity:0.9; border-color:var(--border-h); }
.nav-btn.active { opacity:1; color:var(--cyan); border-color:var(--cyan); }

/* Settings page */
.page { display:none; }
.page.active { display:block; }
#page-monitor { display:block; }

.settings-wrap { max-width:800px; margin:0 auto; padding:24px; }
.settings-wrap h2 { font-size:0.85em; color:var(--cyan); margin:24px 0 12px; padding-bottom:6px;
                     border-bottom:1px solid var(--border); text-transform:uppercase; letter-spacing:1px; }
.settings-wrap h2:first-child { margin-top:0; }

.setting-row { display:grid; grid-template-columns:200px 1fr; gap:12px; align-items:start;
               padding:10px 0; border-bottom:1px solid var(--border); }
.setting-label { font-size:0.82em; font-weight:600; color:var(--text); padding-top:6px; }
.setting-label .req { color:var(--red); margin-left:2px; }
.setting-help { font-size:0.72em; color:var(--text-dim); margin-top:2px; }
.setting-source { font-size:0.65em; padding:1px 6px; border-radius:3px; margin-left:6px; }
.src-env   { background:#1a2a1a; color:var(--green); }
.src-local { background:#1a1a2a; color:var(--blue); }
.src-dotenv { background:#2a2a1a; color:var(--yellow); }
.src-default { background:var(--border); color:var(--text-dim); }

.setting-input { width:100%; background:var(--bg); border:1px solid var(--border); border-radius:4px;
                 padding:7px 10px; color:var(--text); font-family:var(--font); font-size:0.85em; }
.setting-input:focus { outline:none; border-color:var(--cyan); }
.setting-input.secret-set { color:var(--text-dim); }
select.setting-input { cursor:pointer; }

.toggle-row { display:flex; align-items:center; gap:8px; }
.toggle { position:relative; width:36px; height:20px; cursor:pointer; }
.toggle input { display:none; }
.toggle .slider { position:absolute; top:0;left:0;right:0;bottom:0; background:var(--border);
                  border-radius:10px; transition:0.3s; }
.toggle .slider:before { content:''; position:absolute; height:14px;width:14px;left:3px;bottom:3px;
                          background:var(--text-dim); border-radius:50%; transition:0.3s; }
.toggle input:checked + .slider { background:var(--cyan); }
.toggle input:checked + .slider:before { transform:translateX(16px); background:white; }

.save-bar { position:sticky; bottom:0; background:var(--surface); border-top:1px solid var(--border);
            padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
.save-btn { background:var(--cyan); color:#000; border:none; padding:8px 24px; border-radius:4px;
            font-family:var(--font); font-weight:700; font-size:0.85em; cursor:pointer; }
.save-btn:hover { opacity:0.9; }
.save-btn:disabled { opacity:0.4; cursor:default; }
.save-msg { font-size:0.82em; color:var(--green); }

.issues-bar { background:#1a1210; border:1px solid #3a2010; border-radius:6px; padding:10px 14px;
              margin-bottom:16px; }
.issues-bar .issue { font-size:0.82em; color:var(--yellow); padding:2px 0; }
.issues-bar .issue:before { content:'\\26A0 '; }

/* Responsive */
@media (max-width:1100px) {
  .cards { grid-template-columns:repeat(3,1fr); }
  .main { grid-template-columns:1fr; }
  .side { max-height:none; border-top:1px solid var(--border); }
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1><span class="live-dot" id="live-dot"></span>AutoTrader</h1>
    <div class="sub">Self-Improving Quant &mdash; Autonomous Strategy Experimentation</div>
  </div>
  <div class="hdr-right">
    <span class="ts" id="last-update">--</span>
    <button class="nav-btn" id="btn-monitor" onclick="showPage('monitor')" style="opacity:1">Monitor</button>
    <button class="nav-btn" id="btn-settings" onclick="showPage('settings')">Settings</button>
  </div>
</div>

<!-- PAGE: MONITOR -->
<div id="page-monitor" class="page active">

<!-- SUMMARY CARDS -->
<div class="cards">
  <div class="card"><h3>Experiments</h3><div class="val blue" id="c-total">-</div><div class="delta" id="c-rate">-</div></div>
  <div class="card"><h3>Kept</h3><div class="val green" id="c-kept">-</div>
    <div class="bar"><div class="bar-fill" id="c-kept-bar" style="width:0;background:var(--green)"></div></div></div>
  <div class="card"><h3>Discarded</h3><div class="val yellow" id="c-disc">-</div>
    <div class="bar"><div class="bar-fill" id="c-disc-bar" style="width:0;background:var(--yellow)"></div></div></div>
  <div class="card"><h3>Best Score</h3><div class="val green" id="c-score">-</div><div class="delta" id="c-score-d">-</div></div>
  <div class="card"><h3>Best Sharpe</h3><div class="val blue" id="c-sharpe">-</div></div>
  <div class="card"><h3>Best Return</h3><div class="val" id="c-return">-</div></div>
</div>

<!-- MAIN GRID -->
<div class="main">
  <!-- LEFT: Charts + Table -->
  <div>
    <div class="chart-panel">
      <h2>Composite Score Evolution</h2>
      <canvas id="chart-score"></canvas>
    </div>
    <div class="chart-panel">
      <h2>Component Metrics</h2>
      <canvas id="chart-components"></canvas>
    </div>
    <div class="tbl-wrap">
      <h2>Experiment History</h2>
      <table>
        <thead><tr>
          <th>ID</th><th>Score</th><th>Sharpe</th><th>Return</th><th>DD</th>
          <th>Hit</th><th>Trades</th><th>Time</th><th>Status</th><th>Description</th>
        </tr></thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
  </div>

  <!-- RIGHT: Side panel -->
  <div class="side">
    <!-- Live activity feed -->
    <div class="side-section">
      <h2>Live Activity</h2>
      <div id="feed"></div>
    </div>

    <!-- Safety gauges -->
    <div class="side-section">
      <h2>Safety Status</h2>
      <div id="safety-status"></div>
      <div class="gauge" id="g-exp">
        <div class="gauge-label"><span>Experiments / hour</span><span id="g-exp-v">0/12</span></div>
        <div class="gauge-track"><div class="gauge-fill" style="width:0"></div></div>
      </div>
      <div class="gauge" id="g-swap">
        <div class="gauge-label"><span>Swaps / 24h</span><span id="g-swap-v">0/6</span></div>
        <div class="gauge-track"><div class="gauge-fill" style="width:0"></div></div>
      </div>
      <div class="gauge" id="g-crash">
        <div class="gauge-label"><span>Consecutive crashes</span><span id="g-crash-v">0/10</span></div>
        <div class="gauge-track"><div class="gauge-fill" style="width:0"></div></div>
      </div>
    </div>

    <!-- Active strategy -->
    <div class="side-section">
      <h2>Active Strategy</h2>
      <div id="active-strat" style="font-size:0.82em;color:var(--text-dim)">Loading...</div>
    </div>

    <!-- Strategy lineage -->
    <div class="side-section">
      <h2>Strategy Lineage</h2>
      <div id="lineage"></div>
    </div>

    <!-- Swap audit trail -->
    <div class="side-section">
      <h2>Swap Audit Trail</h2>
      <div id="audit" style="font-size:0.8em;"></div>
    </div>
  </div>
</div>

</div><!-- /page-monitor -->

<!-- PAGE: SETTINGS -->
<div id="page-settings" class="page">
<div class="settings-wrap">
  <div id="settings-issues"></div>
  <div id="settings-form"></div>
  <div class="save-bar">
    <span class="save-msg" id="save-msg"></span>
    <button class="save-btn" id="save-btn" onclick="saveSettings()">Save Settings</button>
  </div>
</div>
</div><!-- /page-settings -->

<!-- MODAL -->
<div class="modal-bg" id="modal-bg" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="close-btn" onclick="closeModal()">&times;</button>
    <h2 id="modal-title">Experiment Detail</h2>
    <div id="modal-body"></div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────
let allExperiments = [];
let summary = {};
let strategies = [];

// ── Data fetching ──────────────────────────────────────────────────────
async function loadAll() {
  try {
    const [sum, exps, strats, audit, live] = await Promise.all([
      fetch('/api/summary').then(r=>r.json()),
      fetch('/api/experiments?limit=200').then(r=>r.json()),
      fetch('/api/strategies').then(r=>r.json()),
      fetch('/api/audit?limit=20').then(r=>r.json()),
      fetch('/api/live').then(r=>r.json()),
    ]);
    summary = sum;
    allExperiments = exps;
    strategies = strats;
    renderCards(sum);
    renderScoreChart(sum.score_history);
    renderComponentChart(sum.score_history);
    renderTable(exps);
    renderFeed(exps.slice(-15).reverse());
    renderSafety(live);
    renderActiveStrategy(live, strats);
    renderLineage(strats);
    renderAudit(audit);
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    document.getElementById('live-dot').style.background =
      live.connected && live.running ? 'var(--green)' : 'var(--yellow)';
  } catch(e) {
    console.error('loadAll failed:', e);
  }
}

// ── SSE real-time stream ───────────────────────────────────────────────
function connectSSE() {
  try {
    const es = new EventSource('/api/stream');
    es.addEventListener('experiment', (e) => {
      const d = JSON.parse(e.data);
      // Prepend to feed
      const feed = document.getElementById('feed');
      const item = makeFeedItem(d);
      feed.insertBefore(item, feed.firstChild);
      if (feed.children.length > 20) feed.removeChild(feed.lastChild);
      // Refresh full data every new experiment
      loadAll();
    });
    es.onerror = () => { setTimeout(connectSSE, 5000); es.close(); };
  } catch(e) { /* SSE not available, fall back to polling */ }
}

// ── Renderers ──────────────────────────────────────────────────────────

function renderCards(s) {
  document.getElementById('c-total').textContent = s.total;
  document.getElementById('c-rate').textContent = `${s.keep_rate}% keep rate`;
  document.getElementById('c-kept').textContent = s.kept;
  document.getElementById('c-disc').textContent = s.discarded;
  document.getElementById('c-score').textContent = s.best_score.toFixed(4);
  document.getElementById('c-score-d').textContent = s.best_description;
  document.getElementById('c-sharpe').textContent = s.best_sharpe.toFixed(2);
  const retEl = document.getElementById('c-return');
  retEl.textContent = (s.best_return * 100).toFixed(2) + '%';
  retEl.className = 'val ' + (s.best_return >= 0 ? 'green' : 'red');
  // Bars
  const tot = Math.max(s.total, 1);
  document.getElementById('c-kept-bar').style.width = (s.kept/tot*100)+'%';
  document.getElementById('c-disc-bar').style.width = (s.discarded/tot*100)+'%';
}

function renderScoreChart(history) {
  const canvas = document.getElementById('chart-score');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!history.length) { ctx.fillStyle='#555'; ctx.fillText('No data yet', w/2-30, h/2); return; }

  const scores = history.map(p=>p.score);
  const pad = {t:16,r:12,b:20,l:48};
  const maxS = Math.max(...scores)*1.08 || 1;
  const minS = Math.min(0, Math.min(...scores)*0.95);
  const xScale = i => pad.l + i/(Math.max(history.length-1,1))*(w-pad.l-pad.r);
  const yScale = v => pad.t + (1-(v-minS)/(maxS-minS||1))*(h-pad.t-pad.b);

  // Grid + labels
  ctx.strokeStyle = '#161828'; ctx.lineWidth = 0.5;
  for (let i=0; i<5; i++) {
    const v = minS + (maxS-minS)*i/4;
    const y = yScale(v);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
    ctx.fillStyle='#555'; ctx.font='9px monospace'; ctx.textAlign='right';
    ctx.fillText(v.toFixed(3), pad.l-4, y+3);
  }

  // Area fill
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(0));
  history.forEach((p,i) => ctx.lineTo(xScale(i), yScale(p.score)));
  ctx.lineTo(xScale(history.length-1), yScale(0));
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.t, 0, h-pad.b);
  grad.addColorStop(0, 'rgba(0,212,255,0.15)');
  grad.addColorStop(1, 'rgba(0,212,255,0.01)');
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath(); ctx.strokeStyle='#00d4ff'; ctx.lineWidth=1.8;
  history.forEach((p,i) => { i===0 ? ctx.moveTo(xScale(i),yScale(p.score)) : ctx.lineTo(xScale(i),yScale(p.score)); });
  ctx.stroke();

  // Dots with status color
  history.forEach((p,i) => {
    ctx.beginPath(); ctx.arc(xScale(i), yScale(p.score), 3.5, 0, Math.PI*2);
    ctx.fillStyle = p.status==='keep'?'#00e676':p.status==='discard'?'#ffd740':'#ff5252';
    ctx.fill();
  });

  // Best score line
  if (scores.length) {
    const best = Math.max(...scores);
    ctx.setLineDash([4,4]); ctx.strokeStyle='rgba(0,230,118,0.3)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l, yScale(best)); ctx.lineTo(w-pad.r, yScale(best)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='rgba(0,230,118,0.5)'; ctx.font='9px monospace'; ctx.textAlign='left';
    ctx.fillText('BEST '+best.toFixed(4), pad.l+4, yScale(best)-4);
  }
}

function renderComponentChart(history) {
  const canvas = document.getElementById('chart-components');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!history.length) return;
  const pad = {t:16,r:12,b:20,l:48};
  const xScale = i => pad.l + i/(Math.max(history.length-1,1))*(w-pad.l-pad.r);

  const series = [
    { key:'sharpe', label:'Sharpe', color:'#448aff' },
    { key:'ret',    label:'Return', color:'#00e676' },
    { key:'dd',     label:'Drawdown', color:'#ff5252' },
    { key:'hit',    label:'Hit Rate', color:'#ffd740' },
  ];

  // Normalize each series to [0,1]
  series.forEach(s => {
    const vals = history.map(p=>p[s.key]||0);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    s.norm = vals.map(v => mx===mn ? 0.5 : (v-mn)/(mx-mn));
  });

  const yScale = v => pad.t + (1-v)*(h-pad.t-pad.b);

  // Grid
  ctx.strokeStyle='#161828'; ctx.lineWidth=0.5;
  [0,0.25,0.5,0.75,1].forEach(v => {
    const y=yScale(v); ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
  });

  // Lines
  series.forEach(s => {
    ctx.beginPath(); ctx.strokeStyle=s.color; ctx.lineWidth=1.2; ctx.globalAlpha=0.8;
    s.norm.forEach((v,i) => { i===0?ctx.moveTo(xScale(i),yScale(v)):ctx.lineTo(xScale(i),yScale(v)); });
    ctx.stroke(); ctx.globalAlpha=1;
  });

  // Legend
  let lx = pad.l;
  series.forEach(s => {
    ctx.fillStyle=s.color; ctx.fillRect(lx, h-12, 10, 3);
    ctx.fillStyle='#888'; ctx.font='9px monospace'; ctx.textAlign='left';
    ctx.fillText(s.label, lx+14, h-8);
    lx += 80;
  });
}

function renderTable(exps) {
  const tbody = document.getElementById('tbl-body');
  tbody.innerHTML = exps.slice().reverse().map(e => `
    <tr class="row-${e.status} clickable" onclick="showDetail('${e.spec_id}','${e.experiment_id}')">
      <td>${e.experiment_id.slice(0,8)}</td>
      <td>${e.composite_score.toFixed(4)}</td>
      <td style="color:${e.sharpe_ratio>=0?'var(--green)':'var(--red)'}">${e.sharpe_ratio.toFixed(3)}</td>
      <td style="color:${e.cumulative_return>=0?'var(--green)':'var(--red)'}">${(e.cumulative_return*100).toFixed(2)}%</td>
      <td>${(e.max_drawdown*100).toFixed(2)}%</td>
      <td>${e.hit_rate.toFixed(3)}</td>
      <td>${e.num_trades}</td>
      <td>${e.duration.toFixed(1)}s</td>
      <td style="color:${e.status==='keep'?'var(--green)':e.status==='discard'?'var(--yellow)':'var(--red)'}">${e.status}</td>
      <td title="${e.description}">${e.description.slice(0,35)}</td>
    </tr>`).join('');
}

function makeFeedItem(d) {
  const el = document.createElement('div');
  el.className = 'feed-item';
  const sc = d.status==='keep'?'fi-keep':d.status==='discard'?'fi-discard':'fi-crash';
  el.innerHTML = `<span class="fi-status ${sc}"></span>${d.description||d.experiment_id}
    <span class="fi-score" style="color:${d.composite_score>0.5?'var(--green)':'var(--text-dim)'}">${(d.composite_score||0).toFixed(4)}</span>`;
  return el;
}

function renderFeed(exps) {
  const feed = document.getElementById('feed');
  feed.innerHTML = '';
  exps.forEach(e => feed.appendChild(makeFeedItem(e)));
  if (!exps.length) feed.innerHTML = '<div style="color:var(--text-dim);font-size:0.82em">Waiting for experiments...</div>';
}

function renderSafety(live) {
  const ss = document.getElementById('safety-status');
  if (!live.connected) {
    ss.innerHTML = '<div style="color:var(--yellow);font-size:0.82em;margin-bottom:8px;">Orchestrator not connected (view-only mode)</div>';
  } else if (live.safety && live.safety.halted) {
    ss.innerHTML = '<div style="color:var(--red);font-size:0.82em;margin-bottom:8px;font-weight:bold;">HALTED - Too many crashes</div>';
  } else {
    ss.innerHTML = '<div style="color:var(--green);font-size:0.82em;margin-bottom:8px;">Running</div>';
  }

  const sf = (live.safety || {});
  updateGauge('g-exp', sf.experiments_last_hour||0, sf.max_experiments_per_hour||12);
  updateGauge('g-swap', sf.swaps_last_24h||0, sf.max_swaps_per_day||6);
  updateGauge('g-crash', sf.consecutive_crashes||0, 10);
}

function updateGauge(id, val, max) {
  const el = document.getElementById(id);
  const pct = Math.min(val/max*100, 100);
  el.querySelector('.gauge-fill').style.width = pct+'%';
  el.querySelector('span:last-child').textContent = val+'/'+max;
  el.className = 'gauge ' + (pct > 80 ? 'gauge-crit' : pct > 50 ? 'gauge-warn' : 'gauge-ok');
}

function renderActiveStrategy(live, strats) {
  const el = document.getElementById('active-strat');
  const sw = (live.swapper || {});
  if (sw.active_spec_id) {
    el.innerHTML = `<div style="color:var(--cyan);font-weight:600;margin-bottom:4px;">${sw.active_description||'unnamed'}</div>
      <div style="color:var(--text-dim);">ID: ${sw.active_spec_id}</div>
      <div style="margin-top:6px;"><a href="#" onclick="showDetail('${sw.active_spec_id}','');return false;" style="color:var(--blue);text-decoration:none;">View signal code &rarr;</a></div>`;
  } else {
    el.innerHTML = '<span style="color:var(--text-dim)">No active strategy</span>';
  }
}

function renderLineage(strats) {
  const el = document.getElementById('lineage');
  if (!strats.length) { el.innerHTML='<span style="color:var(--text-dim);font-size:0.82em">No strategies yet</span>'; return; }

  // Build tree: find roots (no parent) and chain
  const byId = {};
  strats.forEach(s => byId[s.spec_id] = s);
  const roots = strats.filter(s => !s.parent_id || !byId[s.parent_id]);
  const last = strats[strats.length-1];

  // Show last 8 in chain
  const chain = [];
  let cur = last;
  while (cur && chain.length < 8) {
    chain.unshift(cur);
    cur = cur.parent_id ? byId[cur.parent_id] : null;
  }

  el.innerHTML = chain.map((s,i) => {
    const isActive = i === chain.length-1;
    return (i>0?'<div class="lineage-line"></div>':'') +
      `<div class="lineage-node ${isActive?'ln-active':''}">
        <div class="ln-dot"></div>
        <div>
          <div style="color:${isActive?'var(--cyan)':'var(--text)'};font-size:0.82em;">${s.description||s.spec_id}</div>
          <div style="color:var(--text-dim);font-size:0.7em;">${s.spec_id.slice(0,8)}</div>
        </div>
      </div>`;
  }).join('');
}

function renderAudit(records) {
  const el = document.getElementById('audit');
  if (!records.length) { el.innerHTML='<span style="color:var(--text-dim)">No swaps yet</span>'; return; }
  el.innerHTML = records.slice().reverse().slice(0,10).map(r => {
    const ts = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '';
    const color = r.success ? 'var(--green)' : 'var(--red)';
    return `<div style="padding:4px 0;border-bottom:1px solid var(--border);">
      <span style="color:${color};">${r.action}</span>
      <span style="color:var(--text-dim);float:right;">${ts}</span>
      <div style="color:var(--text-dim);font-size:0.75em;">${r.new_spec_id||''} ${r.reason||''}</div>
    </div>`;
  }).join('');
}

// ── Modal: show strategy code ──────────────────────────────────────────

async function showDetail(specId, expId) {
  if (!specId) return;
  try {
    const data = await fetch('/api/strategy/'+specId).then(r=>r.json());
    if (data.error) { alert('Strategy not found'); return; }

    document.getElementById('modal-title').textContent =
      (data.description || 'Strategy') + ' [' + specId.slice(0,8) + ']';

    const code = highlightPython(data.signal_code || 'No code');
    document.getElementById('modal-body').innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
        <div><span style="color:var(--text-dim)">Spec ID:</span> ${data.spec_id}</div>
        <div><span style="color:var(--text-dim)">Parent:</span> ${data.parent_id||'none (root)'}</div>
        <div><span style="color:var(--text-dim)">Created:</span> ${data.timestamp ? new Date(data.timestamp).toLocaleString() : '?'}</div>
      </div>
      <h3 style="color:var(--cyan);font-size:0.8em;margin-bottom:6px;">SIGNAL FUNCTION</h3>
      <div class="code-box">${code}</div>
      ${data.parameters && Object.keys(data.parameters).length ? `
        <h3 style="color:var(--cyan);font-size:0.8em;margin:12px 0 6px;">PARAMETERS</h3>
        <pre style="font-size:0.8em;color:var(--text-dim);">${JSON.stringify(data.parameters,null,2)}</pre>
      ` : ''}`;

    document.getElementById('modal-bg').classList.add('open');
  } catch(e) { console.error(e); }
}

function closeModal() { document.getElementById('modal-bg').classList.remove('open'); }
document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });

// ── Syntax highlighting (lightweight) ──────────────────────────────────

function highlightPython(code) {
  return code
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/(#.*)/gm, '<span class="cm">$1</span>')
    .replace(new RegExp('["\\']{3}[\\\\s\\\\S]*?["\\']{3}','gm'), m => '<span class="str">'+m+'</span>')
    .replace(/'[^']*'/gm, m => '<span class="str">'+m+'</span>')
    .replace(/"[^"]*"/gm, m => '<span class="str">'+m+'</span>')
    .replace(/\b(def|return|import|from|if|else|elif|for|while|in|not|and|or|is|None|True|False|as|with|try|except|class|lambda|yield|pass|break|continue)\b/g,
      '<span class="kw">$1</span>')
    .replace(/\b(generate_signals|pd|np|DataFrame|Series)\b/g, '<span class="fn">$1</span>')
    .replace(/(&amp;|&lt;|&gt;|[|+\\-*/=!])/g, '<span class="op">$1</span>');
}

// ── Page navigation ────────────────────────────────────────────────────

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  document.getElementById('btn-'+name).classList.add('active');
  if (name === 'settings') loadSettings();
}

// ── Settings ───────────────────────────────────────────────────────────

let settingsData = {};
let settingsDirty = false;

async function loadSettings() {
  try {
    const data = await fetch('/api/settings').then(r=>r.json());
    settingsData = data;
    renderSettingsForm(data.groups);
    renderSettingsIssues(data.issues);
    settingsDirty = false;
    document.getElementById('save-msg').textContent = '';
  } catch(e) { console.error('loadSettings:', e); }
}

function renderSettingsIssues(issues) {
  const el = document.getElementById('settings-issues');
  if (!issues || !issues.length) { el.innerHTML = ''; return; }
  el.innerHTML = '<div class="issues-bar">' +
    issues.map(i => '<div class="issue">'+i.message+'</div>').join('') + '</div>';
}

function renderSettingsForm(groups) {
  const form = document.getElementById('settings-form');
  let html = '';
  for (const [group, settings] of Object.entries(groups)) {
    html += '<h2>'+group+'</h2>';
    for (const s of settings) {
      const srcClass = 'src-'+(s.source||'default');
      const srcLabel = s.source==='env'?'ENV':s.source==='local'?'LOCAL':s.source==='dotenv'?'.ENV':'DEFAULT';
      html += '<div class="setting-row">';
      html += '<div class="setting-label">'+s.label+(s.required?'<span class="req">*</span>':'')+
        '<span class="setting-source '+srcClass+'">'+srcLabel+'</span>'+
        '<div class="setting-help">'+s.help+'</div></div>';
      html += '<div>';

      if (s.type === 'secret') {
        html += '<input class="setting-input'+(s.is_set?' secret-set':'')+'" type="password" '+
          'data-key="'+s.key+'" value="'+(s.value||'')+'" '+
          'placeholder="'+(s.is_set?'(set - click to change)':'Enter '+s.label)+'" '+
          'onfocus="this.type=\\'text\\';this.classList.remove(\\'secret-set\\')" '+
          'onblur="if(!this.value)this.type=\\'password\\'" '+
          'oninput="markDirty()">';
      } else if (s.type === 'select') {
        html += '<select class="setting-input" data-key="'+s.key+'" onchange="markDirty()">';
        for (const opt of (s.options||[])) {
          html += '<option value="'+opt+'"'+(s.value===opt?' selected':'')+'>'+opt+'</option>';
        }
        html += '</select>';
      } else if (s.type === 'toggle') {
        const checked = (s.value||'').toLowerCase() === 'true';
        html += '<div class="toggle-row"><label class="toggle"><input type="checkbox" data-key="'+s.key+'" '+
          (checked?'checked':'')+' onchange="markDirty()"><span class="slider"></span></label>'+
          '<span style="font-size:0.82em;color:var(--text-dim)">'+(checked?'Enabled':'Disabled')+'</span></div>';
      } else {
        html += '<input class="setting-input" type="text" data-key="'+s.key+'" '+
          'value="'+(s.value||'')+'" placeholder="'+s.label+'" oninput="markDirty()">';
      }

      html += '</div></div>';
    }
  }
  form.innerHTML = html;
}

function markDirty() {
  settingsDirty = true;
  document.getElementById('save-msg').textContent = 'Unsaved changes';
  document.getElementById('save-msg').style.color = 'var(--yellow)';
}

async function saveSettings() {
  const updates = {};
  document.querySelectorAll('.setting-input, .toggle input').forEach(el => {
    const key = el.dataset.key;
    if (!key) return;
    if (el.type === 'checkbox') {
      updates[key] = el.checked ? 'true' : 'false';
    } else {
      const val = el.value.trim();
      if (val) updates[key] = val;
    }
  });

  try {
    const resp = await fetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(updates),
    }).then(r=>r.json());

    document.getElementById('save-msg').textContent =
      'Saved ' + (resp.saved||[]).length + ' setting(s)';
    document.getElementById('save-msg').style.color = 'var(--green)';
    settingsDirty = false;

    // Reload to show updated sources
    setTimeout(loadSettings, 500);
  } catch(e) {
    document.getElementById('save-msg').textContent = 'Save failed: '+e;
    document.getElementById('save-msg').style.color = 'var(--red)';
  }
}

// Warn before leaving with unsaved changes
window.addEventListener('beforeunload', (e) => {
  if (settingsDirty) { e.preventDefault(); e.returnValue = ''; }
});

// ── Init ───────────────────────────────────────────────────────────────
loadAll();
connectSSE();
setInterval(loadAll, 5000);
</script>
</body>
</html>"""


# ── Module entry-point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8501)
