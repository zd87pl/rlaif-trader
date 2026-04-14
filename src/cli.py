"""
RLAIF Trading — Click CLI

Usage:
    rlaif status              Show system status
    rlaif analyze SYMBOL      Run multi-agent analysis
    rlaif options SYMBOL      Options-specific analysis
    rlaif paper               Start paper trading
    rlaif live                Start live trading (requires confirmation)
    rlaif backtest            Run backtest
    rlaif setup               Setup wizard
    rlaif model list          List available MLX models
    rlaif model load ID       Download / load a model
    rlaif risk                Show risk dashboard
    rlaif kill                Emergency kill switch

Install as CLI:
    pip install -e .          (with entry_points pointing here)
    # or just: python -m src.cli
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .chat_interface import ChatRouter, run_chat_session

load_dotenv()

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pipeline(mode: str = "paper", config: str = "configs/config.yaml"):
    """Lazy-create a TradingPipeline — import here to keep CLI startup fast."""
    from main import TradingPipeline
    return TradingPipeline(config_path=config, mode=mode)


def _ok(msg: str):
    click.echo(click.style("  ✓ ", fg="green", bold=True) + msg)


def _warn(msg: str):
    click.echo(click.style("  ⚠ ", fg="yellow", bold=True) + msg)


def _err(msg: str):
    click.echo(click.style("  ✗ ", fg="red", bold=True) + msg)


def _header(msg: str):
    click.echo()
    click.echo(click.style(f"  ━━━ {msg} ━━━", fg="cyan", bold=True))
    click.echo()


def _kv(key: str, value, color: str = "white"):
    click.echo(f"  {click.style(key + ':', bold=True):>28s}  {click.style(str(value), fg=color)}")


def _json_out(data: dict):
    click.echo(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------
@click.group()
@click.option("--config", "-c", default="configs/config.yaml", help="Path to config YAML.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose / debug logging.")
@click.pass_context
def cli(ctx, config, verbose):
    """RLAIF Trading Pipeline — multi-agent LLM + options + RLAIF."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--message", "-m", default=None, help="Single plain-English request instead of interactive chat.")
@click.pass_context
def chat(ctx, message):
    """Start a plain-English chat session over the trading pipeline."""
    router = ChatRouter(
        pipeline_factory=lambda config: _pipeline(config=config),
        config_path=ctx.obj["config"],
    )

    if message:
        click.echo(router.handle(message))
        return

    _header("Chat")
    run_chat_session(router, input_fn=click.prompt, output_fn=click.echo)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def status(ctx):
    """Show system status: model, broker, risk state."""
    _header("System Status")
    try:
        pipe = _pipeline(config=ctx.obj["config"])
        st = pipe.status()

        _kv("Mode", st.get("mode", "?"), "yellow")
        _kv("LLM Client", st.get("llm_client", "?"), "green")
        _kv("Foundation Model", st.get("foundation_model", "?"), "green")
        _kv("Broker", st.get("broker", "?"), "green")
        _kv("Broker Connected", st.get("broker_connected", "?"), "green" if st.get("broker_connected") else "red")
        _kv("RAG Enabled", st.get("rag_enabled", False), "green" if st.get("rag_enabled") else "yellow")
        _kv("RLAIF Enabled", st.get("rlaif_enabled", False), "green" if st.get("rlaif_enabled") else "yellow")
        _kv("Risk State", st.get("risk_state", "?"), "green")

        click.echo()
        click.echo(click.style("  Agents:", bold=True))
        for agent in st.get("agents", []):
            _ok(agent)

        click.echo()
        click.echo(click.style("  Asset Universe:", bold=True))
        assets = st.get("asset_universe", [])
        click.echo(f"    {', '.join(assets)}")
        click.echo()

    except Exception as exc:
        _err(f"Failed to get status: {exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("symbol")
@click.option("--json-output", "-j", is_flag=True, help="Raw JSON output.")
@click.pass_context
def analyze(ctx, symbol, json_output):
    """Run full multi-agent analysis on SYMBOL."""
    symbol = symbol.upper()
    _header(f"Analysing {symbol}")

    pipe = _pipeline(config=ctx.obj["config"])
    result = pipe.analyze(symbol)

    if json_output:
        _json_out(result)
        return

    _kv("Symbol", result["symbol"], "cyan")
    _kv("Timestamp", result["timestamp"])
    _kv("Elapsed", f"{result.get('elapsed_seconds', '?')}s")

    # Per-agent summaries
    click.echo()
    click.echo(click.style("  Agent Results:", bold=True))
    for agent_name, output in result.get("agents", {}).items():
        if isinstance(output, dict) and "error" in output:
            _err(f"{agent_name}: {output['error']}")
        else:
            signal = output.get("signal", output.get("recommendation", "—")) if isinstance(output, dict) else "—"
            confidence = output.get("confidence", "—") if isinstance(output, dict) else "—"
            click.echo(f"    {click.style(agent_name, bold=True):>24s}  signal={signal}  confidence={confidence}")

    # Manager decision
    click.echo()
    mgr = result.get("manager_decision", {})
    if isinstance(mgr, dict) and "error" not in mgr:
        _kv("Manager Action", mgr.get("action", mgr.get("decision", "—")), "yellow")
        _kv("Manager Confidence", mgr.get("confidence", "—"), "yellow")
        reasoning = mgr.get("reasoning", "")
        if reasoning:
            click.echo()
            click.echo(click.style("  Reasoning:", bold=True))
            click.echo(f"    {reasoning[:500]}")
    elif isinstance(mgr, dict):
        _err(f"Manager: {mgr.get('error')}")

    # Trades
    trades = result.get("recommended_trades", [])
    if trades:
        click.echo()
        click.echo(click.style("  Recommended Trades:", bold=True))
        for t in trades:
            click.echo(f"    • {t}")
    click.echo()


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("symbol")
@click.option("--json-output", "-j", is_flag=True, help="Raw JSON output.")
@click.pass_context
def options(ctx, symbol, json_output):
    """Options-specific analysis: vol surface, flow, strategies."""
    symbol = symbol.upper()
    _header(f"Options Analysis — {symbol}")

    pipe = _pipeline(config=ctx.obj["config"])
    result = pipe.analyze_options(symbol)

    if json_output:
        _json_out(result)
        return

    _kv("Symbol", result["symbol"], "cyan")
    _kv("Timestamp", result["timestamp"])

    vol = result.get("vol_surface")
    if vol:
        click.echo()
        click.echo(click.style("  Volatility Surface:", bold=True))
        if isinstance(vol, dict):
            for k, v in list(vol.items())[:6]:
                click.echo(f"    {k}: {v}")

    flow = result.get("unusual_flow")
    if flow:
        click.echo()
        click.echo(click.style("  Unusual Flow:", bold=True))
        if isinstance(flow, list):
            for item in flow[:5]:
                click.echo(f"    • {item}")
        else:
            click.echo(f"    {flow}")

    strategies = result.get("recommended_strategies")
    if strategies:
        click.echo()
        click.echo(click.style("  Recommended Strategies:", bold=True))
        if isinstance(strategies, list):
            for s in strategies[:5]:
                click.echo(f"    • {s}")
        else:
            click.echo(f"    {strategies}")
    click.echo()


# ---------------------------------------------------------------------------
# paper
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def paper(ctx):
    """Start paper trading with the scheduler."""
    _header("Paper Trading")
    _warn("Starting paper-trading loop. Press Ctrl+C to stop.")
    click.echo()

    pipe = _pipeline(mode="paper", config=ctx.obj["config"])
    try:
        pipe.run_paper()
    except KeyboardInterrupt:
        click.echo()
        _warn("Paper trading stopped by user.")


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def live(ctx):
    """Start LIVE trading (requires confirmation)."""
    _header("Live Trading")
    _err("WARNING: This will place REAL orders with REAL money.")
    click.echo()

    if not click.confirm(click.style("  Are you sure you want to trade LIVE?", fg="red", bold=True)):
        _warn("Aborted.")
        return

    # Double-confirm
    confirm_text = click.prompt(
        click.style('  Type "CONFIRM LIVE" to proceed', fg="red"),
        type=str,
    )
    if confirm_text.strip() != "CONFIRM LIVE":
        _warn("Aborted — confirmation text did not match.")
        return

    _ok("Starting live trading …")
    pipe = _pipeline(mode="live", config=ctx.obj["config"])
    try:
        pipe.run_live()
    except KeyboardInterrupt:
        click.echo()
        _warn("Live trading stopped by user.")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--start", "-s", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", "-e", required=True, help="End date (YYYY-MM-DD).")
@click.option("--symbols", "-S", default=None, help="Comma-separated symbols (default: config universe).")
@click.option("--json-output", "-j", is_flag=True, help="Raw JSON output.")
@click.pass_context
def backtest(ctx, start, end, symbols, json_output):
    """Run a backtest over a date range."""
    _header(f"Backtest  {start} → {end}")
    sym_list = [s.strip().upper() for s in symbols.split(",")] if symbols else None

    pipe = _pipeline(config=ctx.obj["config"])
    result = pipe.backtest(start_date=start, end_date=end, symbols=sym_list)

    if json_output:
        _json_out(result)
        return

    _kv("Period", f"{result['start_date']} → {result['end_date']}")
    _kv("Symbols", ", ".join(result.get("symbols", [])))
    _kv("Trades Generated", len(result.get("trades", [])))

    metrics = result.get("metrics", {})
    if metrics:
        click.echo()
        click.echo(click.style("  Metrics:", bold=True))
        for k, v in metrics.items():
            _kv(k, v)
    click.echo()


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def setup(ctx):
    """Run setup wizard: check deps, download model, configure broker."""
    _header("Setup Wizard")

    # 1) Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        _ok(f"Python {py_ver}")
    else:
        _err(f"Python {py_ver} — need ≥ 3.10")

    # 2) Key dependencies
    deps = ["click", "torch", "numpy", "pandas", "anthropic", "alpaca_trade_api"]
    for dep in deps:
        try:
            __import__(dep)
            _ok(dep)
        except ImportError:
            _warn(f"{dep} not installed")

    # 3) MLX (Apple Silicon only)
    try:
        import mlx
        _ok("mlx (Apple Silicon)")
    except ImportError:
        _warn("mlx not available (optional, Apple Silicon only)")

    # 4) API keys
    click.echo()
    click.echo(click.style("  API Keys:", bold=True))
    for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY"]:
        val = os.getenv(key, "")
        if val:
            _ok(f"{key} = {val[:6]}…{'*' * 8}")
        else:
            _warn(f"{key} not set")

    # 5) Config file
    click.echo()
    cfg_path = Path(ctx.obj["config"])
    if cfg_path.exists():
        _ok(f"Config found: {cfg_path}")
    else:
        _err(f"Config missing: {cfg_path}")

    # 6) Foundation model
    click.echo()
    click.echo(click.style("  Foundation Model:", bold=True))
    try:
        from src.utils import load_config
        cfg = load_config(str(cfg_path))
        fm = cfg.get("foundation_model", {}).get("model_type", "timesfm")
        _kv("Configured", fm)
    except Exception:
        _warn("Could not read config")

    click.echo()
    _ok("Setup check complete. Fix any warnings above before trading.")
    click.echo()


# ---------------------------------------------------------------------------
# model (group)
# ---------------------------------------------------------------------------
@cli.group(name="model")
def model_group():
    """Manage MLX / foundation models."""
    pass


@model_group.command(name="list")
def model_list():
    """List available MLX models."""
    _header("Available Models")
    try:
        from src.models.mlx_manager import MLXModelManager
        mgr = MLXModelManager()
        models = mgr.list_models() if hasattr(mgr, "list_models") else []
        if models:
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name", m.get("id", str(m)))
                    size = m.get("size", "?")
                    click.echo(f"    {click.style(name, fg='cyan'):>40s}  {size}")
                else:
                    click.echo(f"    {click.style(str(m), fg='cyan')}")
        else:
            _warn("No models found. Use 'rlaif model load <ID>' to download one.")
    except Exception as exc:
        _err(f"Could not list models: {exc}")
    click.echo()


@model_group.command(name="load")
@click.argument("model_id")
def model_load(model_id):
    """Download / load a specific MLX model by ID."""
    _header(f"Loading Model: {model_id}")
    try:
        from src.models.mlx_manager import MLXModelManager
        mgr = MLXModelManager()
        click.echo(f"  Downloading {model_id} …")
        result = mgr.load_model(model_id) if hasattr(mgr, "load_model") else mgr.download(model_id)
        _ok(f"Model loaded: {model_id}")
    except Exception as exc:
        _err(f"Failed to load model: {exc}")
    click.echo()


# ---------------------------------------------------------------------------
# autotrader
# ---------------------------------------------------------------------------
@cli.group(name="autotrader")
def autotrader_group():
    """Autonomous strategy experimentation loop (autoresearch for trading)."""
    pass


@autotrader_group.command(name="start")
@click.option("--mode", "-m", default=None, type=click.Choice(["continuous", "on_event", "hourly"]),
              help="Override mode from config.")
@click.pass_context
def autotrader_start(ctx, mode):
    """Start the autonomous experimentation loop (NEVER STOP)."""
    _header("AutoTrader — Self-Improving Quant")
    _warn("Starting autonomous experimentation loop. Press Ctrl+C to stop.")
    click.echo()

    pipe = _pipeline(config=ctx.obj["config"])
    if pipe.autotrader is None:
        _err("AutoTrader not initialised. Check configs/autotrader.yaml (enabled: true)")
        return

    try:
        pipe.run_autotrader(mode=mode)
    except KeyboardInterrupt:
        click.echo()
        _warn("AutoTrader stopped by user.")


@autotrader_group.command(name="status")
@click.pass_context
def autotrader_status(ctx):
    """Show autotrader status: experiments run, best score, safety state."""
    _header("AutoTrader Status")

    pipe = _pipeline(config=ctx.obj["config"])
    st = pipe.autotrader_status()

    if not st.get("enabled", True):
        _warn(f"AutoTrader disabled: {st.get('reason', 'unknown')}")
        return

    _kv("Running", st.get("running", False), "green" if st.get("running") else "yellow")
    _kv("Mode", st.get("mode", "?"))
    _kv("Iterations", st.get("iteration_count", 0))
    _kv("Best Score", f"{st.get('current_best_score', 0):.6f}", "green")
    _kv("Threshold", f"{st.get('improvement_threshold', 0):.4f}")

    click.echo()
    click.echo(click.style("  Experiment Log:", bold=True))
    click.echo(f"    {st.get('log_summary', 'No data')}")

    safety = st.get("safety", {})
    if safety:
        click.echo()
        click.echo(click.style("  Safety:", bold=True))
        _kv("Halted", safety.get("halted", False), "red" if safety.get("halted") else "green")
        _kv("Crashes", safety.get("consecutive_crashes", 0))
        _kv("Experiments/hr", f"{safety.get('experiments_last_hour', 0)}/{safety.get('max_experiments_per_hour', 12)}")
        _kv("Swaps/24h", f"{safety.get('swaps_last_24h', 0)}/{safety.get('max_swaps_per_day', 6)}")

    swapper = st.get("swapper", {})
    if swapper:
        click.echo()
        click.echo(click.style("  Active Strategy:", bold=True))
        _kv("Spec ID", swapper.get("active_spec_id", "none"))
        _kv("Description", swapper.get("active_description", "none"))
        _kv("Strategies on disk", swapper.get("strategies_on_disk", 0))

    click.echo()


@autotrader_group.command(name="history")
@click.option("--limit", "-n", default=20, help="Number of recent experiments to show.")
@click.pass_context
def autotrader_history(ctx, limit):
    """Show recent experiment history."""
    _header(f"AutoTrader History (last {limit})")

    try:
        from src.autotrader import ExperimentLog
        log = ExperimentLog()
        results = log.recent(limit)

        if not results:
            _warn("No experiments yet.")
            return

        # Header
        click.echo(f"  {'ID':>12s}  {'Score':>8s}  {'Sharpe':>7s}  {'Return':>8s}  {'DD':>7s}  {'Status':>8s}  Description")
        click.echo(f"  {'─' * 12}  {'─' * 8}  {'─' * 7}  {'─' * 8}  {'─' * 7}  {'─' * 8}  {'─' * 30}")

        for r in results:
            status_color = {"keep": "green", "discard": "yellow", "crash": "red", "timeout": "red"}.get(r.status, "white")
            click.echo(
                f"  {r.experiment_id:>12s}  "
                f"{r.composite_score:>8.4f}  "
                f"{r.sharpe_ratio:>7.3f}  "
                f"{r.cumulative_return:>8.4f}  "
                f"{r.max_drawdown:>7.4f}  "
                f"{click.style(r.status:>8s, fg=status_color)}  "
                f"{r.description[:40]}"
            )

        click.echo()
        click.echo(f"  {log.format_summary()}")
    except Exception as exc:
        _err(f"Failed to read history: {exc}")
    click.echo()


@autotrader_group.command(name="dashboard")
@click.option("--port", "-p", default=8501, help="Dashboard port.")
@click.pass_context
def autotrader_dashboard(ctx, port):
    """Launch the real-time experiment dashboard (web UI)."""
    _header("AutoTrader Dashboard")
    try:
        import uvicorn
        from src.autotrader.dashboard import create_app, wire_orchestrator
        app = create_app()

        # Wire live orchestrator if pipeline initializes
        try:
            pipe = _pipeline(config=ctx.obj["config"])
            if pipe.autotrader:
                wire_orchestrator(pipe.autotrader)
                _ok("Orchestrator connected (live status enabled)")
        except Exception:
            _warn("Running in view-only mode (no orchestrator)")

        _ok(f"Dashboard at http://localhost:{port}")
        _ok("Real-time updates via SSE + 5s polling fallback")
        click.echo()
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as exc:
        _err(f"Install dependencies: pip install fastapi uvicorn sse-starlette\n  ({exc})")
    except Exception as exc:
        _err(f"Dashboard failed: {exc}")


@autotrader_group.command(name="run-once")
@click.pass_context
def autotrader_run_once(ctx):
    """Run a single experiment iteration (for testing)."""
    _header("AutoTrader — Single Experiment")

    pipe = _pipeline(config=ctx.obj["config"])
    if pipe.autotrader is None:
        _err("AutoTrader not initialised.")
        return

    click.echo("  Running single experiment...")
    result = pipe.autotrader.run_single()

    status_color = {"keep": "green", "discard": "yellow", "crash": "red"}.get(result.status, "white")
    click.echo()
    _kv("Status", click.style(result.status.upper(), fg=status_color))
    _kv("Composite Score", f"{result.composite_score:.6f}")
    _kv("Sharpe Ratio", f"{result.sharpe_ratio:.4f}")
    _kv("Cumulative Return", f"{result.cumulative_return:.4f}")
    _kv("Max Drawdown", f"{result.max_drawdown:.4f}")
    _kv("Hit Rate", f"{result.hit_rate:.4f}")
    _kv("Num Trades", result.num_trades)
    _kv("Duration", f"{result.backtest_duration_seconds:.1f}s")
    if result.error:
        _err(f"Error: {result.error}")
    click.echo()


# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def risk(ctx):
    """Show current risk dashboard."""
    _header("Risk Dashboard")

    pipe = _pipeline(config=ctx.obj["config"])

    # Risk engine status
    try:
        risk_status = pipe.risk_engine.status() if hasattr(pipe.risk_engine, "status") else {}
    except Exception:
        risk_status = {}

    if isinstance(risk_status, dict) and risk_status:
        for k, v in risk_status.items():
            color = "red" if "breach" in str(v).lower() else "green"
            _kv(k, v, color)
    else:
        _kv("Risk State", "OK (no active positions)", "green")

    # Environment limits from config
    env_cfg = pipe.config.get("environment", {})
    click.echo()
    click.echo(click.style("  Risk Limits:", bold=True))
    _kv("Max Drawdown", f"{env_cfg.get('max_drawdown_pct', '?')}%")
    _kv("Max Position Size", f"{env_cfg.get('max_position_size', '?')}")
    _kv("Max Sector Exposure", f"{env_cfg.get('max_sector_exposure', '?')}")
    _kv("Max Leverage", f"{env_cfg.get('max_leverage', '?')}x")

    # Circuit breakers
    breakers = env_cfg.get("circuit_breakers", [])
    if breakers:
        click.echo()
        click.echo(click.style("  Circuit Breakers:", bold=True))
        for cb in breakers:
            click.echo(f"    • {cb.get('type', '?')} @ {cb.get('threshold', '?')}% → {cb.get('action', '?')}")
    click.echo()


# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------
@cli.command()
@click.pass_context
def kill(ctx):
    """Emergency kill switch — cancel all orders and flatten positions."""
    _header("EMERGENCY KILL SWITCH")
    _err("This will CANCEL all open orders and FLATTEN all positions.")
    click.echo()

    if not click.confirm(click.style("  Proceed with emergency shutdown?", fg="red", bold=True)):
        _warn("Aborted.")
        return

    pipe = _pipeline(config=ctx.obj["config"])
    pipe.kill()
    _ok("Kill switch executed. All orders cancelled, positions flattened.")
    click.echo()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    cli(obj={})


if __name__ == "__main__":
    main()
