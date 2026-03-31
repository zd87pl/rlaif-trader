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
