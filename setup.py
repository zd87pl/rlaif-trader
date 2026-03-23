#!/usr/bin/env python3
"""
Interactive setup for the RLAIF Trading Pipeline.

Walks you through everything: Python env, API keys, broker choice,
strategy config, and validation. Run once, trade forever.

Usage:
    python setup.py
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Colors & formatting
# ─────────────────────────────────────────────────────────────────────────────

class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


def banner():
    print(f"""{C.CYAN}{C.BOLD}
    ╔══════════════════════════════════════════════════════════╗
    ║           RLAIF TRADING PIPELINE — SETUP                ║
    ║                                                          ║
    ║   Multi-strategy AI trading with Claude, Alpaca, IBKR    ║
    ╚══════════════════════════════════════════════════════════╝{C.RESET}
    """)


def section(title):
    print(f"\n{C.BOLD}{C.MAGENTA}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}{'─' * 60}{C.RESET}\n")


def ok(msg):
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def warn(msg):
    print(f"  {C.YELLOW}!{C.RESET} {msg}")


def fail(msg):
    print(f"  {C.RED}✗{C.RESET} {msg}")


def info(msg):
    print(f"  {C.DIM}{msg}{C.RESET}")


def ask(prompt, default="", secret=False, required=False):
    """Prompt user for input."""
    suffix = f" [{default}]" if default else ""
    if secret:
        suffix += " (input hidden)"

    while True:
        try:
            if secret:
                import getpass
                value = getpass.getpass(f"  {C.CYAN}>{C.RESET} {prompt}{suffix}: ")
            else:
                value = input(f"  {C.CYAN}>{C.RESET} {prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not value and default:
            return default
        if not value and required:
            fail("This field is required.")
            continue
        if value or not required:
            return value


def choose(prompt, options, default=None):
    """Let user pick from numbered options."""
    print(f"  {prompt}")
    for i, (label, desc) in enumerate(options, 1):
        marker = f" {C.GREEN}(default){C.RESET}" if default and label == default else ""
        print(f"    {C.BOLD}{i}{C.RESET}) {label}{C.DIM} — {desc}{C.RESET}{marker}")

    while True:
        raw = ask("Choice", default=str(next(
            (i for i, (l, _) in enumerate(options, 1) if l == default), 1
        )))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            # Maybe they typed the label
            for label, _ in options:
                if raw.lower() == label.lower():
                    return label
        fail(f"Pick 1-{len(options)}")


def yesno(prompt, default=True):
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    raw = ask(f"{prompt} [{hint}]")
    if not raw:
        return default
    return raw.lower().startswith("y")


# ─────────────────────────────────────────────────────────────────────────────
# Setup steps
# ─────────────────────────────────────────────────────────────────────────────

class Setup:
    def __init__(self):
        self.root = Path(__file__).parent.resolve()
        self.env_path = self.root / ".env"
        self.env = {}  # Will hold all env vars to write
        self.config = {}  # Will hold setup config summary

    def run(self):
        banner()

        self.step_python_check()
        self.step_install_deps()
        self.step_broker_choice()
        self.step_api_keys()
        self.step_trading_config()
        self.step_strategy_config()
        self.step_write_env()
        self.step_create_dirs()
        self.step_validate()
        self.step_summary()

    # ── Step 1: Python ────────────────────────────────────────────────────

    def step_python_check(self):
        section("1/9  PYTHON ENVIRONMENT")

        ver = sys.version_info
        if ver >= (3, 11):
            ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")
        elif ver >= (3, 10):
            warn(f"Python {ver.major}.{ver.minor} — 3.11+ recommended")
        else:
            fail(f"Python {ver.major}.{ver.minor} — need 3.11+")
            if not yesno("Continue anyway?", default=False):
                sys.exit(1)

        # Check if in a venv
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        if in_venv:
            ok(f"Virtual environment active: {sys.prefix}")
        else:
            warn("No virtual environment detected.")
            if yesno("Create one now?"):
                venv_path = self.root / "venv"
                print(f"  Creating venv at {venv_path}...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                ok(f"Virtual environment created at {venv_path}")
                # Determine activation command
                if os.name == "nt":
                    activate = f"{venv_path}\\Scripts\\activate"
                else:
                    activate = f"source {venv_path}/bin/activate"
                print(f"\n  {C.YELLOW}Activate it and re-run setup:{C.RESET}")
                print(f"    {C.BOLD}{activate}{C.RESET}")
                print(f"    {C.BOLD}python setup.py{C.RESET}")
                sys.exit(0)

    # ── Step 2: Dependencies ──────────────────────────────────────────────

    def step_install_deps(self):
        section("2/9  DEPENDENCIES")

        install_mode = choose(
            "Install mode:",
            [
                ("minimal", "Core trading only (fastest)"),
                ("full", "All features including dev tools"),
                ("skip", "Skip — I'll install manually"),
            ],
            default="minimal",
        )

        if install_mode == "skip":
            info("Skipping dependency installation.")
            return

        cmd = [sys.executable, "-m", "pip", "install", "-e"]
        if install_mode == "full":
            cmd.append(".[dev]")
        else:
            cmd.append(".")

        print(f"\n  Running: {' '.join(cmd)}")
        print(f"  {C.DIM}This may take a few minutes...{C.RESET}\n")

        result = subprocess.run(cmd, cwd=str(self.root), capture_output=True, text=True)
        if result.returncode == 0:
            ok("Dependencies installed")
        else:
            warn("Some dependencies may have failed (non-critical ones often do)")
            info("You can check with: pip install -e . 2>&1 | tail -20")

    # ── Step 3: Broker ────────────────────────────────────────────────────

    def step_broker_choice(self):
        section("3/9  BROKER")

        print(textwrap.dedent(f"""\
          {C.DIM}Your broker executes trades. You need at least one.
          Alpaca is free and has paper trading. IBKR is for serious accounts.{C.RESET}
        """))

        self.config["broker"] = choose(
            "Primary broker:",
            [
                ("alpaca", "Alpaca — Free paper trading, US equities"),
                ("ibkr", "Interactive Brokers — Paper + live, global markets"),
                ("both", "Both — Use Alpaca for data, IBKR for execution"),
                ("none", "None yet — I'll configure later"),
            ],
            default="alpaca",
        )

        if self.config["broker"] in ("ibkr", "both"):
            if not yesno("Do you have ib_insync installed?"):
                print(f"  Installing ib_insync...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "ib_insync"],
                    capture_output=True,
                )
                ok("ib_insync installed")

            self.env["IBKR_PORT"] = ask(
                "IBKR TWS/Gateway port",
                default="7497",  # Paper trading default
            )

    # ── Step 4: API Keys ──────────────────────────────────────────────────

    def step_api_keys(self):
        section("4/9  API KEYS")

        # Alpaca
        if self.config["broker"] in ("alpaca", "both", "none"):
            print(f"  {C.BOLD}Alpaca{C.RESET} {C.DIM}(free at https://alpaca.markets){C.RESET}")
            print(f"  {C.DIM}Required for market data even if using IBKR for trading.{C.RESET}")
            self.env["ALPACA_API_KEY"] = ask("Alpaca API Key", secret=True)
            self.env["ALPACA_SECRET_KEY"] = ask("Alpaca Secret Key", secret=True)
            self.env["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"
            if self.env["ALPACA_API_KEY"]:
                ok("Alpaca keys set")
            else:
                warn("Alpaca keys skipped — you'll need them for market data")

        # Anthropic
        print(f"\n  {C.BOLD}Anthropic Claude{C.RESET} {C.DIM}(for AI agent strategy){C.RESET}")
        print(f"  {C.DIM}Optional — system works without it using technical strategies only.{C.RESET}")
        self.env["ANTHROPIC_API_KEY"] = ask("Anthropic API Key (Enter to skip)", secret=True)
        if self.env["ANTHROPIC_API_KEY"]:
            ok("Anthropic key set")

            model = choose(
                "Claude model:",
                [
                    ("claude-sonnet-4-20250514", "Sonnet 4 — Best balance of speed & quality"),
                    ("claude-opus-4-20250514", "Opus 4 — Most capable, slower, more expensive"),
                    ("claude-haiku-4-5-20251001", "Haiku 4.5 — Fastest, cheapest"),
                ],
                default="claude-sonnet-4-20250514",
            )
            self.env["LLM_MODEL"] = model
        else:
            info("No Claude key — agent strategy will be disabled")
            self.env["LLM_MODEL"] = "claude-sonnet-4-20250514"

        # Finnhub
        print(f"\n  {C.BOLD}Finnhub{C.RESET} {C.DIM}(free at https://finnhub.io — news & sentiment){C.RESET}")
        self.env["FINNHUB_API_KEY"] = ask("Finnhub API Key (Enter to skip)", secret=True)
        if self.env["FINNHUB_API_KEY"]:
            ok("Finnhub key set — news + social sentiment enabled")

        # Polygon
        print(f"\n  {C.BOLD}Polygon{C.RESET} {C.DIM}(https://polygon.io — additional news source){C.RESET}")
        self.env["POLYGON_API_KEY"] = ask("Polygon API Key (Enter to skip)", secret=True)
        if self.env["POLYGON_API_KEY"]:
            ok("Polygon key set")

        # Free sources
        print()
        ok("yfinance fundamentals — enabled (no key needed)")
        ok("SEC EDGAR filings — enabled (no key needed)")

    # ── Step 5: Trading config ────────────────────────────────────────────

    def step_trading_config(self):
        section("5/9  TRADING PARAMETERS")

        print(f"  {C.DIM}These control risk. Conservative defaults are safe for paper trading.{C.RESET}\n")

        self.env["INITIAL_CAPITAL"] = ask("Starting capital ($)", default="100000")
        self.env["MAX_POSITION_SIZE"] = ask(
            "Max position size (% of portfolio)", default="10"
        )
        self.env["MAX_DRAWDOWN_PCT"] = ask(
            "Circuit breaker — max daily loss (%)", default="3"
        )

        max_pos = ask("Max simultaneous positions", default="10")
        self.config["max_positions"] = max_pos

        # Symbols
        print(f"\n  {C.BOLD}Trading universe{C.RESET}")
        default_symbols = "AAPL,MSFT,NVDA,GOOGL,AMZN,TSLA,META,JPM,V,WMT"
        symbols = ask("Symbols (comma-separated)", default=default_symbols)
        self.env["DEFAULT_ASSETS"] = symbols
        self.config["symbols"] = [s.strip() for s in symbols.split(",")]

        ok(f"Trading {len(self.config['symbols'])} symbols, "
           f"${self.env['INITIAL_CAPITAL']} capital, "
           f"{self.env['MAX_POSITION_SIZE']}% max position")

    # ── Step 6: Strategy config ───────────────────────────────────────────

    def step_strategy_config(self):
        section("6/9  STRATEGY CONFIGURATION")

        print(textwrap.dedent(f"""\
          {C.DIM}The system runs multiple strategies simultaneously and combines
          them into one signal. More strategies = more signal, less noise.{C.RESET}
        """))

        # Strategies
        strategies = []

        if yesno("Enable Momentum strategy (trend following)?"):
            strategies.append("momentum")
            ok("Momentum — enabled")
        else:
            info("Momentum — disabled")

        if yesno("Enable Mean Reversion strategy (buy dips, sell rips)?"):
            strategies.append("mean_reversion")
            ok("Mean Reversion — enabled")
        else:
            info("Mean Reversion — disabled")

        if self.env.get("ANTHROPIC_API_KEY"):
            if yesno("Enable Agent strategy (Claude AI — costs ~$0.05/signal)?"):
                strategies.append("agent")
                ok("Agent (Claude) — enabled")
            else:
                info("Agent — disabled (saves API costs)")
        else:
            info("Agent — disabled (no Anthropic key)")

        if not strategies:
            warn("No strategies selected. Enabling momentum as default.")
            strategies = ["momentum"]

        self.config["strategies"] = strategies

        # Ensemble mode
        if len(strategies) > 1:
            print()
            self.config["ensemble"] = choose(
                "How should strategies be combined?",
                [
                    ("conviction_weighted", "Only trade when strategies agree (safest)"),
                    ("majority_vote", "Majority action wins"),
                    ("weighted_average", "Weighted average of all scores"),
                ],
                default="conviction_weighted",
            )
            ok(f"Ensemble mode: {self.config['ensemble']}")
        else:
            self.config["ensemble"] = "weighted_average"

    # ── Step 7: Write .env ────────────────────────────────────────────────

    def step_write_env(self):
        section("7/9  WRITING CONFIGURATION")

        # Populate remaining defaults
        defaults = {
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
            "DEVICE": self._detect_device(),
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_TOKENS": "4096",
            "LOG_LEVEL": "INFO",
            "LOG_DIR": "./logs",
            "LOG_FORMAT": "json",
            "DATA_DIR": "./historical_data",
            "CACHE_DIR": "./data_cache",
            "BAR_INTERVAL": "1Day",
            "LOOKBACK_DAYS": "365",
            "ENVIRONMENT": "development",
            "SECRET_KEY": os.urandom(32).hex(),
        }

        for key, val in defaults.items():
            if key not in self.env or not self.env[key]:
                self.env[key] = val

        # Remove empty keys
        self.env = {k: v for k, v in self.env.items() if v}

        # Check for existing .env
        if self.env_path.exists():
            if not yesno(f".env already exists. Overwrite?", default=False):
                backup = self.env_path.with_suffix(".env.backup")
                shutil.copy(self.env_path, backup)
                ok(f"Backed up to {backup}")

        # Write .env
        lines = [
            "# RLAIF Trading Pipeline Configuration",
            f"# Generated by setup.py on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# API Keys",
        ]

        api_keys = ["ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                     "ALPACA_BASE_URL", "FINNHUB_API_KEY", "POLYGON_API_KEY"]
        for key in api_keys:
            if key in self.env:
                lines.append(f"{key}={self.env[key]}")
        lines.append("")

        lines.append("# Model Configuration")
        model_keys = ["LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "DEVICE"]
        for key in model_keys:
            if key in self.env:
                lines.append(f"{key}={self.env[key]}")
        lines.append("")

        lines.append("# Trading Configuration")
        trading_keys = ["INITIAL_CAPITAL", "MAX_POSITION_SIZE", "MAX_DRAWDOWN_PCT",
                        "DEFAULT_ASSETS"]
        for key in trading_keys:
            if key in self.env:
                lines.append(f"{key}={self.env[key]}")
        lines.append("")

        lines.append("# Data & Logging")
        misc_keys = ["DATA_DIR", "CACHE_DIR", "LOG_LEVEL", "LOG_DIR", "LOG_FORMAT",
                      "BAR_INTERVAL", "LOOKBACK_DAYS", "ENVIRONMENT", "SECRET_KEY"]
        for key in misc_keys:
            if key in self.env:
                lines.append(f"{key}={self.env[key]}")

        # Add any remaining keys not yet written
        written = set(api_keys + model_keys + trading_keys + misc_keys)
        remaining = {k: v for k, v in self.env.items() if k not in written}
        if remaining:
            lines.append("")
            lines.append("# Additional Configuration")
            for k, v in remaining.items():
                lines.append(f"{k}={v}")

        self.env_path.write_text("\n".join(lines) + "\n")
        ok(f".env written to {self.env_path}")

        # Write setup config for reference
        config_summary = self.root / "data" / "setup_config.json"
        config_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(config_summary, "w") as f:
            json.dump(self.config, f, indent=2)

    # ── Step 8: Create directories ────────────────────────────────────────

    def step_create_dirs(self):
        section("8/9  CREATING DIRECTORIES")

        dirs = [
            "logs",
            "logs/audit",
            "data",
            "data/portfolio",
            "data/rlaif",
            "data/rlaif/positions",
            "data_cache",
            "historical_data",
            "models/checkpoints",
        ]

        for d in dirs:
            path = self.root / d
            path.mkdir(parents=True, exist_ok=True)

        ok(f"Created {len(dirs)} directories")

    # ── Step 9: Validate ──────────────────────────────────────────────────

    def step_validate(self):
        section("9/9  VALIDATION")

        # Check key imports
        checks = [
            ("pandas", "Core data processing"),
            ("numpy", "Numerical computing"),
            ("dotenv", "Environment loading"),
        ]

        for module, desc in checks:
            try:
                __import__(module)
                ok(f"{module} — {desc}")
            except ImportError:
                fail(f"{module} not installed — {desc}")

        # Check optional imports
        optional = [
            ("anthropic", "Claude AI agents"),
            ("alpaca.trading.client", "Alpaca broker"),
            ("yfinance", "Fundamental data (free)"),
            ("torch", "PyTorch (ML models)"),
            ("transformers", "FinBERT sentiment model"),
        ]

        for module, desc in optional:
            try:
                __import__(module)
                ok(f"{module} — {desc}")
            except ImportError:
                warn(f"{module} not available — {desc}")

        # Validate API keys
        print()
        if self.env.get("ALPACA_API_KEY") and self.env["ALPACA_API_KEY"] not in ("", "PKxxx"):
            try:
                from alpaca.trading.client import TradingClient
                client = TradingClient(
                    api_key=self.env["ALPACA_API_KEY"],
                    secret_key=self.env.get("ALPACA_SECRET_KEY", ""),
                    paper=True,
                )
                account = client.get_account()
                ok(f"Alpaca connected — equity: ${float(account.equity):,.2f}")
            except Exception as e:
                fail(f"Alpaca connection failed: {e}")
        else:
            info("Alpaca — not configured, skipping validation")

        if self.env.get("ANTHROPIC_API_KEY") and self.env["ANTHROPIC_API_KEY"] not in ("", "sk-ant-xxx"):
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=self.env["ANTHROPIC_API_KEY"])
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Say OK"}],
                )
                ok(f"Anthropic connected — model: {self.env.get('LLM_MODEL', 'default')}")
            except Exception as e:
                fail(f"Anthropic connection failed: {e}")
        else:
            info("Anthropic — not configured, skipping validation")

        if self.env.get("FINNHUB_API_KEY"):
            try:
                import requests
                resp = requests.get(
                    "https://finnhub.io/api/v1/news",
                    params={"category": "general", "token": self.env["FINNHUB_API_KEY"]},
                    timeout=5,
                )
                if resp.status_code == 200:
                    ok(f"Finnhub connected — {len(resp.json())} news articles")
                else:
                    fail(f"Finnhub returned status {resp.status_code}")
            except Exception as e:
                fail(f"Finnhub connection failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────

    def step_summary(self):
        print(f"\n{C.BOLD}{C.GREEN}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}{C.GREEN}  SETUP COMPLETE{C.RESET}")
        print(f"{C.BOLD}{C.GREEN}{'═' * 60}{C.RESET}\n")

        print(f"  {C.BOLD}Configuration:{C.RESET}")
        print(f"    Broker:      {self.config.get('broker', 'none')}")
        print(f"    Strategies:  {', '.join(self.config.get('strategies', []))}")
        print(f"    Ensemble:    {self.config.get('ensemble', 'n/a')}")
        print(f"    Symbols:     {len(self.config.get('symbols', []))} stocks")
        print(f"    Capital:     ${self.env.get('INITIAL_CAPITAL', '100000')}")
        print(f"    Device:      {self.env.get('DEVICE', 'cpu')}")

        data_sources = []
        if self.env.get("ALPACA_API_KEY"):
            data_sources.append("Alpaca (prices)")
        data_sources.append("yfinance (fundamentals)")
        data_sources.append("SEC EDGAR (filings)")
        if self.env.get("FINNHUB_API_KEY"):
            data_sources.append("Finnhub (news+sentiment)")
        if self.env.get("POLYGON_API_KEY"):
            data_sources.append("Polygon (news)")
        print(f"    Data:        {', '.join(data_sources)}")

        strats = self.config.get("strategies", [])
        has_agents = "agent" in strats

        print(f"\n  {C.BOLD}What to do next:{C.RESET}\n")

        print(f"    {C.CYAN}1. Backtest (free, no risk):{C.RESET}")
        strat_arg = " ".join(f"--strategies {s}" for s in strats if s != "agent")
        if strat_arg:
            print(f"       python run_backtest.py --symbols AAPL MSFT {strat_arg}")
        else:
            print(f"       python run_backtest.py --symbols AAPL MSFT")

        if has_agents:
            print(f"\n    {C.CYAN}2. Backtest with agents (costs API credits):{C.RESET}")
            print(f"       python run_backtest.py --symbols AAPL --use-agents")

        print(f"\n    {C.CYAN}{'3' if has_agents else '2'}. Paper trade:{C.RESET}")
        if self.config.get("broker") in ("alpaca", "both"):
            no_agents = " --no-agents" if not has_agents else ""
            print(f"       python run_trading.py --mode alpaca_paper --symbols AAPL MSFT{no_agents}")
        elif self.config.get("broker") == "ibkr":
            print(f"       python run_trading.py --mode ibkr_paper --symbols AAPL MSFT")

        print(f"\n    {C.CYAN}{'4' if has_agents else '3'}. Dry run (signals only, no trades):{C.RESET}")
        print(f"       python run_trading.py --mode dry_run --symbols AAPL MSFT NVDA")

        print(f"\n  {C.DIM}Config saved to .env — edit anytime.{C.RESET}")
        print()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _detect_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                ok(f"CUDA GPU detected: {gpu_name}")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                ok("Apple Silicon GPU detected")
                return "mps"
        except ImportError:
            pass
        info("No GPU detected — using CPU")
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        setup = Setup()
        setup.run()
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}Setup interrupted. Run again anytime with: python setup.py{C.RESET}\n")
        sys.exit(0)
