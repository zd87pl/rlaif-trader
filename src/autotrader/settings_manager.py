"""Persistent settings manager for the AutoTrader dashboard.

Stores API keys and configuration in .env.local (gitignored) so they
survive process restarts. Settings are loaded in priority order:

    1. Environment variables (highest)
    2. .env.local (dashboard-managed, persisted)
    3. .env (user-managed defaults)
    4. Hardcoded defaults (lowest)

Security:
- Keys are masked when read via the API (show only last 4 chars)
- .env.local is chmod 600 (owner-only)
- Never included in git (.gitignore covers .env.local)
"""

from __future__ import annotations

import os
import stat
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Project root: walk up from this file to find the repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_LOCAL = _PROJECT_ROOT / ".env.local"
_ENV_DEFAULT = _PROJECT_ROOT / ".env"

# ── Setting definitions ─────────────────────────────────────────────────

SETTING_DEFS: List[Dict[str, Any]] = [
    # --- API Keys ---
    {
        "key": "ANTHROPIC_API_KEY",
        "label": "Anthropic API Key",
        "group": "API Keys",
        "type": "secret",
        "help": "Required for Claude-powered thesis generation. Get one at console.anthropic.com",
        "required": True,
    },
    {
        "key": "ALPACA_API_KEY",
        "label": "Alpaca API Key",
        "group": "API Keys",
        "type": "secret",
        "help": "For US stock/options market data and trading. alpaca.markets",
    },
    {
        "key": "ALPACA_SECRET_KEY",
        "label": "Alpaca Secret Key",
        "group": "API Keys",
        "type": "secret",
        "help": "Alpaca API secret (paired with the API key above)",
    },
    {
        "key": "ALPACA_BASE_URL",
        "label": "Alpaca Base URL",
        "group": "API Keys",
        "type": "text",
        "default": "https://paper-api.alpaca.markets",
        "help": "paper-api.alpaca.markets for paper, api.alpaca.markets for live",
    },
    {
        "key": "BINANCE_API_KEY",
        "label": "Binance API Key",
        "group": "API Keys",
        "type": "secret",
        "help": "For 24/7 crypto trading. binance.com/en/my/settings/api-management",
    },
    {
        "key": "BINANCE_API_SECRET",
        "label": "Binance API Secret",
        "group": "API Keys",
        "type": "secret",
        "help": "Binance API secret (paired with the key above)",
    },
    {
        "key": "TRADIER_ACCESS_TOKEN",
        "label": "Tradier Access Token",
        "group": "API Keys",
        "type": "secret",
        "help": "For options multi-leg trading. developer.tradier.com",
    },
    # --- Broker ---
    {
        "key": "BROKER",
        "label": "Active Broker",
        "group": "Broker",
        "type": "select",
        "options": ["alpaca", "binance", "tradier", "paper"],
        "default": "paper",
        "help": "Which broker to use for live trading",
    },
    {
        "key": "DATA_SOURCE",
        "label": "Market Data Source",
        "group": "Broker",
        "type": "select",
        "options": ["auto", "ccxt", "alpaca"],
        "default": "auto",
        "help": "Where to pull OHLCV data. 'auto' tries CCXT (free) first, then Alpaca.",
    },
    {
        "key": "CCXT_EXCHANGE",
        "label": "CCXT Exchange",
        "group": "Broker",
        "type": "select",
        "options": ["binance", "bybit", "okx", "kraken", "coinbase", "kucoin"],
        "default": "kraken",
        "help": "Which exchange to pull crypto data from (free, no API key needed)",
    },
    {
        "key": "BINANCE_TESTNET",
        "label": "Binance Testnet",
        "group": "Broker",
        "type": "toggle",
        "default": "true",
        "help": "Use Binance testnet (paper trading). Disable for real money.",
    },
    {
        "key": "BINANCE_FUTURES",
        "label": "Binance Futures",
        "group": "Broker",
        "type": "toggle",
        "default": "false",
        "help": "Use USDT-M futures instead of spot trading",
    },
    # --- LLM ---
    {
        "key": "LLM_BACKEND",
        "label": "LLM Backend",
        "group": "LLM",
        "type": "select",
        "options": ["claude-cli", "claude", "mlx"],
        "default": "claude-cli",
        "help": "claude-cli = uses local claude CLI (no API key), claude = Anthropic API, mlx = local Apple Silicon",
    },
    {
        "key": "MLX_MODEL",
        "label": "MLX Model ID",
        "group": "LLM",
        "type": "text",
        "default": "",
        "help": "Local MLX model for Apple Silicon (optional)",
    },
    # --- Strategy ---
    {
        "key": "RISK_PREFERENCE",
        "label": "Risk Appetite",
        "group": "Strategy",
        "type": "select",
        "options": ["conservative", "moderate", "aggressive"],
        "default": "moderate",
        "help": "How aggressive should the AI advisor be with your capital?",
    },
    {
        "key": "AUTO_REASSESS",
        "label": "Auto-Reassess Strategy",
        "group": "Strategy",
        "type": "toggle",
        "default": "true",
        "help": "Automatically re-evaluate strategy when market conditions change",
    },
    {
        "key": "REASSESS_INTERVAL_MIN",
        "label": "Reassess Interval (minutes)",
        "group": "Strategy",
        "type": "text",
        "default": "60",
        "help": "How often the AI advisor re-evaluates the portfolio strategy (15-480)",
    },
    # --- Runtime ---
    {
        "key": "DEVICE",
        "label": "Compute Device",
        "group": "Runtime",
        "type": "select",
        "options": ["cpu", "cuda", "mps"],
        "default": "cpu",
        "help": "Device for ML inference (cpu, cuda for NVIDIA, mps for Apple Silicon)",
    },
]


class SettingsManager:
    """Read/write persistent settings from .env.local."""

    def __init__(self, env_local_path: Optional[Path] = None):
        self.path = env_local_path or _ENV_LOCAL
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load .env.local into cache."""
        self._cache.clear()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    self._cache[key.strip()] = value.strip()

    def _save(self) -> None:
        """Write cache to .env.local with restrictive permissions."""
        lines = ["# AutoTrader settings (managed by dashboard)", "# DO NOT commit this file\n"]
        for key, value in self._cache.items():
            lines.append(f"{key}={value}")
        self.path.write_text("\n".join(lines) + "\n")
        # chmod 600: owner read/write only
        try:
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        logger.info("Settings saved to %s", self.path)

    def get(self, key: str) -> str:
        """Get a setting value. Priority: env var > .env.local > .env > default."""
        # 1. Real env var (highest priority, but skip empty strings)
        env_val = os.environ.get(key)
        if env_val is not None and env_val != "":
            return env_val
        # 2. .env.local
        if key in self._cache:
            return self._cache[key]
        # 3. .env (read on demand)
        if _ENV_DEFAULT.exists():
            for line in _ENV_DEFAULT.read_text().splitlines():
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip()
        # 4. Default from definition
        for d in SETTING_DEFS:
            if d["key"] == key:
                return d.get("default", "")
        return ""

    def set(self, key: str, value: str) -> None:
        """Set a value in .env.local and save."""
        self._cache[key] = value
        # Also inject into the current process environment
        os.environ[key] = value
        self._save()

    def set_many(self, updates: Dict[str, str]) -> None:
        """Batch update multiple settings."""
        for key, value in updates.items():
            self._cache[key] = value
            os.environ[key] = value
        self._save()

    def delete(self, key: str) -> None:
        """Remove a setting from .env.local."""
        self._cache.pop(key, None)
        self._save()

    def get_all_masked(self) -> List[Dict[str, Any]]:
        """Return all settings with secrets masked for the UI."""
        result = []
        for defn in SETTING_DEFS:
            key = defn["key"]
            raw_value = self.get(key)
            # Determine source
            source = "default"
            if os.environ.get(key):
                source = "env"
            elif key in self._cache:
                source = "local"
            elif _ENV_DEFAULT.exists():
                for line in _ENV_DEFAULT.read_text().splitlines():
                    if line.startswith(key + "=") and line.split("=", 1)[1].strip():
                        source = "dotenv"
                        break

            is_set = bool(raw_value)
            masked = self._mask(raw_value) if defn["type"] == "secret" else raw_value

            result.append({
                **defn,
                "value": masked,
                "is_set": is_set,
                "source": source,
            })
        return result

    def get_grouped(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return settings grouped by category for the UI."""
        all_settings = self.get_all_masked()
        groups: Dict[str, List[Dict[str, Any]]] = OrderedDict()
        for s in all_settings:
            g = s.get("group", "Other")
            if g not in groups:
                groups[g] = []
            groups[g].append(s)
        return groups

    def validate(self) -> List[Dict[str, str]]:
        """Check which required settings are missing."""
        issues = []
        for defn in SETTING_DEFS:
            if defn.get("required") and not self.get(defn["key"]):
                issues.append({
                    "key": defn["key"],
                    "label": defn["label"],
                    "severity": "error",
                    "message": f"{defn['label']} is required but not set",
                })
        return issues

    @staticmethod
    def _mask(value: str) -> str:
        """Mask a secret value, showing only the last 4 chars."""
        if not value:
            return ""
        if len(value) <= 4:
            return "*" * len(value)
        return "*" * (len(value) - 4) + value[-4:]
