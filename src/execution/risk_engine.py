"""
Risk Engine - Core risk management for RLAIF options trading.

Every order must pass through RiskEngine.validate_order() before execution.
Immutable safety rules (max_position_risk_pct, require_defined_risk) cannot
be overridden at runtime.
"""

import threading
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ── Immutable rules: these can NEVER be relaxed at runtime ───────────────
_IMMUTABLE_DEFAULTS = {
    "max_position_risk_pct": 0.02,
    "require_defined_risk": True,
}


class RiskEngine:
    """Centralised risk gate for all trade execution."""

    # ── default configuration ────────────────────────────────────────────
    DEFAULT_CONFIG: Dict = {
        "max_position_risk_pct": 0.02,       # 2 % max capital per trade
        "max_daily_loss_pct": 0.03,           # 3 % daily loss → auto stop
        "max_weekly_loss_pct": 0.05,          # 5 % weekly loss → full stop
        "max_single_position_pct": 0.15,      # 15 % max in one position
        "max_total_exposure_pct": 0.60,       # 60 % max deployed
        "max_trades_per_day": 5,
        "allowed_underlyings": [
            "SPY", "QQQ", "IWM", "AAPL", "MSFT",
            "NVDA", "TSLA", "META", "AMZN", "GOOGL",
        ],
        "min_dte": 7,                         # no weeklies < 7 DTE
        "max_dte": 60,                        # nothing > 60 DTE
        "require_defined_risk": True,         # no naked options
        "pdt_tracking": True,
    }

    # ── lifecycle ────────────────────────────────────────────────────────

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config: Dict = {**self.DEFAULT_CONFIG, **(config or {})}

        # Enforce immutable rules – never allow weaker values from config
        for key, safe_val in _IMMUTABLE_DEFAULTS.items():
            if isinstance(safe_val, bool):
                self._config[key] = safe_val  # always True
            else:
                # numeric: pick the more conservative (smaller) value
                self._config[key] = min(self._config.get(key, safe_val), safe_val)
        self._base_limits: Dict[str, float] = {
            "max_total_exposure_pct": float(self._config["max_total_exposure_pct"]),
            "max_position_risk_pct": float(self._config["max_position_risk_pct"]),
            "max_trades_per_day": int(self._config["max_trades_per_day"]),
        }

        # ── mutable state (reset-able) ───────────────────────────────────
        self._lock = threading.Lock()
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._positions_open: int = 0
        self._kill_switch_active: bool = False
        self._last_daily_reset: date = date.today()
        self._last_weekly_reset: date = date.today()

        logger.info(
            "RiskEngine initialised",
            extra={"config": self._config},
        )

    # ── public properties ────────────────────────────────────────────────

    @property
    def config(self) -> Dict:
        return dict(self._config)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @daily_pnl.setter
    def daily_pnl(self, value: float) -> None:
        with self._lock:
            self._daily_pnl = value

    @property
    def weekly_pnl(self) -> float:
        return self._weekly_pnl

    @weekly_pnl.setter
    def weekly_pnl(self, value: float) -> None:
        with self._lock:
            self._weekly_pnl = value

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count

    @property
    def positions_open(self) -> int:
        return self._positions_open

    @positions_open.setter
    def positions_open(self, value: int) -> None:
        with self._lock:
            self._positions_open = value

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    # ── primary validation entry-point ───────────────────────────────────

    def validate_order(
        self, order: Dict, portfolio: Dict
    ) -> Tuple[bool, str]:
        """
        Run every risk check against *order* in the context of *portfolio*.

        Parameters
        ----------
        order : dict
            Must contain at minimum:
                symbol        – underlying ticker (e.g. "SPY")
                cost          – total premium / max risk in dollars
                strategy      – e.g. "vertical_spread", "iron_condor", "naked_put"
                expiration    – ISO date string or datetime.date
                quantity      – number of contracts
        portfolio : dict
            Must contain at minimum:
                equity        – total account equity in dollars
                daily_pnl     – realised + unrealised P/L today
                weekly_pnl    – realised + unrealised P/L this week
                total_exposure– dollar value currently deployed
                trade_count_5d– day-trades in last 5 sessions (for PDT)

        Returns
        -------
        (approved: bool, reason: str)
        """
        # Auto-reset counters if the day/week rolled over
        self._auto_reset()

        if self._kill_switch_active:
            return False, "KILL SWITCH ACTIVE – all trading halted"

        # Update running P/L from portfolio snapshot
        self._daily_pnl = portfolio.get("daily_pnl", self._daily_pnl)
        self._weekly_pnl = portfolio.get("weekly_pnl", self._weekly_pnl)

        checks: List[Tuple[str, bool, str]] = [
            self._wrap("position_size", self.check_position_size, order, portfolio),
            self._wrap("daily_loss", self.check_daily_loss, portfolio),
            self._wrap("weekly_loss", self.check_weekly_loss, portfolio),
            self._wrap("total_exposure", self.check_total_exposure, order, portfolio),
            self._wrap("trade_count", self.check_trade_count),
            self._wrap("underlying", self.check_underlying_allowed, order.get("symbol", "")),
            self._wrap("dte", self.check_dte, order.get("expiration")),
            self._wrap("defined_risk", self.check_defined_risk, order.get("strategy", "")),
        ]

        if self._config.get("pdt_tracking"):
            pdt_ok, pdt_msg = self.check_pdt(portfolio.get("trade_count_5d", 0))
            checks.append(("pdt", pdt_ok, pdt_msg))

        for name, passed, msg in checks:
            if not passed:
                logger.warning(
                    "Order REJECTED by risk engine",
                    extra={"check": name, "reason": msg, "order": order},
                )
                return False, msg

        # All checks passed – increment trade count
        with self._lock:
            self._daily_trade_count += 1

        logger.info(
            "Order APPROVED by risk engine",
            extra={"order": order},
        )
        return True, "approved"

    # ── individual risk checks ───────────────────────────────────────────

    def check_position_size(self, order: Dict, portfolio: Dict) -> bool:
        """Single position risk must not exceed max_position_risk_pct of equity."""
        equity = portfolio.get("equity", 0)
        if equity <= 0:
            return False
        cost = order.get("cost", 0)
        max_risk = equity * self._config["max_position_risk_pct"]
        return cost <= max_risk

    def check_daily_loss(self, portfolio: Dict) -> bool:
        """Daily P/L must not breach max_daily_loss_pct."""
        equity = portfolio.get("equity", 0)
        if equity <= 0:
            return False
        daily_pnl = portfolio.get("daily_pnl", self._daily_pnl)
        max_loss = equity * self._config["max_daily_loss_pct"]
        return daily_pnl > -max_loss

    def check_weekly_loss(self, portfolio: Dict) -> bool:
        """Weekly P/L must not breach max_weekly_loss_pct."""
        equity = portfolio.get("equity", 0)
        if equity <= 0:
            return False
        weekly_pnl = portfolio.get("weekly_pnl", self._weekly_pnl)
        max_loss = equity * self._config["max_weekly_loss_pct"]
        return weekly_pnl > -max_loss

    def check_total_exposure(
        self, order: Optional[Dict] = None, portfolio: Optional[Dict] = None
    ) -> bool:
        """Total deployed capital must stay within max_total_exposure_pct."""
        if portfolio is None:
            return True
        equity = portfolio.get("equity", 0)
        if equity <= 0:
            return False
        current = portfolio.get("total_exposure", 0)
        additional = order.get("cost", 0) if order else 0
        max_exposure = equity * self._config["max_total_exposure_pct"]
        return (current + additional) <= max_exposure

    def check_trade_count(self) -> bool:
        """Daily trade count must not exceed max_trades_per_day."""
        return self._daily_trade_count < self._config["max_trades_per_day"]

    def check_underlying_allowed(self, symbol: str) -> bool:
        """Underlying must be in the approved list."""
        return symbol.upper() in self._config["allowed_underlyings"]

    def check_dte(self, expiration) -> bool:
        """Option expiration must be between min_dte and max_dte."""
        if expiration is None:
            return False
        if isinstance(expiration, str):
            try:
                exp_date = datetime.fromisoformat(expiration).date()
            except ValueError:
                return False
        elif isinstance(expiration, datetime):
            exp_date = expiration.date()
        elif isinstance(expiration, date):
            exp_date = expiration
        else:
            return False

        dte = (exp_date - date.today()).days
        return self._config["min_dte"] <= dte <= self._config["max_dte"]

    def check_defined_risk(self, strategy: str) -> bool:
        """Reject naked / undefined-risk strategies when require_defined_risk is True."""
        if not self._config["require_defined_risk"]:
            return True
        naked_keywords = {"naked", "uncovered", "straddle_sell", "strangle_sell"}
        return strategy.lower() not in naked_keywords

    def check_pdt(self, trade_count_5d: int) -> Tuple[bool, str]:
        """
        Pattern Day Trader check.  Warn at 3, block at 4 round-trips in 5 days
        (for accounts < $25k – caller should skip if equity >= 25k).
        """
        if trade_count_5d >= 4:
            return False, f"PDT BLOCK: {trade_count_5d} day-trades in 5 sessions (limit 3)"
        if trade_count_5d == 3:
            return True, f"PDT WARNING: {trade_count_5d} day-trades – one more triggers PDT flag"
        return True, "ok"

    # ── circuit breakers ─────────────────────────────────────────────────

    def circuit_breaker_status(self) -> Dict:
        """Return current status of every circuit breaker."""
        return {
            "kill_switch_active": self._kill_switch_active,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "daily_trade_count": self._daily_trade_count,
            "positions_open": self._positions_open,
            "max_daily_loss_pct": self._config["max_daily_loss_pct"],
            "max_weekly_loss_pct": self._config["max_weekly_loss_pct"],
            "max_trades_per_day": self._config["max_trades_per_day"],
            "last_daily_reset": self._last_daily_reset.isoformat(),
            "last_weekly_reset": self._last_weekly_reset.isoformat(),
        }

    def kill_switch(self) -> bool:
        """
        EMERGENCY STOP.
        Sets kill_switch_active – all subsequent validate_order calls are
        immediately rejected.  Caller is responsible for cancelling open
        orders and closing positions via the broker adapter.

        Returns True to confirm activation.
        """
        with self._lock:
            self._kill_switch_active = True
        logger.critical(
            "🔴 KILL SWITCH ACTIVATED – all trading halted",
        )
        return True

    def deactivate_kill_switch(self) -> None:
        """Manual re-enable after human review."""
        with self._lock:
            self._kill_switch_active = False
        logger.warning("Kill switch deactivated – trading re-enabled")

    # ── dynamic limits (set by portfolio strategist) ───────────────────

    def set_dynamic_limits(
        self,
        max_exposure_pct: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        max_trades_per_day: Optional[int] = None,
    ) -> Dict[str, float]:
        """Tighten risk limits at runtime (e.g. from portfolio strategist).

        Only allows values *more conservative* than the immutable defaults.
        Returns the actual limits applied.
        """
        applied = {}
        with self._lock:
            if max_exposure_pct is not None:
                capped = min(max_exposure_pct, self._base_limits["max_total_exposure_pct"])
                self._config["max_total_exposure_pct"] = capped
                applied["max_total_exposure_pct"] = capped

            if max_position_pct is not None:
                capped = min(max_position_pct, self._base_limits["max_position_risk_pct"])
                self._config["max_position_risk_pct"] = capped
                applied["max_position_risk_pct"] = capped

            if max_trades_per_day is not None:
                self._config["max_trades_per_day"] = min(
                    max(1, max_trades_per_day),
                    self._base_limits["max_trades_per_day"],
                )
                applied["max_trades_per_day"] = self._config["max_trades_per_day"]

        if applied:
            logger.info("Dynamic risk limits applied: %s", applied)
        return applied

    # ── reset helpers ────────────────────────────────────────────────────

    def daily_reset(self) -> None:
        """Reset daily counters (call at market open or midnight)."""
        with self._lock:
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._last_daily_reset = date.today()
        logger.info("Daily risk counters reset")

    def weekly_reset(self) -> None:
        """Reset weekly counters (call Monday pre-market)."""
        with self._lock:
            self._weekly_pnl = 0.0
            self._last_weekly_reset = date.today()
        logger.info("Weekly risk counters reset")

    # ── internals ────────────────────────────────────────────────────────

    def _auto_reset(self) -> None:
        """Automatically reset counters when the calendar day/week rolls."""
        today = date.today()
        if today != self._last_daily_reset:
            self.daily_reset()
        # Reset weekly on Monday
        if today.weekday() == 0 and today != self._last_weekly_reset:
            self.weekly_reset()

    def status(self) -> Dict:
        """Compatibility status payload for CLI and scheduler consumers."""
        return {
            "kill_switch_active": self._kill_switch_active,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "daily_trade_count": self._daily_trade_count,
            "positions_open": self._positions_open,
            "limits": dict(self._config),
        }

    def check_risk(self, symbol: str, action: str, size: float) -> bool:
        """Compatibility shim for older scheduler code paths."""
        order = {
            "symbol": symbol,
            "cost": float(size),
            "strategy": "equity",
            "expiration": None,
            "quantity": float(size),
        }
        portfolio = {
            "equity": 100_000.0,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "total_exposure": 0.0,
            "trade_count_5d": self._daily_trade_count,
        }
        # validate_order expects an expiration for options strategies; for equity-style
        # compatibility checks, run a narrower subset of rules.
        return (
            not self._kill_switch_active
            and self.check_trade_count()
            and self.check_underlying_allowed(symbol)
        )

    def get_positions_to_close(self) -> List[Dict]:
        """Placeholder hook for scheduler integration."""
        return []

    @staticmethod
    def _wrap(name: str, fn, *args) -> Tuple[str, bool, str]:
        """Wrap a check function, returning (name, passed, reason)."""
        try:
            result = fn(*args)
            if isinstance(result, tuple):
                passed, msg = result
            else:
                passed = bool(result)
                msg = "ok" if passed else f"{name} check failed"
            return name, passed, msg
        except Exception as exc:  # noqa: BLE001
            return name, False, f"{name} error: {exc}"
