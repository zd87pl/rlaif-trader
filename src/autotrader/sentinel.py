"""Market Sentinel: continuous monitoring for regime changes and anomalies.

Detects events that trigger the experimentation loop:
1. Regime changes (wraps existing RegimeDetector)
2. Volatility spikes
3. Drawdown breaches
4. Correlation breaks
5. Scheduled periodic triggers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    REGIME_CHANGE = "regime_change"
    VOL_SPIKE = "vol_spike"
    DRAWDOWN_BREACH = "drawdown_breach"
    CORRELATION_BREAK = "correlation_break"
    PNL_DEVIATION = "pnl_deviation"
    SCHEDULED = "scheduled"


@dataclass
class MarketEvent:
    """A detected market event that may trigger experimentation."""

    event_type: EventType
    severity: float  # 0.0-1.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        evt = self.event_type.value if hasattr(self.event_type, "value") else str(self.event_type)
        return f"[{evt}] severity={self.severity:.2f} {self.details}"


class MarketSentinel:
    """Monitors market conditions and emits events that trigger experiments.

    Wraps existing RegimeDetector and TechnicalFeatureEngine for detection,
    adding drawdown monitoring and scheduled triggers on top.
    """

    def __init__(
        self,
        regime_detector: Any = None,
        technical_engine: Any = None,
        data_client: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.regime_detector = regime_detector
        self.technical_engine = technical_engine
        self.data_client = data_client

        cfg = config or {}
        self.check_interval = cfg.get("check_interval_seconds", 30)
        self.regime_sensitivity = cfg.get("regime_sensitivity", 0.5)
        self.vol_spike_threshold = cfg.get("vol_spike_threshold", 25)
        self.vol_spike_delta_std = cfg.get("vol_spike_delta_std", 2.0)
        self.drawdown_breach_pct = cfg.get("drawdown_breach_pct", 5.0)
        self.pnl_deviation_zscore = cfg.get("pnl_deviation_zscore", 2.0)
        self.scheduled_interval = cfg.get("scheduled_interval_seconds", 1800)

        # State
        self._previous_regime = None
        self._last_scheduled = 0.0
        self._vol_history: List[float] = []
        self._return_history: List[float] = []

    def check(
        self,
        portfolio_state: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[MarketEvent]:
        """Run all detection channels and return any triggered events.

        Parameters
        ----------
        portfolio_state : dict, optional
            Current portfolio: equity, daily_pnl, unrealized_pnl, positions
        market_data : dict, optional
            Recent OHLCV data per symbol: {symbol: DataFrame}

        Returns
        -------
        list[MarketEvent]
            Events detected in this check cycle.
        """
        events: List[MarketEvent] = []

        # 1. Regime change detection
        regime_event = self._check_regime(market_data)
        if regime_event:
            events.append(regime_event)

        # 2. Volatility spike
        vol_event = self._check_vol_spike(market_data)
        if vol_event:
            events.append(vol_event)

        # 3. Drawdown breach
        dd_event = self._check_drawdown(portfolio_state)
        if dd_event:
            events.append(dd_event)

        # 4. P&L deviation
        pnl_event = self._check_pnl_deviation(portfolio_state)
        if pnl_event:
            events.append(pnl_event)

        # 5. Scheduled trigger
        sched_event = self._check_scheduled()
        if sched_event:
            events.append(sched_event)

        if events:
            logger.info(
                "Sentinel detected %d event(s): %s",
                len(events),
                ", ".join(e.event_type.value for e in events),
            )

        return events

    def _check_regime(
        self, market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Optional[MarketEvent]:
        """Detect regime changes using the existing RegimeDetector."""
        if not self.regime_detector:
            return None

        try:
            # Use the first available symbol's data for regime detection
            if market_data:
                df = next(iter(market_data.values()))
            elif self.data_client:
                df = self.data_client.download_bars(
                    symbols="SPY", start="2024-01-01",
                    end=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    timeframe="1Day",
                )
            else:
                return None

            snapshot = self.regime_detector.detect_regime(df)
            current_regime = snapshot.regime if hasattr(snapshot, "regime") else snapshot

            if self._previous_regime is not None and current_regime != self._previous_regime:
                confidence = getattr(snapshot, "confidence", 0.5)
                event = MarketEvent(
                    event_type=EventType.REGIME_CHANGE,
                    severity=min(1.0, confidence + 0.3),
                    details={
                        "previous_regime": str(self._previous_regime),
                        "current_regime": str(current_regime),
                        "confidence": confidence,
                    },
                )
                self._previous_regime = current_regime
                return event

            self._previous_regime = current_regime

        except Exception as e:
            logger.debug("Regime check failed: %s", e)

        return None

    def _check_vol_spike(
        self, market_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Optional[MarketEvent]:
        """Detect volatility spikes from recent price action."""
        if not market_data:
            return None

        try:
            for symbol, df in market_data.items():
                if "close" not in df.columns or len(df) < 20:
                    continue

                # Realized volatility (20-day annualized)
                returns = df["close"].pct_change().dropna()
                if len(returns) < 20:
                    continue

                current_vol = float(returns.tail(5).std() * np.sqrt(252) * 100)
                avg_vol = float(returns.tail(20).std() * np.sqrt(252) * 100)

                self._vol_history.append(current_vol)
                if len(self._vol_history) > 100:
                    self._vol_history = self._vol_history[-100:]

                if len(self._vol_history) >= 10:
                    vol_mean = np.mean(self._vol_history)
                    vol_std = max(np.std(self._vol_history), 1e-10)
                    z_score = (current_vol - vol_mean) / vol_std

                    if z_score > self.vol_spike_delta_std:
                        return MarketEvent(
                            event_type=EventType.VOL_SPIKE,
                            severity=min(1.0, z_score / 4.0),
                            details={
                                "symbol": symbol,
                                "current_vol": round(current_vol, 2),
                                "avg_vol": round(avg_vol, 2),
                                "z_score": round(z_score, 2),
                            },
                        )
        except Exception as e:
            logger.debug("Vol spike check failed: %s", e)

        return None

    def _check_drawdown(
        self, portfolio_state: Optional[Dict[str, Any]]
    ) -> Optional[MarketEvent]:
        """Detect if current strategy drawdown breaches threshold."""
        if not portfolio_state:
            return None

        equity = portfolio_state.get("equity", 0)
        peak_equity = portfolio_state.get("peak_equity", equity)
        if peak_equity <= 0:
            return None

        drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
        if drawdown_pct > self.drawdown_breach_pct:
            return MarketEvent(
                event_type=EventType.DRAWDOWN_BREACH,
                severity=min(1.0, drawdown_pct / (self.drawdown_breach_pct * 2)),
                details={
                    "drawdown_pct": round(drawdown_pct, 2),
                    "threshold_pct": self.drawdown_breach_pct,
                    "equity": equity,
                    "peak_equity": peak_equity,
                },
            )
        return None

    def _check_pnl_deviation(
        self, portfolio_state: Optional[Dict[str, Any]]
    ) -> Optional[MarketEvent]:
        """Detect significant P&L deviation from expected performance."""
        if not portfolio_state:
            return None

        daily_pnl = portfolio_state.get("daily_pnl", 0)
        expected_daily = portfolio_state.get("expected_daily_pnl", 0)
        equity = portfolio_state.get("equity", 1)

        if equity <= 0:
            return None

        daily_return = daily_pnl / equity
        self._return_history.append(daily_return)
        if len(self._return_history) > 60:
            self._return_history = self._return_history[-60:]

        if len(self._return_history) >= 10:
            mean_return = np.mean(self._return_history)
            std_return = max(np.std(self._return_history), 1e-10)
            z_score = (daily_return - mean_return) / std_return

            if abs(z_score) > self.pnl_deviation_zscore:
                return MarketEvent(
                    event_type=EventType.PNL_DEVIATION,
                    severity=min(1.0, abs(z_score) / 4.0),
                    details={
                        "daily_return": round(daily_return, 6),
                        "mean_return": round(mean_return, 6),
                        "z_score": round(z_score, 2),
                        "direction": "positive" if z_score > 0 else "negative",
                    },
                )
        return None

    def update_timing(
        self,
        check_interval: Optional[int] = None,
        scheduled_interval: Optional[int] = None,
    ) -> None:
        """Update timing parameters at runtime (called by strategist)."""
        if check_interval is not None:
            self.check_interval = max(5, check_interval)
        if scheduled_interval is not None:
            self.scheduled_interval = max(60, scheduled_interval)
        logger.info(
            "Sentinel timing updated: check=%ds, scheduled=%ds",
            self.check_interval, self.scheduled_interval,
        )

    def _check_scheduled(self) -> Optional[MarketEvent]:
        """Emit periodic scheduled triggers regardless of market conditions."""
        now = time.time()
        if now - self._last_scheduled >= self.scheduled_interval:
            self._last_scheduled = now
            return MarketEvent(
                event_type=EventType.SCHEDULED,
                severity=0.3,
                details={"interval_seconds": self.scheduled_interval},
            )
        return None
