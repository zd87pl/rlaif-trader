"""
Trading Scheduler / Daemon

Orchestrates the full trading day lifecycle:
  - Pre-market analysis
  - Market open signal execution
  - Intraday position monitoring
  - End-of-day routines
  - Post-market RLAIF updates
  - Weekly reward model retraining

Designed to run as a long-lived daemon process with graceful shutdown
and thread-safe state management.
"""

import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pytz

from ..utils.logging import get_logger

logger = get_logger(__name__)

# NYSE market holidays (fixed dates, approximate for current year)
NYSE_HOLIDAYS_2025 = [
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
]

NYSE_HOLIDAYS_2026 = [
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
]

ET = pytz.timezone("US/Eastern")


def _is_market_holiday(dt: datetime) -> bool:
    """Check if a date is an NYSE holiday."""
    date_str = dt.strftime("%Y-%m-%d")
    return date_str in NYSE_HOLIDAYS_2025 or date_str in NYSE_HOLIDAYS_2026


def _is_trading_day(dt: datetime) -> bool:
    """Check if a date is a valid trading day (weekday + not holiday)."""
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return not _is_market_holiday(dt)


def _now_et() -> datetime:
    """Get current time in Eastern Time."""
    return datetime.now(ET)


class TradingScheduler:
    """
    Main trading daemon that schedules and runs all trading operations.

    Manages the full lifecycle:
      09:00 ET  Pre-market analysis
      09:30 ET  Market open - execute signals
      10:00-15:30 ET  Monitor positions every 15 min
      15:45 ET  Close flagged positions
      16:00 ET  EOD P&L + summary
      16:30 ET  Post-market RLAIF update
      Sunday 18:00 ET  Weekly reward model retrain
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pipeline=None,
        oms=None,
        risk_engine=None,
        alert_manager=None,
    ):
        """
        Args:
            config: Scheduler configuration dict. Keys:
                - symbols: List[str] of symbols to trade
                - max_positions: int
                - risk_limits: Dict
                - retrain_enabled: bool
                - dry_run: bool (paper trade mode)
            pipeline: Multi-agent analysis pipeline (callable or object with .run())
            oms: Order Management System (object with .submit_order(), .cancel_order(), etc.)
            risk_engine: Risk engine (object with .check_risk(), .get_exposure(), etc.)
            alert_manager: Alert/notification manager (object with .send_alert())
        """
        self.config = config
        self.pipeline = pipeline
        self.oms = oms
        self.risk_engine = risk_engine
        self.alert_manager = alert_manager

        # State
        self._lock = threading.RLock()
        self._running = False
        self._paused = False
        self._shutdown_event = threading.Event()
        self._last_run_times: Dict[str, Optional[datetime]] = {
            "pre_market": None,
            "market_open": None,
            "monitor": None,
            "close_flagged": None,
            "eod": None,
            "post_market": None,
            "weekly_retrain": None,
        }

        # Extracted config
        self._symbols: List[str] = config.get("symbols", [])
        self._dry_run: bool = config.get("dry_run", True)
        self._retrain_enabled: bool = config.get("retrain_enabled", True)
        self._monitor_interval_min: int = config.get("monitor_interval_min", 15)

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"TradingScheduler initialized: {len(self._symbols)} symbols, "
            f"dry_run={self._dry_run}"
        )

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def last_run_times(self) -> Dict[str, Optional[datetime]]:
        with self._lock:
            return dict(self._last_run_times)

    def pause(self) -> None:
        """Pause scheduling (finish current task then idle)."""
        with self._lock:
            self._paused = True
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume scheduling."""
        with self._lock:
            self._paused = False
        logger.info("Scheduler resumed")

    # ==================================================================================
    # Main loop
    # ==================================================================================

    def run(self) -> None:
        """
        Main daemon loop. Runs forever until SIGINT/SIGTERM or stop() is called.

        Polls every 30 seconds, checks what tasks should run based on current ET time,
        and dispatches accordingly. Skips non-trading days for market tasks.
        """
        with self._lock:
            self._running = True

        logger.info("TradingScheduler starting main loop")

        try:
            while not self._shutdown_event.is_set():
                try:
                    self._tick()
                except Exception as e:
                    logger.error(f"Error in scheduler tick: {e}", exc_info=True)
                    self._send_alert(f"Scheduler error: {e}", level="error")

                # Sleep in small increments so shutdown is responsive
                self._shutdown_event.wait(timeout=30)
        finally:
            with self._lock:
                self._running = False
            logger.info("TradingScheduler stopped")

    def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("Scheduler shutdown requested")
        self._shutdown_event.set()

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")
        self.stop()

    def _tick(self) -> None:
        """Single scheduler tick - check time and dispatch tasks."""
        with self._lock:
            if self._paused:
                return

        now = _now_et()
        today_is_trading = _is_trading_day(now)
        current_time = now.time()

        # ------------------------------------------------------------------
        # Trading-day tasks
        # ------------------------------------------------------------------
        if today_is_trading:
            # 09:00 Pre-market analysis
            if self._should_run("pre_market", now, target_hour=9, target_minute=0):
                self._run_task("pre_market", self._do_pre_market)

            # 09:30 Market open - execute signals
            if self._should_run("market_open", now, target_hour=9, target_minute=30):
                self._run_task("market_open", self._do_market_open)

            # 10:00-15:30 Monitor every N minutes
            if (
                current_time >= datetime.strptime("10:00", "%H:%M").time()
                and current_time <= datetime.strptime("15:30", "%H:%M").time()
            ):
                last_monitor = self._last_run_times.get("monitor")
                if last_monitor is None or (
                    now - last_monitor
                ).total_seconds() >= self._monitor_interval_min * 60:
                    self._run_task("monitor", self._do_monitor)

            # 15:45 Close flagged positions
            if self._should_run("close_flagged", now, target_hour=15, target_minute=45):
                self._run_task("close_flagged", self._do_close_flagged)

            # 16:00 End of day
            if self._should_run("eod", now, target_hour=16, target_minute=0):
                self._run_task("eod", self._do_eod)

            # 16:30 Post-market RLAIF update
            if self._should_run("post_market", now, target_hour=16, target_minute=30):
                self._run_task("post_market", self._do_post_market)

        # ------------------------------------------------------------------
        # Weekly tasks (Sunday 18:00 ET)
        # ------------------------------------------------------------------
        if now.weekday() == 6:  # Sunday
            if self._should_run("weekly_retrain", now, target_hour=18, target_minute=0):
                if self._retrain_enabled:
                    self._run_task("weekly_retrain", self._do_weekly_retrain)

    def _should_run(
        self,
        task_name: str,
        now: datetime,
        target_hour: int,
        target_minute: int,
    ) -> bool:
        """
        Check if a task should run now.

        Returns True if:
          - Current time is within a 10-minute window after target time
          - Task hasn't already run today (or this week for weekly tasks)
        """
        current_time = now.time()
        target = datetime.strptime(f"{target_hour}:{target_minute}", "%H:%M").time()
        # Window: target to target + 10 min
        window_end = (
            datetime.combine(datetime.today(), target) + timedelta(minutes=10)
        ).time()

        if not (target <= current_time <= window_end):
            return False

        last = self._last_run_times.get(task_name)
        if last is None:
            return True

        # For weekly tasks, check it hasn't run this week
        if task_name == "weekly_retrain":
            return (now - last).days >= 6

        # For daily tasks, check it hasn't run today
        return last.date() < now.date()

    def _run_task(self, task_name: str, func: Callable) -> None:
        """Execute a task with locking and bookkeeping."""
        logger.info(f"Running scheduled task: {task_name}")
        start = time.time()
        try:
            func()
            elapsed = time.time() - start
            with self._lock:
                self._last_run_times[task_name] = _now_et()
            logger.info(f"Task {task_name} completed in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}", exc_info=True)
            self._send_alert(f"Task {task_name} failed: {e}", level="error")

    # ==================================================================================
    # Scheduled task implementations
    # ==================================================================================

    def _do_pre_market(self) -> None:
        """09:00 ET - Pre-market analysis."""
        results = self.pre_market_analysis(self._symbols)
        logger.info(f"Pre-market analysis generated {len(results)} signals")

        # Store signals for market open execution
        with self._lock:
            self._pending_signals = results

    def _do_market_open(self) -> None:
        """09:30 ET - Execute top signals."""
        with self._lock:
            signals = getattr(self, "_pending_signals", [])

        if not signals:
            logger.info("No signals to execute at market open")
            return

        results = self.execute_signals(signals)
        logger.info(f"Executed {len(results)} orders at market open")

    def _do_monitor(self) -> None:
        """10:00-15:30 ET - Monitor positions."""
        status = self.monitor_positions()
        logger.info(
            f"Position monitor: {status.get('total_positions', 0)} positions, "
            f"unrealized P&L: ${status.get('total_unrealized_pnl', 0):.2f}"
        )

    def _do_close_flagged(self) -> None:
        """15:45 ET - Close positions flagged for exit."""
        if self.oms is None:
            logger.warning("No OMS configured, skipping close_flagged")
            return

        try:
            # Get positions flagged for exit by risk engine or stop-loss
            flagged = []
            if self.risk_engine is not None:
                flagged = self.risk_engine.get_positions_to_close()
            for pos in flagged:
                logger.info(f"Closing flagged position: {pos.get('symbol', '?')}")
                if not self._dry_run:
                    self.oms.submit_order(
                        symbol=pos["symbol"],
                        side="sell" if pos.get("side") == "long" else "buy",
                        qty=pos.get("quantity", 0),
                        order_type="market",
                    )
        except Exception as e:
            logger.error(f"Error closing flagged positions: {e}", exc_info=True)

    def _do_eod(self) -> None:
        """16:00 ET - End of day routine."""
        summary = self.end_of_day_routine()
        self._send_alert(
            f"EOD Summary: P&L=${summary.get('daily_pnl', 0):.2f}, "
            f"Trades={summary.get('trades_executed', 0)}",
            level="info",
        )

    def _do_post_market(self) -> None:
        """16:30 ET - Post-market RLAIF outcome updates."""
        try:
            if self.pipeline is not None and hasattr(self.pipeline, "outcome_tracker"):
                tracker = self.pipeline.outcome_tracker
                tracker.update_positions()
                logger.info("Post-market: updated RLAIF outcomes")
            else:
                logger.info("Post-market: no outcome tracker configured")
        except Exception as e:
            logger.error(f"Post-market RLAIF update failed: {e}", exc_info=True)

    def _do_weekly_retrain(self) -> None:
        """Sunday 18:00 ET - Retrain reward model."""
        result = self.weekend_retrain()
        self._send_alert(
            f"Weekly retrain complete: {result.get('new_pairs', 0)} new preference pairs, "
            f"loss={result.get('final_loss', 'N/A')}",
            level="info",
        )

    # ==================================================================================
    # Public API methods
    # ==================================================================================

    def pre_market_analysis(self, symbols: List[str]) -> List[Dict]:
        """
        Run full multi-agent analysis pipeline + options analysis.

        Args:
            symbols: List of ticker symbols to analyze.

        Returns:
            List of signal dicts with keys:
                symbol, action, confidence, score, entry_price, stop_loss,
                take_profit, options_analysis (if available)
        """
        logger.info(f"Starting pre-market analysis for {len(symbols)} symbols")
        signals: List[Dict] = []

        for symbol in symbols:
            try:
                signal_data: Dict[str, Any] = {
                    "symbol": symbol,
                    "timestamp": _now_et().isoformat(),
                    "action": "hold",
                    "confidence": 0.0,
                    "score": 0.0,
                    "entry_price": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "options_analysis": None,
                    "agent_analyses": {},
                }

                # Run multi-agent pipeline if available
                if self.pipeline is not None:
                    try:
                        if hasattr(self.pipeline, "run"):
                            result = self.pipeline.run(symbol)
                        elif callable(self.pipeline):
                            result = self.pipeline(symbol)
                        else:
                            result = {}

                        if isinstance(result, dict):
                            signal_data.update({
                                "action": result.get("action", "hold"),
                                "confidence": result.get("confidence", 0.0),
                                "score": result.get("score", 0.0),
                                "entry_price": result.get("entry_price"),
                                "stop_loss": result.get("stop_loss"),
                                "take_profit": result.get("take_profit"),
                                "agent_analyses": result.get("agent_analyses", {}),
                                "options_analysis": result.get("options_analysis"),
                            })
                    except Exception as e:
                        logger.error(f"Pipeline error for {symbol}: {e}")

                # Only include actionable signals
                if signal_data["action"] != "hold" and signal_data["confidence"] > 0.3:
                    signals.append(signal_data)
                    logger.info(
                        f"Signal: {signal_data['action']} {symbol} "
                        f"(conf={signal_data['confidence']:.2f})"
                    )

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        logger.info(f"Pre-market analysis complete: {len(signals)} actionable signals")
        return signals

    def execute_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Execute top signals through OMS with risk checks.

        Args:
            signals: List of signal dicts from pre_market_analysis.

        Returns:
            List of execution result dicts.
        """
        logger.info(f"Executing {len(signals)} signals")
        results: List[Dict] = []

        max_positions = self.config.get("max_positions", 5)
        executed = 0

        for sig in signals:
            if executed >= max_positions:
                logger.info(f"Max positions ({max_positions}) reached, stopping")
                break

            symbol = sig["symbol"]
            action = sig["action"]

            # Risk check
            if self.risk_engine is not None:
                try:
                    risk_ok = self.risk_engine.check_risk(
                        symbol=symbol,
                        action=action,
                        size=sig.get("position_size", 100),
                    )
                    if not risk_ok:
                        logger.warning(f"Risk check failed for {symbol}, skipping")
                        results.append({
                            "symbol": symbol,
                            "status": "rejected",
                            "reason": "risk_check_failed",
                        })
                        continue
                except Exception as e:
                    logger.error(f"Risk engine error for {symbol}: {e}")
                    continue

            # Submit order
            order_result: Dict[str, Any] = {
                "symbol": symbol,
                "action": action,
                "confidence": sig.get("confidence", 0),
                "timestamp": _now_et().isoformat(),
            }

            if self._dry_run:
                order_result["status"] = "simulated"
                order_result["order_id"] = f"dry_{symbol}_{int(time.time())}"
                logger.info(f"[DRY RUN] Would execute: {action} {symbol}")
            elif self.oms is not None:
                try:
                    oms_result = self.oms.submit_order(
                        symbol=symbol,
                        side="buy" if action == "buy" else "sell",
                        qty=sig.get("position_size", 100),
                        order_type=sig.get("order_type", "market"),
                        stop_loss=sig.get("stop_loss"),
                        take_profit=sig.get("take_profit"),
                    )
                    order_result["status"] = "submitted"
                    order_result["order_id"] = oms_result.get("order_id")
                    order_result["fill_price"] = oms_result.get("fill_price")
                except Exception as e:
                    order_result["status"] = "error"
                    order_result["error"] = str(e)
                    logger.error(f"OMS error for {symbol}: {e}")
            else:
                order_result["status"] = "no_oms"
                logger.warning("No OMS configured")

            results.append(order_result)
            executed += 1

        logger.info(f"Execution complete: {executed} orders submitted")
        return results

    def monitor_positions(self) -> Dict:
        """
        Check all open positions: stops, P&L, Greeks drift.

        Returns:
            Summary dict with position status.
        """
        summary: Dict[str, Any] = {
            "timestamp": _now_et().isoformat(),
            "total_positions": 0,
            "total_unrealized_pnl": 0.0,
            "positions": [],
            "alerts": [],
        }

        # Update from OMS
        if self.oms is not None:
            try:
                positions = self.oms.get_positions() if hasattr(self.oms, "get_positions") else []
                summary["total_positions"] = len(positions)

                for pos in positions:
                    pos_info: Dict[str, Any] = {
                        "symbol": pos.get("symbol", "?"),
                        "side": pos.get("side", "?"),
                        "quantity": pos.get("quantity", 0),
                        "unrealized_pnl": pos.get("unrealized_pnl", 0),
                        "entry_price": pos.get("entry_price", 0),
                        "current_price": pos.get("current_price", 0),
                    }
                    summary["total_unrealized_pnl"] += pos_info["unrealized_pnl"]
                    summary["positions"].append(pos_info)

                    # Check stop-loss breach
                    if pos.get("stop_loss") and pos.get("current_price"):
                        if pos["side"] == "long" and pos["current_price"] <= pos["stop_loss"]:
                            alert = f"STOP HIT: {pos['symbol']} @ {pos['current_price']}"
                            summary["alerts"].append(alert)
                            self._send_alert(alert, level="warning")
                        elif pos["side"] == "short" and pos["current_price"] >= pos["stop_loss"]:
                            alert = f"STOP HIT: {pos['symbol']} @ {pos['current_price']}"
                            summary["alerts"].append(alert)
                            self._send_alert(alert, level="warning")

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}", exc_info=True)

        # Check Greeks drift for options positions
        if self.risk_engine is not None and hasattr(self.risk_engine, "check_greeks"):
            try:
                greeks_alerts = self.risk_engine.check_greeks()
                for alert in greeks_alerts:
                    summary["alerts"].append(alert)
                    self._send_alert(alert, level="warning")
            except Exception as e:
                logger.error(f"Error checking Greeks: {e}")

        return summary

    def end_of_day_routine(self) -> Dict:
        """
        Compute daily P&L, update outcome tracker, send summary.

        Returns:
            EOD summary dict.
        """
        logger.info("Running end-of-day routine")

        summary: Dict[str, Any] = {
            "date": _now_et().strftime("%Y-%m-%d"),
            "daily_pnl": 0.0,
            "trades_executed": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "positions_open": 0,
            "positions_closed_today": 0,
        }

        # Gather P&L from OMS
        if self.oms is not None:
            try:
                if hasattr(self.oms, "get_daily_pnl"):
                    summary["daily_pnl"] = self.oms.get_daily_pnl()
                if hasattr(self.oms, "get_daily_trades"):
                    trades = self.oms.get_daily_trades()
                    summary["trades_executed"] = len(trades)
                    summary["winning_trades"] = sum(
                        1 for t in trades if t.get("pnl", 0) > 0
                    )
                    summary["losing_trades"] = sum(
                        1 for t in trades if t.get("pnl", 0) < 0
                    )
                if hasattr(self.oms, "get_positions"):
                    summary["positions_open"] = len(self.oms.get_positions())
            except Exception as e:
                logger.error(f"Error computing EOD stats: {e}")

        # Update outcome tracker
        if self.pipeline is not None and hasattr(self.pipeline, "outcome_tracker"):
            try:
                self.pipeline.outcome_tracker.update_positions()
            except Exception as e:
                logger.error(f"Error updating outcome tracker: {e}")

        logger.info(
            f"EOD: P&L=${summary['daily_pnl']:.2f}, "
            f"trades={summary['trades_executed']}, "
            f"W/L={summary['winning_trades']}/{summary['losing_trades']}"
        )

        return summary

    def weekend_retrain(self) -> Dict:
        """
        Retrain reward model on new preference pairs.

        Returns:
            Retrain result dict.
        """
        logger.info("Starting weekly reward model retrain")

        result: Dict[str, Any] = {
            "timestamp": _now_et().isoformat(),
            "new_pairs": 0,
            "total_pairs": 0,
            "final_loss": None,
            "status": "skipped",
        }

        if self.pipeline is None:
            logger.warning("No pipeline configured, skipping retrain")
            return result

        try:
            # Generate new preference pairs from recent outcomes
            if hasattr(self.pipeline, "preference_generator"):
                pref_gen = self.pipeline.preference_generator
                new_pairs = pref_gen.generate_preferences(min_samples=10)
                result["new_pairs"] = len(new_pairs)
                result["total_pairs"] = len(pref_gen.preferences)

            # Retrain reward model
            if hasattr(self.pipeline, "reward_model"):
                reward_model = self.pipeline.reward_model
                if hasattr(self.pipeline, "preference_generator"):
                    training_data = self.pipeline.preference_generator.get_training_data()
                    if len(training_data) >= 10:
                        train_result = reward_model.train(training_data)
                        result["final_loss"] = train_result.get("final_loss")
                        result["status"] = "completed"
                        logger.info(
                            f"Retrain complete: loss={result['final_loss']}, "
                            f"{result['new_pairs']} new pairs"
                        )
                    else:
                        result["status"] = "insufficient_data"
                        logger.info("Not enough training data for retrain")
                else:
                    result["status"] = "no_preference_generator"
            else:
                result["status"] = "no_reward_model"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Retrain failed: {e}", exc_info=True)

        return result

    # ==================================================================================
    # Helpers
    # ==================================================================================

    def _send_alert(self, message: str, level: str = "info") -> None:
        """Send an alert/notification."""
        if self.alert_manager is not None:
            try:
                self.alert_manager.send_alert(message=message, level=level)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            log_fn = getattr(logger, level, logger.info)
            log_fn(f"[ALERT] {message}")
