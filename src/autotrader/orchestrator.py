"""Experiment Orchestrator: the NEVER STOP loop.

Direct analog of autoresearch-mlx's program.md protocol:
    LOOP FOREVER:
        1. sentinel.check() -> events
        2. thesis_gen.propose(events) -> thesis with signal code
        3. safety.validate(thesis.code)
        4. runner.run_experiment(thesis) -> result
        5. log.append(result)
        6. IF improved: swapper.swap() and update current best
        7. ELSE: discard
        8. Feed result to RLAIF preference generator
        9. REPEAT
"""

from __future__ import annotations

import signal
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .experiment_log import ExperimentLog
from .experiment_runner import ExperimentRunner
from .metrics import ExperimentResult
from .safety import SafetyGuard
from .sentinel import MarketEvent, MarketSentinel
from .strategy_spec import StrategySpec
from .strategy_swapper import StrategyHotSwapper
from .thesis_generator import ThesisGenerator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentOrchestrator:
    """Main autonomous experimentation loop.

    Parameters
    ----------
    sentinel : MarketSentinel
    thesis_gen : ThesisGenerator
    runner : ExperimentRunner
    swapper : StrategyHotSwapper
    safety : SafetyGuard
    log : ExperimentLog
    mode : str
        'continuous' - run as fast as possible
        'on_event' - only experiment when sentinel detects events
        'hourly' - run at most once per hour
    improvement_threshold : float
        Minimum composite score improvement to trigger a swap.
    rlaif_callback : callable, optional
        Called with (ExperimentResult, StrategySpec) after each experiment
        to feed into the RLAIF preference generator.
    """

    def __init__(
        self,
        sentinel: MarketSentinel,
        thesis_gen: ThesisGenerator,
        runner: ExperimentRunner,
        swapper: StrategyHotSwapper,
        safety: SafetyGuard,
        log: ExperimentLog,
        mode: str = "continuous",
        improvement_threshold: float = 0.01,
        rlaif_callback: Optional[Callable] = None,
        time_budget_seconds: int = 300,
        strategist: Any = None,
        risk_engine: Any = None,
        auto_reassess: bool = True,
        reassess_interval_minutes: Optional[int] = None,
    ):
        self.sentinel = sentinel
        self.thesis_gen = thesis_gen
        self.runner = runner
        self.swapper = swapper
        self.safety = safety
        self.log = log
        self.mode = mode
        self.improvement_threshold = improvement_threshold
        self.rlaif_callback = rlaif_callback
        self.time_budget_seconds = time_budget_seconds
        self.strategist = strategist
        self.risk_engine = risk_engine
        self.auto_reassess = auto_reassess
        self.reassess_interval_minutes = reassess_interval_minutes

        # State
        self._running = False
        self._current_best_score = 0.0
        self._iteration_count = 0
        self._last_experiment_time = 0.0
        self._thesis_guidance: str = ""  # injected by strategist

    def run(
        self,
        initial_spec: Optional[StrategySpec] = None,
        portfolio_state_fn: Optional[Callable] = None,
        market_data_fn: Optional[Callable] = None,
    ) -> None:
        """Start the NEVER STOP loop.

        Parameters
        ----------
        initial_spec : StrategySpec, optional
            Starting strategy. If None, uses default baseline.
        portfolio_state_fn : callable, optional
            Returns current portfolio state dict when called.
        market_data_fn : callable, optional
            Returns current market data dict when called.
        """
        if initial_spec is None:
            initial_spec = StrategySpec()

        # Initialize
        self.swapper.set_active(initial_spec)
        self._current_best_score = self.log.best_score()
        self._running = True

        # If we have no baseline score, run one to establish it
        if self._current_best_score == 0.0:
            logger.info("Running baseline experiment to establish initial score...")
            baseline_result = self.runner.run_experiment(initial_spec)
            if baseline_result.status != "crash":
                self._current_best_score = baseline_result.composite_score
                baseline_result.status = "keep"
                baseline_result.description = "baseline"
                self.log.append(baseline_result)
                logger.info(
                    "Baseline score: %.6f (Sharpe=%.4f, Return=%.4f)",
                    self._current_best_score,
                    baseline_result.sharpe_ratio,
                    baseline_result.cumulative_return,
                )

        logger.info(
            "AutoTrader orchestrator starting (mode=%s, threshold=%.4f, best=%.6f)",
            self.mode,
            self.improvement_threshold,
            self._current_best_score,
        )

        # Handle graceful shutdown
        def _signal_handler(signum, frame):
            logger.info("Shutdown signal received, stopping orchestrator...")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # === THE LOOP ===
        while self._running:
            try:
                self._iteration(portfolio_state_fn, market_data_fn)
            except Exception as e:
                logger.error("Orchestrator iteration error: %s\n%s", e, traceback.format_exc())
                self.safety.record_crash()
                if self.safety.is_halted:
                    logger.critical("AutoTrader HALTED after too many crashes")
                    break
                time.sleep(10)

    def _apply_directive(self, directive: Any) -> None:
        """Apply a PortfolioDirective from the strategist to reconfigure the loop."""
        # Update sentinel timing
        self.sentinel.update_timing(
            check_interval=directive.check_interval_seconds,
            scheduled_interval=directive.check_interval_seconds * 2,
        )
        # Update composite metric weights
        w = directive.composite_weights
        self.runner.metric.update_weights(
            sharpe=w.get("sharpe"),
            return_w=w.get("return"),
            drawdown=w.get("drawdown"),
            hit_rate=w.get("hit_rate"),
        )
        # Update improvement threshold
        self.improvement_threshold = directive.improvement_threshold
        # Update experiment frequency (mode)
        if directive.experiment_frequency in {"continuous", "hourly", "4h", "daily", "on_event"}:
            self.mode = directive.experiment_frequency
        # Inject thesis guidance
        self._thesis_guidance = directive.thesis_guidance or ""
        # Update risk engine
        if self.risk_engine and hasattr(self.risk_engine, "set_dynamic_limits"):
            self.risk_engine.set_dynamic_limits(
                max_exposure_pct=directive.risk_budget_pct,
                max_position_pct=directive.max_position_pct,
            )
        # Update default symbols on runner
        if directive.symbols_focus:
            self.runner.default_symbols = directive.symbols_focus

        logger.info(
            "Directive applied: style=%s, risk=%.0f%%, check=%ds, mode=%s",
            directive.strategy_style,
            directive.risk_budget_pct * 100,
            directive.check_interval_seconds,
            self.mode,
        )

    def _iteration(
        self,
        portfolio_state_fn: Optional[Callable],
        market_data_fn: Optional[Callable],
    ) -> None:
        """Single iteration of the loop."""
        self._iteration_count += 1

        # Periodic strategist reassessment
        if self.auto_reassess and self.strategist and self.strategist.needs_reassessment():
            try:
                portfolio_state = portfolio_state_fn() if portfolio_state_fn else None
                directive = self.strategist.assess(
                    wallet_balance=portfolio_state.get("account", {}).get("equity") if portfolio_state else None,
                    current_positions=portfolio_state.get("positions", []) if portfolio_state else None,
                    risk_preference=self.strategist._current_directive.risk_preference
                        if self.strategist._current_directive else "moderate",
                    reassess_after_minutes_override=self.reassess_interval_minutes,
                )
                self._apply_directive(directive)
            except Exception as e:
                logger.debug("Strategist reassessment failed: %s", e)

        # Rate control based on mode
        cadence_seconds = {
            "hourly": 3600,
            "4h": 4 * 3600,
            "daily": 24 * 3600,
        }.get(self.mode)
        if cadence_seconds is not None:
            elapsed = time.time() - self._last_experiment_time
            if elapsed < cadence_seconds:
                time.sleep(min(30, cadence_seconds - elapsed))
                return

        # Check safety
        if self.safety.is_halted:
            logger.warning("AutoTrader halted, sleeping...")
            time.sleep(60)
            return

        rate_ok, rate_msg = self.safety.check_experiment_rate()
        if not rate_ok:
            logger.info("Rate limited: %s", rate_msg)
            time.sleep(30)
            return

        # 1. Check sentinel for events
        portfolio_state = portfolio_state_fn() if portfolio_state_fn else None
        market_data = market_data_fn() if market_data_fn else None
        events = self.sentinel.check(portfolio_state, market_data)

        if self.mode == "on_event" and not events:
            time.sleep(self.sentinel.check_interval)
            return

        # If no events in continuous mode, use a scheduled placeholder
        if not events:
            from .sentinel import EventType
            events = [MarketEvent(
                event_type=EventType.SCHEDULED,
                severity=0.2,
                details={"iteration": self._iteration_count},
            )]

        # 2. Generate thesis
        current_spec = self.swapper.active_spec or StrategySpec()
        current_perf = {
            "composite_score": self._current_best_score,
            "sharpe_ratio": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
        }
        # Fill from last kept result
        recent = self.log.recent(5)
        kept = [r for r in recent if r.status == "keep"]
        if kept:
            last_kept = kept[-1]
            current_perf["sharpe_ratio"] = last_kept.sharpe_ratio
            current_perf["cumulative_return"] = last_kept.cumulative_return
            current_perf["max_drawdown"] = last_kept.max_drawdown
            current_perf["hit_rate"] = last_kept.hit_rate

        event = max(events, key=lambda e: e.severity)  # Use highest severity event
        history = self.log.recent(30)

        thesis = self.thesis_gen.propose(
            event=event,
            current_spec=current_spec,
            current_performance=current_perf,
            experiment_history=history,
            strategist_guidance=self._thesis_guidance,
        )

        if not thesis.proposed_spec:
            logger.warning("Thesis generator produced no spec, skipping")
            time.sleep(5)
            return

        logger.info(
            "Iteration %d: Testing thesis '%s' (triggered by %s)",
            self._iteration_count,
            thesis.description,
            event.event_type,
        )

        # 3. Validate code
        valid, reason = self.safety.validate_signal_code(
            thesis.proposed_spec.signal_code
        )
        if not valid:
            logger.warning("Thesis code rejected: %s", reason)
            result = ExperimentResult(
                thesis_id=thesis.thesis_id,
                spec_id=thesis.proposed_spec.spec_id,
                status="crash",
                error=f"Code validation: {reason}",
                description=thesis.description,
            )
            self.log.append(result)
            self.safety.record_crash()
            return

        # 4. Run experiment
        result = self.runner.run_experiment(
            spec=thesis.proposed_spec,
            time_budget_seconds=self.time_budget_seconds,
        )
        result.thesis_id = thesis.thesis_id
        self._last_experiment_time = time.time()

        # 5. Keep or discard
        if result.status == "crash":
            self.log.append(result)
            self.safety.record_crash()
            logger.warning(
                "Experiment CRASHED: %s (%s)",
                result.error,
                thesis.description,
            )
            return

        improvement = result.composite_score - self._current_best_score

        if improvement >= self.improvement_threshold:
            # KEEP: swap strategy
            result.status = "keep"
            self.log.append(result)

            swap_result = self.swapper.swap(
                new_spec=thesis.proposed_spec,
                experiment_result=result,
            )

            if swap_result.success:
                self._current_best_score = result.composite_score
                logger.info(
                    "KEPT: %.6f (+%.6f) [%s] Sharpe=%.4f Return=%.4f DD=%.4f",
                    result.composite_score,
                    improvement,
                    thesis.description,
                    result.sharpe_ratio,
                    result.cumulative_return,
                    result.max_drawdown,
                )
            else:
                logger.warning("Swap failed despite good result: %s", swap_result.reason)
        else:
            # DISCARD
            result.status = "discard"
            self.log.append(result)
            logger.info(
                "DISCARD: %.6f (%+.6f) [%s]",
                result.composite_score,
                improvement,
                thesis.description,
            )

        # 6. Feed to RLAIF
        if self.rlaif_callback:
            try:
                self.rlaif_callback(result, thesis.proposed_spec)
            except Exception as e:
                logger.debug("RLAIF callback failed: %s", e)

        self.safety.record_success()

    def stop(self) -> None:
        """Gracefully stop the loop."""
        self._running = False
        logger.info("AutoTrader orchestrator stop requested")

    def run_single(
        self,
        event: Optional[MarketEvent] = None,
    ) -> ExperimentResult:
        """Run a single experiment iteration (for testing/debugging).

        Returns the experiment result.
        """
        current_spec = self.swapper.active_spec or StrategySpec()
        if event is None:
            from .sentinel import EventType
            event = MarketEvent(
                event_type=EventType.SCHEDULED,
                severity=0.5,
                details={"mode": "single_run"},
            )

        history = self.log.recent(30)
        current_perf = {
            "composite_score": self._current_best_score,
            "sharpe_ratio": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
        }

        thesis = self.thesis_gen.propose(
            event=event,
            current_spec=current_spec,
            current_performance=current_perf,
            experiment_history=history,
            strategist_guidance=self._thesis_guidance,
        )

        if not thesis.proposed_spec:
            return ExperimentResult(
                status="crash",
                error="Thesis generator produced no spec",
            )

        result = self.runner.run_experiment(
            spec=thesis.proposed_spec,
            time_budget_seconds=self.time_budget_seconds,
        )
        result.thesis_id = thesis.thesis_id

        improvement = result.composite_score - self._current_best_score

        if result.status != "crash" and improvement >= self.improvement_threshold:
            result.status = "keep"
            self.swapper.swap(thesis.proposed_spec, result)
            self._current_best_score = result.composite_score
        elif result.status != "crash":
            result.status = "discard"

        self.log.append(result)
        return result

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "mode": self.mode,
            "iteration_count": self._iteration_count,
            "current_best_score": self._current_best_score,
            "improvement_threshold": self.improvement_threshold,
            "log_summary": self.log.format_summary(),
            "safety": self.safety.status(),
            "swapper": self.swapper.status(),
        }
