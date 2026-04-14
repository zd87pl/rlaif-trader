"""Strategy Hot-Swapper: apply kept strategy modifications to live trading.

YOLO mode: immediate swap on improvement. No paper trial gate.
The swapper:
1. Validates new code via SafetyGuard
2. Saves strategy snapshot to disk (git-trackable)
3. Closes positions misaligned with new strategy
4. Updates the pipeline's active signal function
5. Logs the swap to audit trail
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import ExperimentResult
from .safety import SafetyGuard
from .strategy_spec import StrategySpec
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SwapResult:
    """Result of a strategy hot-swap attempt."""

    success: bool = False
    old_spec_id: str = ""
    new_spec_id: str = ""
    positions_closed: int = 0
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StrategyHotSwapper:
    """Hot-swap trading strategies in YOLO mode.

    Parameters
    ----------
    oms : OrderManagementSystem, optional
        For closing/opening positions during swap.
    risk_engine : optional
        For kill switch integration.
    safety : SafetyGuard
        Rate limiting and code validation.
    strategies_dir : str
        Directory for persisting strategy snapshots.
    audit_log_path : str
        Path for the audit trail JSONL file.
    """

    def __init__(
        self,
        oms: Any = None,
        risk_engine: Any = None,
        safety: Optional[SafetyGuard] = None,
        strategies_dir: str = "data/autotrader/strategies",
        audit_log_path: str = "data/autotrader/audit.jsonl",
    ):
        self.oms = oms
        self.risk_engine = risk_engine
        self.safety = safety or SafetyGuard()
        self.strategies_dir = Path(strategies_dir)
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Active strategy reference (set by orchestrator)
        self._active_spec: Optional[StrategySpec] = None
        self._active_signal_fn: Any = None

    @property
    def active_spec(self) -> Optional[StrategySpec]:
        return self._active_spec

    def set_active(self, spec: StrategySpec) -> None:
        """Set the active strategy (called during initialization)."""
        self._active_spec = spec
        self._compile_signal_fn(spec)
        spec.save(self.strategies_dir)
        logger.info("Active strategy set: %s [%s]", spec.spec_id, spec.description)

    def swap(
        self,
        new_spec: StrategySpec,
        experiment_result: Optional[ExperimentResult] = None,
    ) -> SwapResult:
        """Execute a YOLO strategy swap.

        Parameters
        ----------
        new_spec : StrategySpec
            The new strategy to deploy.
        experiment_result : ExperimentResult, optional
            The experiment that justified this swap (for audit).

        Returns
        -------
        SwapResult
        """
        old_spec = self._active_spec
        result = SwapResult(
            old_spec_id=old_spec.spec_id if old_spec else "",
            new_spec_id=new_spec.spec_id,
        )

        # 1. Validate the new code
        valid, reason = self.safety.validate_signal_code(new_spec.signal_code)
        if not valid:
            result.reason = f"Code validation failed: {reason}"
            self._audit("swap_rejected", result, experiment_result)
            return result

        # 2. Check swap rate limits
        swap_ok, swap_msg = self.safety.check_swap_rate()
        if not swap_ok:
            result.reason = swap_msg
            self._audit("swap_rate_limited", result, experiment_result)
            return result

        # 3. Compile the new signal function
        try:
            self._compile_signal_fn(new_spec)
        except Exception as e:
            result.reason = f"Signal function compilation failed: {e}"
            self._audit("swap_compile_failed", result, experiment_result)
            return result

        # 4. Close existing positions (YOLO: close all and let new strategy reopen)
        positions_closed = 0
        if self.oms:
            try:
                close_result = self.oms.close_all_positions()
                positions_closed = close_result.get("closed", 0)
            except Exception as e:
                logger.warning("Failed to close positions during swap: %s", e)

        # 5. Save new strategy and update active
        new_spec.save(self.strategies_dir)
        self._active_spec = new_spec
        self.safety.record_swap()

        result.success = True
        result.positions_closed = positions_closed
        result.reason = "YOLO swap executed"

        self._audit("swap_executed", result, experiment_result)

        logger.info(
            "STRATEGY SWAPPED: %s -> %s [%s] (closed %d positions)",
            result.old_spec_id,
            result.new_spec_id,
            new_spec.description,
            positions_closed,
        )

        return result

    def _compile_signal_fn(self, spec: StrategySpec) -> None:
        """Compile the signal code into a callable function."""
        import pandas as pd
        import numpy as np

        exec_globals = {"pd": pd, "np": np, "math": __import__("math")}
        exec(spec.signal_code, exec_globals)
        self._active_signal_fn = exec_globals["generate_signals"]

    def generate_signals(self, df: Any, parameters: Optional[Dict] = None) -> Any:
        """Execute the active strategy's signal function.

        This is the interface the trading pipeline calls.
        """
        if self._active_signal_fn is None:
            import pandas as pd
            return pd.Series(0, index=df.index)

        params = parameters or (self._active_spec.parameters if self._active_spec else {})
        return self._active_signal_fn(df, params)

    def rollback(self) -> SwapResult:
        """Rollback to the parent strategy."""
        if not self._active_spec or not self._active_spec.parent_id:
            return SwapResult(reason="No parent strategy to rollback to")

        parent_path = self.strategies_dir / f"{self._active_spec.parent_id}.json"
        if not parent_path.exists():
            return SwapResult(reason=f"Parent spec {self._active_spec.parent_id} not found on disk")

        parent_spec = StrategySpec.load(parent_path)
        return self.swap(parent_spec)

    def _audit(
        self,
        action: str,
        result: SwapResult,
        experiment: Optional[ExperimentResult] = None,
    ) -> None:
        """Append to audit log."""
        record = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_spec_id": result.old_spec_id,
            "new_spec_id": result.new_spec_id,
            "success": result.success,
            "positions_closed": result.positions_closed,
            "reason": result.reason,
        }
        if experiment:
            record["experiment_id"] = experiment.experiment_id
            record["composite_score"] = experiment.composite_score
            record["sharpe_ratio"] = experiment.sharpe_ratio

        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Failed to write audit log: %s", e)

    def status(self) -> Dict[str, Any]:
        return {
            "active_spec_id": self._active_spec.spec_id if self._active_spec else None,
            "active_description": self._active_spec.description if self._active_spec else None,
            "has_signal_fn": self._active_signal_fn is not None,
            "strategies_on_disk": len(list(self.strategies_dir.glob("*.json"))),
        }
