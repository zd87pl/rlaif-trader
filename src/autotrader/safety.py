"""Safety guard for autonomous trading experimentation.

Unlike autoresearch-mlx where a bad experiment just wastes 5 minutes of compute,
a bad strategy swap can lose real money. This module provides:
1. AST-based code validation (sandbox whitelist)
2. Rate limiting (experiments/hour, swaps/day)
3. Improvement thresholds
4. Kill switch integration
"""

from __future__ import annotations

import ast
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Modules that generated signal code is allowed to import
ALLOWED_IMPORTS: Set[str] = {
    "pandas", "pd",
    "numpy", "np",
    "math",
    "ta",
    "statistics",
    "functools",
    "itertools",
    "collections",
    "operator",
    "decimal",
}

# Names that are never allowed in generated code
FORBIDDEN_NAMES: Set[str] = {
    "open", "exec", "eval", "compile", "__import__",
    "subprocess", "os", "sys", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "pickle", "shelve", "ctypes",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit",
    "input",
}

# Forbidden attribute access patterns
FORBIDDEN_ATTRS: Set[str] = {
    "system", "popen", "exec", "spawn",
    "remove", "unlink", "rmdir", "rmtree",
    "write", "writelines",
    "send", "connect", "bind", "listen",
    "__subclasses__", "__bases__", "__mro__",
}


@dataclass
class SafetyGuard:
    """Rate limits, code validation, and kill switch for autonomous experimentation."""

    max_experiments_per_hour: int = 12
    max_swaps_per_day: int = 6
    min_improvement_threshold: float = 0.01
    max_consecutive_crashes: int = 10
    allowed_imports: Set[str] = field(default_factory=lambda: set(ALLOWED_IMPORTS))

    # Internal state
    _experiment_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    _swap_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    _consecutive_crashes: int = 0
    _halted: bool = False
    _risk_engine: Any = None

    def set_risk_engine(self, risk_engine: Any) -> None:
        self._risk_engine = risk_engine

    # ── Code validation ─────────────────────────────────────────────────

    def validate_signal_code(self, code: str) -> Tuple[bool, str]:
        """Validate generated signal code via AST analysis.

        Checks:
        1. Code is valid Python (parseable)
        2. Contains the required generate_signals function
        3. No forbidden imports (only whitelisted modules)
        4. No forbidden function calls (os.system, eval, exec, etc.)
        5. No forbidden attribute access patterns
        6. No class definitions (keep it simple)

        Returns (is_valid, reason).
        """
        # Step 1: Parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        # Step 2: Must define generate_signals
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if "generate_signals" not in func_names:
            return False, "Missing required function: generate_signals(df, parameters)"

        # Step 3-5: Walk the AST
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root not in self.allowed_imports:
                        return False, f"Forbidden import: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root not in self.allowed_imports:
                        return False, f"Forbidden import: from {node.module}"

            # Check function calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                    return False, f"Forbidden call: {func.id}()"
                if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTRS:
                    return False, f"Forbidden attribute call: .{func.attr}()"

            # Check bare name references
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                # Allow as function parameter names
                if not isinstance(node.ctx, ast.Load):
                    continue
                # Check if it's a function def arg
                return False, f"Forbidden name reference: {node.id}"

            # Check attribute access
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                return False, f"Forbidden attribute access: .{node.attr}"

            # No class definitions
            if isinstance(node, ast.ClassDef):
                return False, f"Class definitions not allowed: {node.name}"

            # No async
            if isinstance(node, ast.AsyncFunctionDef):
                return False, "Async functions not allowed"

        return True, "ok"

    # ── Rate limiting ───────────────────────────────────────────────────

    def check_experiment_rate(self) -> Tuple[bool, str]:
        """Check if we can run another experiment."""
        if self._halted:
            return False, "AutoTrader HALTED (too many crashes or manual stop)"

        now = time.time()
        hour_ago = now - 3600
        recent = sum(1 for t in self._experiment_timestamps if t > hour_ago)
        if recent >= self.max_experiments_per_hour:
            return False, f"Rate limit: {recent}/{self.max_experiments_per_hour} experiments/hour"
        return True, "ok"

    def check_swap_rate(self) -> Tuple[bool, str]:
        """Check if we can perform another strategy swap."""
        if self._halted:
            return False, "AutoTrader HALTED"

        now = time.time()
        day_ago = now - 86400
        recent = sum(1 for t in self._swap_timestamps if t > day_ago)
        if recent >= self.max_swaps_per_day:
            return False, f"Rate limit: {recent}/{self.max_swaps_per_day} swaps/day"
        return True, "ok"

    def record_experiment(self) -> None:
        self._experiment_timestamps.append(time.time())

    def record_swap(self) -> None:
        self._swap_timestamps.append(time.time())

    # ── Crash tracking ──────────────────────────────────────────────────

    def record_crash(self) -> None:
        self._consecutive_crashes += 1
        if self._consecutive_crashes >= self.max_consecutive_crashes:
            self._halted = True
            logger.critical(
                "AutoTrader HALTED: %d consecutive crashes",
                self._consecutive_crashes,
            )

    def record_success(self) -> None:
        self._consecutive_crashes = 0

    # ── Swap validation ─────────────────────────────────────────────────

    def validate_swap(
        self,
        new_score: float,
        current_best: float,
    ) -> Tuple[bool, str]:
        """Check if an experiment result justifies a live strategy swap."""
        swap_ok, swap_msg = self.check_swap_rate()
        if not swap_ok:
            return False, swap_msg

        improvement = new_score - current_best
        if improvement < self.min_improvement_threshold:
            return False, (
                f"Improvement {improvement:.6f} below threshold "
                f"{self.min_improvement_threshold:.6f}"
            )

        return True, f"Approved: +{improvement:.6f} improvement"

    # ── Emergency stop ──────────────────────────────────────────────────

    def emergency_stop(self) -> None:
        """Halt all autotrader activity and activate kill switch if available."""
        self._halted = True
        logger.critical("AutoTrader EMERGENCY STOP activated")
        if self._risk_engine and hasattr(self._risk_engine, "kill_switch"):
            self._risk_engine.kill_switch()

    def resume(self) -> None:
        """Resume after manual review."""
        self._halted = False
        self._consecutive_crashes = 0
        logger.warning("AutoTrader resumed after manual review")

    @property
    def is_halted(self) -> bool:
        return self._halted

    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "halted": self._halted,
            "consecutive_crashes": self._consecutive_crashes,
            "experiments_last_hour": sum(
                1 for t in self._experiment_timestamps if t > now - 3600
            ),
            "swaps_last_24h": sum(
                1 for t in self._swap_timestamps if t > now - 86400
            ),
            "max_experiments_per_hour": self.max_experiments_per_hour,
            "max_swaps_per_day": self.max_swaps_per_day,
            "min_improvement_threshold": self.min_improvement_threshold,
        }
