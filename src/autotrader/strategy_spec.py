"""StrategySpec: Serializable strategy definition with embedded Python signal code.

This is the trading analog of autoresearch-mlx's train.py -- the single artifact
that the LLM modifies each experiment iteration. The signal_code field contains
a complete Python function that the ExperimentRunner executes in a sandbox.
"""

from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Default baseline signal function -- simple SMA crossover
BASELINE_SIGNAL_CODE = '''\
def generate_signals(df, parameters):
    """Baseline SMA crossover strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + technical features. Guaranteed columns:
        open, high, low, close, volume, sma_20, sma_50, rsi,
        macd, macd_signal, macd_hist, bb_upper, bb_lower, bb_mid,
        atr, adx, obv, vwap
    parameters : dict
        Auxiliary parameters from StrategySpec.parameters.

    Returns
    -------
    pd.Series
        Signal series: 1 = long, -1 = short, 0 = flat.
    """
    import pandas as pd

    sma_fast = df.get("sma_20")
    sma_slow = df.get("sma_50")
    rsi = df.get("rsi")
    macd_hist = df.get("macd_hist")

    if sma_fast is None or sma_slow is None:
        return pd.Series(0, index=df.index)

    long_signal = (
        (df["close"] > sma_fast)
        & (sma_fast > sma_slow)
        & (macd_hist > 0)
        & (rsi > 45)
        & (rsi < 75)
    )

    short_signal = (
        (df["close"] < sma_fast)
        & (sma_fast < sma_slow)
        & (macd_hist < 0)
        & (rsi < 55)
        & (rsi > 25)
    )

    signals = pd.Series(0, index=df.index)
    signals[long_signal] = 1
    signals[short_signal] = -1
    return signals
'''


@dataclass
class StrategySpec:
    """Immutable snapshot of a trading strategy."""

    spec_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signal_code: str = BASELINE_SIGNAL_CODE
    description: str = "baseline SMA crossover"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> StrategySpec:
        data = json.loads(raw)
        return cls(**data)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.spec_id}.json"
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path) -> StrategySpec:
        return cls.from_json(Path(path).read_text())

    def diff(self, other: StrategySpec) -> str:
        """Human-readable unified diff of signal code."""
        old_lines = self.signal_code.splitlines(keepends=True)
        new_lines = other.signal_code.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"spec/{self.spec_id}",
                tofile=f"spec/{other.spec_id}",
            )
        )

    def derive(
        self,
        signal_code: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StrategySpec:
        """Create a child spec with new signal code, preserving lineage."""
        return StrategySpec(
            parent_id=self.spec_id,
            signal_code=signal_code,
            description=description,
            parameters={**self.parameters, **(parameters or {})},
        )
