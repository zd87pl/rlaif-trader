"""Composite trading metric and experiment result container.

The CompositeMetric is the trading analog of autoresearch-mlx's val_bpb --
a single scalar that determines keep/discard. Higher is better (unlike val_bpb
where lower is better).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    thesis_id: str = ""
    spec_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # The single metric (like val_bpb, but higher = better)
    composite_score: float = 0.0

    # Component metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    cumulative_return: float = 0.0
    num_trades: int = 0
    avg_trade_return: float = 0.0

    # Meta
    backtest_duration_seconds: float = 0.0
    status: str = "pending"  # keep | discard | crash | timeout
    description: str = ""
    error: Optional[str] = None
    backtest_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CompositeMetric:
    """Weighted combination of trading performance metrics.

    Default weights:
        sharpe:   0.35  (risk-adjusted returns dominate)
        return:   0.30  (raw profitability)
        drawdown: 0.20  (penalty for peak-to-trough loss)
        hit_rate: 0.15  (consistency)

    The composite score is normalized so that a "decent" strategy scores ~0.5
    and an excellent one scores ~1.0+.
    """

    def __init__(
        self,
        sharpe_weight: float = 0.35,
        return_weight: float = 0.30,
        drawdown_weight: float = 0.20,
        hit_rate_weight: float = 0.15,
    ):
        self.weights = {
            "sharpe": sharpe_weight,
            "return": return_weight,
            "drawdown": drawdown_weight,
            "hit_rate": hit_rate_weight,
        }

    def compute(self, metrics: Dict[str, float]) -> float:
        """Compute composite score from backtest metrics.

        Parameters
        ----------
        metrics : dict
            Must contain: sharpe_ratio, cumulative_return, max_drawdown, hit_rate

        Returns
        -------
        float
            Composite score (higher is better).
        """
        sharpe = metrics.get("sharpe_ratio", 0.0)
        cum_return = metrics.get("cumulative_return", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)  # negative number
        hit_rate = metrics.get("hit_rate", 0.0)

        # Normalize each component to roughly [0, 1] range
        norm_sharpe = self._sigmoid(sharpe, center=1.0, scale=1.0)
        norm_return = self._sigmoid(cum_return * 100, center=10.0, scale=10.0)
        norm_dd = 1.0 - min(abs(max_dd), 1.0)  # 0% dd -> 1.0, 100% dd -> 0.0
        norm_hit = hit_rate  # already 0-1

        score = (
            self.weights["sharpe"] * norm_sharpe
            + self.weights["return"] * norm_return
            + self.weights["drawdown"] * norm_dd
            + self.weights["hit_rate"] * norm_hit
        )
        return round(float(score), 6)

    @staticmethod
    def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
        """Sigmoid normalization to [0, 1]."""
        return float(1.0 / (1.0 + np.exp(-(x - center) / max(scale, 1e-9))))

    def compute_components(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Return individual component contributions for debugging."""
        sharpe = metrics.get("sharpe_ratio", 0.0)
        cum_return = metrics.get("cumulative_return", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)

        return {
            "sharpe_normalized": self._sigmoid(sharpe, 1.0, 1.0),
            "return_normalized": self._sigmoid(cum_return * 100, 10.0, 10.0),
            "drawdown_normalized": 1.0 - min(abs(max_dd), 1.0),
            "hit_rate_normalized": hit_rate,
            "sharpe_contribution": self.weights["sharpe"] * self._sigmoid(sharpe, 1.0, 1.0),
            "return_contribution": self.weights["return"] * self._sigmoid(cum_return * 100, 10.0, 10.0),
            "drawdown_contribution": self.weights["drawdown"] * (1.0 - min(abs(max_dd), 1.0)),
            "hit_rate_contribution": self.weights["hit_rate"] * hit_rate,
        }
