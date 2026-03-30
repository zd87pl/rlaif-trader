"""
Composite Reward Function for RLAIF — Multi-Objective Trading Rewards

Replaces single-metric rewards with a weighted composite that captures
multiple dimensions of trading decision quality:

  - Outcome quality: return, Sharpe, Sortino, Treynor, CVaR
  - Risk management: drawdown, turnover, theta capture, vol prediction
  - Process quality: reasoning chain verification (from ProcessRewardModel)
  - Novelty: contrarian signals that actually worked

Based on Risk-Aware RL Reward research (Jun 2025) and Trade-R1 (Jan 2026).

Each component is a RewardComponent subclass with a compute() method and a
weight. The CompositeRewardFunction orchestrates them, supports batch
evaluation, preference-pair generation for RLAIF training, and dynamic
weight adaptation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Try importing ProcessRewardModel for process quality scoring
try:
    from .reasoning_verifier import ProcessRewardModel, ReasoningChain
    _HAS_VERIFIER = True
except ImportError:
    _HAS_VERIFIER = False
    logger.warning("reasoning_verifier not available; ProcessQualityReward disabled")

# ---------------------------------------------------------------------------
# Annualized trading-day constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
SQRT_TRADING_DAYS = np.sqrt(TRADING_DAYS_PER_YEAR)


# ===========================================================================
# Base class
# ===========================================================================

class RewardComponent(ABC):
    """Base class for reward components. All must be differentiable-friendly."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this component."""
        ...

    @abstractmethod
    def compute(self, trajectory: Dict[str, Any]) -> float:
        """
        Compute the reward from a trajectory dict.

        Args:
            trajectory: Dict containing at minimum 'returns' (array of
                period returns) and optionally other fields depending on
                the component.

        Returns:
            Scalar reward value (higher is better).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight:.3f})"


# ===========================================================================
# Outcome-quality components
# ===========================================================================

class ReturnReward(RewardComponent):
    """Annualized return."""

    @property
    def name(self) -> str:
        return "return"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        if len(returns) == 0:
            return 0.0
        cumulative = np.prod(1.0 + returns) - 1.0
        n_periods = max(len(returns), 1)
        annualized = (1.0 + cumulative) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1.0
        # Clip extreme values for numerical stability
        return float(np.clip(annualized, -1.0, 10.0))


class SharpeReward(RewardComponent):
    """Annualized Sharpe ratio (penalizes volatility)."""

    def __init__(self, weight: float = 1.0, risk_free_rate: float = 0.05):
        super().__init__(weight)
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        return "sharpe"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        if len(returns) < 2:
            return 0.0
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        excess = returns - daily_rf
        std = np.std(excess, ddof=1)
        if std < 1e-10:
            return 0.0
        sharpe = (np.mean(excess) / std) * SQRT_TRADING_DAYS
        return float(np.clip(sharpe, -5.0, 5.0))


class SortinoReward(RewardComponent):
    """Sortino ratio — penalizes downside volatility only."""

    def __init__(self, weight: float = 1.0, risk_free_rate: float = 0.05):
        super().__init__(weight)
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        return "sortino"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        if len(returns) < 2:
            return 0.0
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 1:
            return float(np.clip(np.mean(excess) * SQRT_TRADING_DAYS * 10.0, 0.0, 5.0))
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std < 1e-10:
            return 0.0
        sortino = (np.mean(excess) / downside_std) * SQRT_TRADING_DAYS
        return float(np.clip(sortino, -5.0, 5.0))


class TreynorReward(RewardComponent):
    """Treynor ratio — systematic-risk-adjusted return."""

    def __init__(self, weight: float = 1.0, risk_free_rate: float = 0.05):
        super().__init__(weight)
        self.risk_free_rate = risk_free_rate

    @property
    def name(self) -> str:
        return "treynor"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        benchmark = np.asarray(
            trajectory.get("benchmark_returns", returns * 0.0), dtype=np.float64
        )
        if len(returns) < 2 or len(benchmark) < 2:
            return 0.0
        # Trim to same length
        n = min(len(returns), len(benchmark))
        returns, benchmark = returns[:n], benchmark[:n]
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        excess_r = returns - daily_rf
        excess_b = benchmark - daily_rf
        cov = np.cov(excess_r, excess_b)
        beta = cov[0, 1] / max(cov[1, 1], 1e-10)
        if abs(beta) < 1e-10:
            return 0.0
        treynor = (np.mean(excess_r) * TRADING_DAYS_PER_YEAR) / beta
        return float(np.clip(treynor, -5.0, 5.0))


class CVaRReward(RewardComponent):
    """Conditional Value-at-Risk penalty at a given quantile (default 5%)."""

    def __init__(self, weight: float = 1.0, quantile: float = 0.05):
        super().__init__(weight)
        self.quantile = quantile

    @property
    def name(self) -> str:
        return "cvar"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        if len(returns) < 5:
            return 0.0
        sorted_r = np.sort(returns)
        cutoff = int(np.ceil(len(sorted_r) * self.quantile))
        cutoff = max(cutoff, 1)
        cvar = np.mean(sorted_r[:cutoff])
        # Invert: less negative CVaR = better.  Normalize so ~0 is neutral.
        return float(np.clip(cvar * SQRT_TRADING_DAYS, -5.0, 0.0))


# ===========================================================================
# Risk-management components
# ===========================================================================

class DrawdownReward(RewardComponent):
    """Maximum drawdown penalty (inverted — lower drawdown = higher reward)."""

    @property
    def name(self) -> str:
        return "drawdown"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        returns = np.asarray(trajectory.get("returns", []), dtype=np.float64)
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.min(dd))  # Most negative value
        # Invert: 0% drawdown -> 0, -50% drawdown -> -0.5
        return float(np.clip(max_dd, -1.0, 0.0))


class TurnoverReward(RewardComponent):
    """Transaction cost / turnover penalty."""

    def __init__(self, weight: float = 1.0, cost_bps: float = 5.0):
        super().__init__(weight)
        self.cost_bps = cost_bps  # basis points per unit turnover

    @property
    def name(self) -> str:
        return "turnover"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        weights = trajectory.get("portfolio_weights", None)
        if weights is None:
            turnover = trajectory.get("turnover", 0.0)
            if isinstance(turnover, (list, np.ndarray)):
                turnover = float(np.mean(turnover))
            return float(np.clip(-turnover * self.cost_bps / 10000.0, -1.0, 0.0))
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim < 2 or len(weights) < 2:
            return 0.0
        diffs = np.abs(np.diff(weights, axis=0))
        avg_turnover = float(np.mean(np.sum(diffs, axis=1)))
        cost = avg_turnover * self.cost_bps / 10000.0
        return float(np.clip(-cost, -1.0, 0.0))


class ThetaCaptureReward(RewardComponent):
    """Options-specific: percentage of theta (time decay) captured."""

    @property
    def name(self) -> str:
        return "theta_capture"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        expected_theta = trajectory.get("expected_theta", 0.0)
        actual_pnl = trajectory.get("theta_pnl", 0.0)
        if abs(expected_theta) < 1e-10:
            return 0.0
        capture_ratio = actual_pnl / expected_theta
        # 100% capture = 1.0, >100% is bonus, <0 is bad
        return float(np.clip(capture_ratio, -1.0, 2.0))


class VolPredictionReward(RewardComponent):
    """Accuracy of realized-vol predictions vs actual realized vol."""

    @property
    def name(self) -> str:
        return "vol_prediction"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        predicted_vol = trajectory.get("predicted_vol", None)
        realized_vol = trajectory.get("realized_vol", None)
        if predicted_vol is None or realized_vol is None:
            return 0.0
        predicted_vol = np.asarray(predicted_vol, dtype=np.float64)
        realized_vol = np.asarray(realized_vol, dtype=np.float64)
        n = min(len(predicted_vol), len(realized_vol))
        if n == 0:
            return 0.0
        predicted_vol, realized_vol = predicted_vol[:n], realized_vol[:n]
        # RMSE-based score: lower error = higher reward
        rmse = np.sqrt(np.mean((predicted_vol - realized_vol) ** 2))
        mean_vol = np.mean(np.abs(realized_vol)) + 1e-10
        normalized_error = rmse / mean_vol
        # Convert to reward: 0 error = 1.0, 100% error = 0.0
        return float(np.clip(1.0 - normalized_error, -1.0, 1.0))


# ===========================================================================
# Process & novelty components
# ===========================================================================

class ProcessQualityReward(RewardComponent):
    """
    Reasoning process quality from ProcessRewardModel.

    Rewards good reasoning even when market outcomes are negative (stochastic).
    """

    def __init__(self, weight: float = 1.0, use_text_encoder: bool = False):
        super().__init__(weight)
        self._verifier: Optional[Any] = None
        self._use_text_encoder = use_text_encoder

    def _get_verifier(self):
        if self._verifier is None and _HAS_VERIFIER:
            self._verifier = ProcessRewardModel(use_text_encoder=self._use_text_encoder)
        return self._verifier

    @property
    def name(self) -> str:
        return "process_quality"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        reasoning_text = trajectory.get("reasoning_text", "")
        reasoning_chain = trajectory.get("reasoning_chain", None)
        market_data = trajectory.get("market_data", {})

        verifier = self._get_verifier()
        if verifier is None:
            return 0.0

        # Parse text into chain if needed
        if reasoning_chain is None and reasoning_text:
            reasoning_chain = verifier.parse_reasoning(reasoning_text)

        if reasoning_chain is None:
            return 0.0

        result = verifier.verify_chain(reasoning_chain, market_data)
        process_score = result.get("process_score", 0.0)
        # Normalize to [-1, 1] range (0.5 is neutral)
        return float(np.clip((process_score - 0.5) * 2.0, -1.0, 1.0))


class NoveltyReward(RewardComponent):
    """
    Bonus for contrarian signals that actually worked.

    Encourages the agent to find non-obvious opportunities by rewarding
    decisions that diverged from consensus but still generated positive
    outcomes. Without positive outcomes the bonus is zero (no reward for
    being contrarian and wrong).
    """

    def __init__(self, weight: float = 1.0, consensus_threshold: float = 0.6):
        super().__init__(weight)
        self.consensus_threshold = consensus_threshold

    @property
    def name(self) -> str:
        return "novelty"

    def compute(self, trajectory: Dict[str, Any]) -> float:
        # How much the decision diverged from consensus
        consensus_direction = trajectory.get("consensus_direction", 0.0)
        agent_direction = trajectory.get("agent_direction", 0.0)
        pnl = trajectory.get("pnl", 0.0)

        # Measure divergence
        if isinstance(consensus_direction, str):
            consensus_direction = {"buy": 1.0, "sell": -1.0, "hold": 0.0}.get(
                consensus_direction.lower(), 0.0
            )
        if isinstance(agent_direction, str):
            agent_direction = {"buy": 1.0, "sell": -1.0, "hold": 0.0}.get(
                agent_direction.lower(), 0.0
            )

        divergence = abs(agent_direction - consensus_direction) / 2.0

        if divergence < (1.0 - self.consensus_threshold):
            # Not contrarian enough
            return 0.0

        if pnl <= 0:
            # Contrarian and wrong — no bonus (but no penalty either)
            return 0.0

        # Scale bonus by divergence and success magnitude
        bonus = divergence * min(pnl / 0.05, 1.0)  # Cap at 5% PnL
        return float(np.clip(bonus, 0.0, 1.0))


# ===========================================================================
# Default component weights
# ===========================================================================

DEFAULT_COMPONENT_CONFIG = {
    # --- Outcome quality (40%) ---
    "return":          {"weight": 0.15, "cls": ReturnReward},
    "sharpe":          {"weight": 0.10, "cls": SharpeReward},
    "sortino":         {"weight": 0.05, "cls": SortinoReward},
    "treynor":         {"weight": 0.05, "cls": TreynorReward},
    "cvar":            {"weight": 0.05, "cls": CVaRReward},
    # --- Risk management (25%) ---
    "drawdown":        {"weight": 0.10, "cls": DrawdownReward},
    "turnover":        {"weight": 0.05, "cls": TurnoverReward},
    "theta_capture":   {"weight": 0.05, "cls": ThetaCaptureReward},
    "vol_prediction":  {"weight": 0.05, "cls": VolPredictionReward},
    # --- Process quality (25%) ---
    "process_quality": {"weight": 0.25, "cls": ProcessQualityReward},
    # --- Novelty (10%) ---
    "novelty":         {"weight": 0.10, "cls": NoveltyReward},
}


# ===========================================================================
# CompositeRewardFunction
# ===========================================================================

class CompositeRewardFunction:
    """
    Multi-objective reward combining all components.

    Based on Risk-Aware RL Reward (Jun 2025) research.  Computes a scalar
    total_reward as the weighted sum of individual components and returns
    a full breakdown for diagnostics and RLAIF preference generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Optional dict overriding default component weights.
                Keys are component names, values are dicts with optional
                'weight' and constructor kwargs.  Components with weight 0
                are disabled.
        """
        self.components: Dict[str, RewardComponent] = {}
        self._build_components(config or {})
        self._weight_history: List[Dict[str, float]] = []
        logger.info(
            "CompositeRewardFunction initialized with %d components: %s",
            len(self.components),
            ", ".join(f"{c.name}({c.weight:.2f})" for c in self.components.values()),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_components(self, config: Dict[str, Any]) -> None:
        """Instantiate and register components from config."""
        for comp_name, defaults in DEFAULT_COMPONENT_CONFIG.items():
            user = config.get(comp_name, {})
            weight = user.get("weight", defaults["weight"])
            if weight <= 0:
                continue
            cls = defaults["cls"]
            # Collect extra kwargs (exclude 'weight' and 'cls')
            kwargs = {k: v for k, v in user.items() if k not in ("weight", "cls")}
            kwargs["weight"] = weight
            try:
                self.components[comp_name] = cls(**kwargs)
            except TypeError as exc:
                logger.warning("Failed to create %s: %s", comp_name, exc)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        process_chain: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute the composite reward for a single decision/outcome pair.

        Args:
            decision: Dict describing the trading decision.  May contain:
                - agent_direction, consensus_direction, reasoning_text,
                  reasoning_chain, market_data, portfolio_weights, etc.
            outcome: Dict describing what happened.  Should contain:
                - returns (array), pnl, benchmark_returns, realized_vol,
                  predicted_vol, theta_pnl, expected_theta, turnover, etc.
            process_chain: Optional pre-parsed ReasoningChain.

        Returns:
            Dict with 'total_reward', 'component_scores', and 'breakdown'.
        """
        # Merge decision + outcome into a single trajectory dict
        trajectory = {**decision, **outcome}
        if process_chain is not None:
            trajectory["reasoning_chain"] = process_chain

        component_scores: Dict[str, float] = {}
        weighted_sum = 0.0

        for comp_name, component in self.components.items():
            try:
                raw_score = component.compute(trajectory)
            except Exception as exc:
                logger.debug("Component %s raised %s", comp_name, exc)
                raw_score = 0.0
            component_scores[comp_name] = raw_score
            weighted_sum += component.weight * raw_score

        # Build human-readable breakdown
        lines = []
        for cname, score in component_scores.items():
            w = self.components[cname].weight
            lines.append(f"  {cname:20s}: raw={score:+.4f}  w={w:.2f}  contrib={w * score:+.4f}")
        breakdown = "\n".join(lines)

        return {
            "total_reward": float(weighted_sum),
            "component_scores": component_scores,
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Batch compute
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        decisions: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
        process_chains: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Vectorized batch computation of composite rewards.

        Args:
            decisions: List of decision dicts.
            outcomes: List of outcome dicts (same length).
            process_chains: Optional list of ReasoningChain objects.

        Returns:
            1-D numpy array of total_reward values, shape (N,).
        """
        n = len(decisions)
        if n != len(outcomes):
            raise ValueError(
                f"decisions ({n}) and outcomes ({len(outcomes)}) must have same length"
            )
        chains = process_chains or [None] * n
        rewards = np.empty(n, dtype=np.float64)
        for i in range(n):
            result = self.compute(decisions[i], outcomes[i], chains[i])
            rewards[i] = result["total_reward"]
        return rewards

    # ------------------------------------------------------------------
    # Preference generation for RLAIF training
    # ------------------------------------------------------------------

    def generate_preference(
        self,
        decision_a: Dict[str, Any],
        outcome_a: Dict[str, Any],
        decision_b: Dict[str, Any],
        outcome_b: Dict[str, Any],
        process_chain_a: Optional[Any] = None,
        process_chain_b: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compare two decision/outcome pairs and produce a preference label.

        Returns:
            Dict with:
                - chosen: 'a' or 'b'
                - margin: absolute difference in total reward
                - reward_a, reward_b: full compute() results
                - reason: human-readable explanation of why one was chosen
        """
        result_a = self.compute(decision_a, outcome_a, process_chain_a)
        result_b = self.compute(decision_b, outcome_b, process_chain_b)

        ra = result_a["total_reward"]
        rb = result_b["total_reward"]
        margin = abs(ra - rb)
        chosen = "a" if ra >= rb else "b"

        # Build reason from top diverging components
        diffs: List[Tuple[str, float]] = []
        for cname in self.components:
            sa = result_a["component_scores"].get(cname, 0.0)
            sb = result_b["component_scores"].get(cname, 0.0)
            w = self.components[cname].weight
            diffs.append((cname, w * (sa - sb)))
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)

        reason_parts = []
        for cname, diff in diffs[:3]:
            direction = "favors A" if diff > 0 else "favors B"
            reason_parts.append(f"{cname} ({direction}, delta={diff:+.4f})")
        reason = f"Chosen={chosen.upper()} (margin={margin:.4f}). Top factors: " + "; ".join(
            reason_parts
        )

        return {
            "chosen": chosen,
            "margin": float(margin),
            "reward_a": result_a,
            "reward_b": result_b,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Dynamic weight adaptation
    # ------------------------------------------------------------------

    def adapt_weights(self, recent_performance: Dict[str, Any]) -> None:
        """
        Dynamically adjust component weights based on recent trading performance.

        Heuristics:
        - If Sharpe is low: increase risk_management weight (drawdown, turnover)
        - If win rate is low: increase process_quality weight
        - If returns are high but drawdown is bad: increase drawdown weight
        - If novelty has been scoring 0: decrease its weight and redistribute

        Args:
            recent_performance: Dict with keys like 'sharpe', 'win_rate',
                'max_drawdown', 'avg_return', 'novelty_hit_rate', etc.
        """
        # Snapshot current weights for history
        current_weights = {n: c.weight for n, c in self.components.items()}
        self._weight_history.append(current_weights.copy())

        sharpe = recent_performance.get("sharpe", 1.0)
        win_rate = recent_performance.get("win_rate", 0.5)
        max_dd = recent_performance.get("max_drawdown", 0.0)  # negative
        avg_return = recent_performance.get("avg_return", 0.0)
        novelty_hit = recent_performance.get("novelty_hit_rate", 0.5)

        adjustments: Dict[str, float] = {}

        # Low Sharpe -> increase risk components
        if sharpe < 0.5:
            for rname in ("drawdown", "turnover", "cvar"):
                if rname in self.components:
                    adjustments[rname] = adjustments.get(rname, 0.0) + 0.02
            for rname in ("return", "sharpe"):
                if rname in self.components:
                    adjustments[rname] = adjustments.get(rname, 0.0) - 0.01

        # Low win rate -> lean on process quality
        if win_rate < 0.4:
            if "process_quality" in self.components:
                adjustments["process_quality"] = adjustments.get("process_quality", 0.0) + 0.03
            if "return" in self.components:
                adjustments["return"] = adjustments.get("return", 0.0) - 0.02

        # Good returns but bad drawdown -> penalize drawdown harder
        if avg_return > 0.05 and max_dd < -0.15:
            if "drawdown" in self.components:
                adjustments["drawdown"] = adjustments.get("drawdown", 0.0) + 0.03

        # Novelty not contributing -> reduce and redistribute
        if novelty_hit < 0.1 and "novelty" in self.components:
            adjustments["novelty"] = adjustments.get("novelty", 0.0) - 0.03
            if "process_quality" in self.components:
                adjustments["process_quality"] = adjustments.get("process_quality", 0.0) + 0.02

        # Apply adjustments with floor
        min_weight = 0.01
        for comp_name, delta in adjustments.items():
            if comp_name in self.components:
                old_w = self.components[comp_name].weight
                new_w = max(old_w + delta, min_weight)
                self.components[comp_name].weight = new_w

        # Re-normalize so weights sum to 1.0
        total = sum(c.weight for c in self.components.values())
        if total > 0:
            for c in self.components.values():
                c.weight /= total

        # Log changes
        new_weights = {n: c.weight for n, c in self.components.items()}
        changes = {
            n: f"{current_weights.get(n, 0):.3f}->{new_weights[n]:.3f}"
            for n in new_weights
            if abs(new_weights[n] - current_weights.get(n, 0)) > 1e-4
        }
        if changes:
            logger.info("Adapted reward weights: %s", changes)
        else:
            logger.debug("No weight adaptation needed")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, float]:
        """Return current component weights."""
        return {n: c.weight for n, c in self.components.items()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Manually set component weights (will be normalized)."""
        for name, w in weights.items():
            if name in self.components:
                self.components[name].weight = max(w, 0.0)
        total = sum(c.weight for c in self.components.values())
        if total > 0:
            for c in self.components.values():
                c.weight /= total

    def summary(self) -> str:
        """Human-readable summary of the composite reward setup."""
        lines = ["CompositeRewardFunction"]
        lines.append(f"  Components: {len(self.components)}")

        # Group by category
        outcome = ["return", "sharpe", "sortino", "treynor", "cvar"]
        risk = ["drawdown", "turnover", "theta_capture", "vol_prediction"]
        process = ["process_quality"]
        novelty = ["novelty"]

        for group_name, group_keys in [
            ("Outcome Quality", outcome),
            ("Risk Management", risk),
            ("Process Quality", process),
            ("Novelty", novelty),
        ]:
            group_total = sum(
                self.components[k].weight for k in group_keys if k in self.components
            )
            lines.append(f"  {group_name} ({group_total:.0%}):")
            for k in group_keys:
                if k in self.components:
                    lines.append(f"    {k}: {self.components[k].weight:.1%}")

        return "\n".join(lines)
