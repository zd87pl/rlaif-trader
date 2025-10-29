"""
Preference Generator for RLAIF

Creates preference pairs from actual trading outcomes. This is the bridge between
market reality and model training.

Process:
1. Store trading decisions with full context
2. Track actual market outcomes over time
3. Generate preference pairs (good decision vs bad decision)
4. Weight preferences by outcome quality and confidence

Key Metrics for Preference:
- Total return (profit/loss)
- Risk-adjusted return (Sharpe, Sortino)
- Maximum drawdown
- Win rate
- Time to target
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.base_agent import AgentResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradingDecision:
    """
    A single trading decision with full context

    This captures everything needed to evaluate and learn from a decision
    """

    decision_id: str
    timestamp: datetime
    symbol: str

    # Agent outputs
    agent_response: AgentResponse
    action: str  # "buy", "sell", "hold"
    position_size: float

    # Context at decision time
    market_data: Dict[str, Any]
    features: Dict[str, Any]
    rag_context: Optional[str] = None

    # Outcome tracking (filled later)
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    realized_pnl: Optional[float] = None
    realized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    outcome_computed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage"""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "agent_response": {
                "analysis": self.agent_response.analysis,
                "score": self.agent_response.score,
                "confidence": self.agent_response.confidence,
                "reasoning": self.agent_response.reasoning,
                "agent_name": self.agent_response.agent_name,
            },
            "action": self.action,
            "position_size": self.position_size,
            "market_data": self.market_data,
            "features": self.features,
            "rag_context": self.rag_context,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "realized_pnl": self.realized_pnl,
            "realized_return": self.realized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "outcome_computed": self.outcome_computed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingDecision":
        """Deserialize from dict"""
        # Reconstruct AgentResponse
        agent_data = data["agent_response"]
        agent_response = AgentResponse(
            analysis=agent_data["analysis"],
            score=agent_data["score"],
            confidence=agent_data["confidence"],
            reasoning=agent_data["reasoning"],
            data={},
            agent_name=agent_data["agent_name"],
        )

        return cls(
            decision_id=data["decision_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            agent_response=agent_response,
            action=data["action"],
            position_size=data["position_size"],
            market_data=data["market_data"],
            features=data["features"],
            rag_context=data.get("rag_context"),
            entry_price=data.get("entry_price"),
            exit_price=data.get("exit_price"),
            exit_timestamp=datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None,
            realized_pnl=data.get("realized_pnl"),
            realized_return=data.get("realized_return"),
            max_drawdown=data.get("max_drawdown"),
            sharpe_ratio=data.get("sharpe_ratio"),
            outcome_computed=data.get("outcome_computed", False),
        )


@dataclass
class PreferencePair:
    """
    A preference pair for training

    Contains two decisions where one clearly outperformed the other
    """

    pair_id: str
    timestamp: datetime

    # The better decision (chosen)
    chosen_decision: TradingDecision
    chosen_score: float  # Composite outcome score

    # The worse decision (rejected)
    rejected_decision: TradingDecision
    rejected_score: float

    # Preference strength (how much better was chosen vs rejected)
    margin: float
    confidence: float

    # Metadata
    comparison_metric: str  # "sharpe", "return", "risk_adjusted", etc.
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "pair_id": self.pair_id,
            "timestamp": self.timestamp.isoformat(),
            "chosen_decision": self.chosen_decision.to_dict(),
            "chosen_score": self.chosen_score,
            "rejected_decision": self.rejected_decision.to_dict(),
            "rejected_score": self.rejected_score,
            "margin": self.margin,
            "confidence": self.confidence,
            "comparison_metric": self.comparison_metric,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        """Deserialize from dict"""
        return cls(
            pair_id=data["pair_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            chosen_decision=TradingDecision.from_dict(data["chosen_decision"]),
            chosen_score=data["chosen_score"],
            rejected_decision=TradingDecision.from_dict(data["rejected_decision"]),
            rejected_score=data["rejected_score"],
            margin=data["margin"],
            confidence=data["confidence"],
            comparison_metric=data["comparison_metric"],
            notes=data.get("notes"),
        )


class PreferenceGenerator:
    """
    Generates preference pairs from trading outcomes

    This is the core of RLAIF - converting market feedback into training data

    Process:
    1. Record decisions as they are made
    2. Track outcomes over time
    3. Compare decisions and generate preferences
    4. Store preferences for training

    Key Features:
    - Multiple comparison metrics (return, Sharpe, risk-adjusted)
    - Confidence weighting based on statistical significance
    - Temporal alignment (compare similar market conditions)
    - Outlier handling
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        min_preference_margin: float = 0.05,  # 5% minimum difference
        hold_period_days: int = 5,  # Default hold period for evaluation
    ):
        """
        Initialize preference generator

        Args:
            storage_path: Where to store decisions and preferences
            min_preference_margin: Minimum performance difference to create preference
            hold_period_days: How long to hold positions for outcome evaluation
        """
        self.storage_path = storage_path or Path("./data/rlaif")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.min_preference_margin = min_preference_margin
        self.hold_period_days = hold_period_days

        # In-memory storage
        self.decisions: List[TradingDecision] = []
        self.preferences: List[PreferencePair] = []

        # Load existing data
        self._load_from_disk()

        logger.info(
            f"PreferenceGenerator initialized: {len(self.decisions)} decisions, "
            f"{len(self.preferences)} preferences"
        )

    def record_decision(
        self,
        symbol: str,
        agent_response: AgentResponse,
        action: str,
        position_size: float,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        rag_context: Optional[str] = None,
        entry_price: Optional[float] = None,
    ) -> TradingDecision:
        """
        Record a trading decision

        Args:
            symbol: Stock symbol
            agent_response: Multi-agent decision output
            action: "buy", "sell", or "hold"
            position_size: Size of position (shares or %)
            market_data: Market data at decision time
            features: Computed features
            rag_context: RAG context used
            entry_price: Entry price if available

        Returns:
            TradingDecision object
        """
        decision_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        decision = TradingDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            symbol=symbol,
            agent_response=agent_response,
            action=action,
            position_size=position_size,
            market_data=market_data,
            features=features,
            rag_context=rag_context,
            entry_price=entry_price or market_data.get("close"),
        )

        self.decisions.append(decision)
        self._save_decision(decision)

        logger.info(f"Recorded decision {decision_id}: {action} {symbol} @ {entry_price}")

        return decision

    def update_outcome(
        self,
        decision_id: str,
        exit_price: float,
        price_history: Optional[pd.Series] = None,
    ) -> None:
        """
        Update decision with actual outcome

        Args:
            decision_id: ID of decision to update
            exit_price: Exit price
            price_history: Price history during hold period (for max drawdown, Sharpe)
        """
        # Find decision
        decision = next((d for d in self.decisions if d.decision_id == decision_id), None)
        if not decision:
            logger.warning(f"Decision {decision_id} not found")
            return

        # Compute realized metrics
        if decision.entry_price is None:
            logger.warning(f"Decision {decision_id} has no entry price")
            return

        decision.exit_price = exit_price
        decision.exit_timestamp = datetime.now()
        decision.realized_pnl = (exit_price - decision.entry_price) * decision.position_size
        decision.realized_return = (exit_price - decision.entry_price) / decision.entry_price

        # Compute risk metrics if we have price history
        if price_history is not None and len(price_history) > 1:
            # Max drawdown
            cumulative_returns = (price_history / decision.entry_price) - 1
            running_max = cumulative_returns.cummax()
            drawdowns = cumulative_returns - running_max
            decision.max_drawdown = drawdowns.min()

            # Sharpe ratio (annualized)
            returns = price_history.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                decision.sharpe_ratio = (
                    returns.mean() / returns.std() * np.sqrt(252)
                )
            else:
                decision.sharpe_ratio = 0.0
        else:
            decision.max_drawdown = 0.0
            decision.sharpe_ratio = 0.0

        decision.outcome_computed = True
        self._save_decision(decision)

        logger.info(
            f"Updated outcome for {decision_id}: Return={decision.realized_return:.2%}, "
            f"Sharpe={decision.sharpe_ratio:.2f}, MaxDD={decision.max_drawdown:.2%}"
        )

    def generate_preferences(
        self,
        comparison_metric: str = "risk_adjusted",
        min_samples: int = 10,
    ) -> List[PreferencePair]:
        """
        Generate preference pairs from decisions with outcomes

        Args:
            comparison_metric: How to compare decisions
                - "return": Raw return
                - "sharpe": Sharpe ratio
                - "risk_adjusted": Composite of return and risk
            min_samples: Minimum number of decisions needed

        Returns:
            List of new preference pairs
        """
        # Filter decisions with computed outcomes
        completed = [d for d in self.decisions if d.outcome_computed]

        if len(completed) < min_samples:
            logger.info(
                f"Not enough completed decisions ({len(completed)} < {min_samples})"
            )
            return []

        logger.info(f"Generating preferences from {len(completed)} completed decisions")

        new_pairs = []

        # Group by symbol for fair comparison
        by_symbol = {}
        for d in completed:
            if d.symbol not in by_symbol:
                by_symbol[d.symbol] = []
            by_symbol[d.symbol].append(d)

        # Generate pairs within each symbol
        for symbol, decisions in by_symbol.items():
            if len(decisions) < 2:
                continue

            # Compute scores for each decision
            scores = []
            for d in decisions:
                if comparison_metric == "return":
                    score = d.realized_return
                elif comparison_metric == "sharpe":
                    score = d.sharpe_ratio
                elif comparison_metric == "risk_adjusted":
                    # Composite: return / (1 + abs(max_drawdown))
                    score = d.realized_return / (1 + abs(d.max_drawdown or 0))
                else:
                    raise ValueError(f"Unknown metric: {comparison_metric}")
                scores.append(score)

            # Sort by score
            sorted_indices = np.argsort(scores)

            # Create pairs: top vs bottom
            # We pair the best decisions with worst decisions for maximum signal
            n_pairs = min(len(decisions) // 2, 5)  # Limit pairs per symbol

            for i in range(n_pairs):
                top_idx = sorted_indices[-(i + 1)]
                bottom_idx = sorted_indices[i]

                chosen = decisions[top_idx]
                rejected = decisions[bottom_idx]

                chosen_score = scores[top_idx]
                rejected_score = scores[bottom_idx]

                # Check margin
                margin = chosen_score - rejected_score
                if margin < self.min_preference_margin:
                    continue

                # Compute confidence based on outcome quality
                confidence = self._compute_confidence(chosen, rejected, margin)

                pair_id = f"pref_{len(self.preferences) + len(new_pairs)}"
                pair = PreferencePair(
                    pair_id=pair_id,
                    timestamp=datetime.now(),
                    chosen_decision=chosen,
                    chosen_score=chosen_score,
                    rejected_decision=rejected,
                    rejected_score=rejected_score,
                    margin=margin,
                    confidence=confidence,
                    comparison_metric=comparison_metric,
                )

                new_pairs.append(pair)
                self.preferences.append(pair)
                self._save_preference(pair)

        logger.info(f"Generated {len(new_pairs)} new preference pairs")

        return new_pairs

    def _compute_confidence(
        self,
        chosen: TradingDecision,
        rejected: TradingDecision,
        margin: float,
    ) -> float:
        """
        Compute confidence in preference based on:
        - Margin size (larger = more confident)
        - Agent confidence (higher = more confident)
        - Outcome consistency (both agents agreed on direction = more confident)
        """
        # Margin contribution (sigmoid scaled)
        margin_conf = 1 / (1 + np.exp(-10 * (margin - 0.1)))

        # Agent confidence contribution
        agent_conf = (chosen.agent_response.confidence + rejected.agent_response.confidence) / 2

        # Direction agreement (did both agents agree on action?)
        direction_agreement = 1.0 if chosen.action == rejected.action else 0.5

        # Weighted combination
        confidence = 0.5 * margin_conf + 0.3 * agent_conf + 0.2 * direction_agreement

        return float(np.clip(confidence, 0.0, 1.0))

    def get_training_data(
        self,
        min_confidence: float = 0.3,
        max_samples: Optional[int] = None,
    ) -> List[PreferencePair]:
        """
        Get preference pairs ready for training

        Args:
            min_confidence: Minimum confidence threshold
            max_samples: Maximum samples to return

        Returns:
            Filtered preference pairs
        """
        # Filter by confidence
        filtered = [p for p in self.preferences if p.confidence >= min_confidence]

        # Sort by confidence (highest first)
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        # Limit samples
        if max_samples is not None:
            filtered = filtered[:max_samples]

        logger.info(
            f"Training data: {len(filtered)} pairs (min_conf={min_confidence:.2f})"
        )

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        completed = [d for d in self.decisions if d.outcome_computed]

        if completed:
            returns = [d.realized_return for d in completed]
            sharpes = [d.sharpe_ratio for d in completed if d.sharpe_ratio is not None]
        else:
            returns = []
            sharpes = []

        return {
            "total_decisions": len(self.decisions),
            "completed_decisions": len(completed),
            "total_preferences": len(self.preferences),
            "avg_return": np.mean(returns) if returns else 0.0,
            "avg_sharpe": np.mean(sharpes) if sharpes else 0.0,
            "avg_preference_confidence": (
                np.mean([p.confidence for p in self.preferences])
                if self.preferences
                else 0.0
            ),
        }

    # ==================================================================================
    # Persistence
    # ==================================================================================

    def _save_decision(self, decision: TradingDecision) -> None:
        """Save decision to disk"""
        path = self.storage_path / "decisions" / f"{decision.decision_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(decision.to_dict(), f, indent=2)

    def _save_preference(self, preference: PreferencePair) -> None:
        """Save preference to disk"""
        path = self.storage_path / "preferences" / f"{preference.pair_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(preference.to_dict(), f, indent=2)

    def _load_from_disk(self) -> None:
        """Load existing decisions and preferences from disk"""
        # Load decisions
        decisions_path = self.storage_path / "decisions"
        if decisions_path.exists():
            for file in decisions_path.glob("*.json"):
                with open(file) as f:
                    data = json.load(f)
                    self.decisions.append(TradingDecision.from_dict(data))

        # Load preferences
        preferences_path = self.storage_path / "preferences"
        if preferences_path.exists():
            for file in preferences_path.glob("*.json"):
                with open(file) as f:
                    data = json.load(f)
                    self.preferences.append(PreferencePair.from_dict(data))
