"""
Strategy ensemble — combines multiple strategies into one signal.

This is where the edge compounds. Individual strategies are mediocre.
An ensemble of uncorrelated strategies is the actual product.

Aggregation modes:
- weighted_average: Weight by strategy weight * confidence
- majority_vote: Majority action wins, score is average
- conviction_weighted: Only trade when strategies agree with high confidence
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Strategy, Signal
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StrategyEnsemble:
    """
    Combines multiple strategies into a single signal per symbol.

    The ensemble is NOT a Strategy subclass — it orchestrates strategies
    and produces the final signal that goes to risk management.
    """

    def __init__(
        self,
        strategies: List[Strategy],
        mode: str = "conviction_weighted",
        min_agreement: float = 0.5,
        min_ensemble_confidence: float = 0.4,
    ):
        """
        Args:
            strategies: List of Strategy instances
            mode: Aggregation mode ("weighted_average", "majority_vote", "conviction_weighted")
            min_agreement: Min fraction of strategies that must agree on direction (for conviction mode)
            min_ensemble_confidence: Min ensemble confidence to generate non-hold signal
        """
        self.strategies = strategies
        self.mode = mode
        self.min_agreement = min_agreement
        self.min_ensemble_confidence = min_ensemble_confidence

        logger.info(
            f"Ensemble initialized: {len(strategies)} strategies, mode={mode}"
        )
        for s in strategies:
            logger.info(f"  - {s.name} (weight={s.weight})")

    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        news_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Run all strategies and combine into one signal.
        """
        # Collect signals from all strategies
        signals: List[Signal] = []
        for strategy in self.strategies:
            try:
                sig = strategy.generate_signal(
                    symbol=symbol,
                    price_data=price_data,
                    features=features,
                    news_data=news_data,
                    fundamental_data=fundamental_data,
                )
                signals.append(sig)
                logger.info(
                    f"  {strategy.name}: {sig.action} score={sig.score:+.2f} "
                    f"conf={sig.confidence:.0%}"
                )
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed for {symbol}: {e}")

        if not signals:
            return Signal(
                symbol=symbol,
                action="hold",
                score=0.0,
                confidence=0.0,
                strategy_name="ensemble",
                reasoning="No strategies produced signals",
            )

        if self.mode == "weighted_average":
            return self._weighted_average(symbol, signals)
        elif self.mode == "majority_vote":
            return self._majority_vote(symbol, signals)
        elif self.mode == "conviction_weighted":
            return self._conviction_weighted(symbol, signals)
        else:
            return self._weighted_average(symbol, signals)

    def _weighted_average(self, symbol: str, signals: List[Signal]) -> Signal:
        """Weighted average of all strategy scores."""
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        reasons = []

        for sig in signals:
            strategy = next(
                (s for s in self.strategies if s.name == sig.strategy_name), None
            )
            w = strategy.weight if strategy else 1.0
            effective_weight = w * sig.confidence

            weighted_score += sig.score * effective_weight
            weighted_confidence += sig.confidence * w
            total_weight += effective_weight
            reasons.append(f"{sig.strategy_name}={sig.action}({sig.score:+.2f})")

        if total_weight > 0:
            score = weighted_score / total_weight
            confidence = weighted_confidence / sum(
                (next((s for s in self.strategies if s.name == sig.strategy_name), None) or sig).weight
                if hasattr(sig, 'strategy_name') else 1.0
                for sig in signals
            )
        else:
            score = 0.0
            confidence = 0.0

        score = max(-1.0, min(1.0, score))
        confidence = min(confidence, 1.0)

        if confidence < self.min_ensemble_confidence:
            action = "hold"
        elif score >= 0.3:
            action = "buy"
        elif score <= -0.3:
            action = "sell"
        else:
            action = "hold"

        return Signal(
            symbol=symbol,
            action=action,
            score=score,
            confidence=confidence,
            strategy_name="ensemble",
            reasoning=f"Weighted avg: {'; '.join(reasons)}",
            metadata={"individual_signals": [
                {"strategy": s.strategy_name, "action": s.action, "score": s.score, "confidence": s.confidence}
                for s in signals
            ]},
        )

    def _majority_vote(self, symbol: str, signals: List[Signal]) -> Signal:
        """Majority vote on action, average score of winners."""
        votes = {"buy": 0, "sell": 0, "hold": 0}
        for sig in signals:
            votes[sig.action] += 1

        winning_action = max(votes, key=votes.get)
        winning_signals = [s for s in signals if s.action == winning_action]

        score = np.mean([s.score for s in winning_signals])
        confidence = np.mean([s.confidence for s in winning_signals])

        agreement = votes[winning_action] / len(signals)
        if agreement < self.min_agreement:
            winning_action = "hold"
            score = 0.0

        reasons = [f"{s.strategy_name}={s.action}({s.score:+.2f})" for s in signals]

        return Signal(
            symbol=symbol,
            action=winning_action,
            score=float(score),
            confidence=float(confidence),
            strategy_name="ensemble",
            reasoning=f"Vote ({votes}): {'; '.join(reasons)}",
            metadata={"votes": votes, "agreement": agreement},
        )

    def _conviction_weighted(self, symbol: str, signals: List[Signal]) -> Signal:
        """
        Only trade when multiple strategies agree with conviction.
        This is the most conservative and highest-quality mode.
        """
        buy_signals = [s for s in signals if s.action == "buy"]
        sell_signals = [s for s in signals if s.action == "sell"]

        total = len(signals)
        buy_frac = len(buy_signals) / total if total > 0 else 0
        sell_frac = len(sell_signals) / total if total > 0 else 0

        reasons = [f"{s.strategy_name}={s.action}({s.score:+.2f})" for s in signals]

        # Need minimum agreement
        if buy_frac >= self.min_agreement:
            # Weight by strategy weight and confidence
            weights = []
            scores = []
            for sig in buy_signals:
                strategy = next(
                    (s for s in self.strategies if s.name == sig.strategy_name), None
                )
                w = (strategy.weight if strategy else 1.0) * sig.confidence
                weights.append(w)
                scores.append(sig.score)

            total_w = sum(weights)
            score = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0
            confidence = buy_frac * np.mean([s.confidence for s in buy_signals])

            if confidence >= self.min_ensemble_confidence:
                return Signal(
                    symbol=symbol,
                    action="buy",
                    score=float(max(0.3, min(1.0, score))),
                    confidence=float(min(confidence, 1.0)),
                    strategy_name="ensemble",
                    reasoning=f"Conviction buy ({buy_frac:.0%} agree): {'; '.join(reasons)}",
                    metadata={"agreement": buy_frac},
                )

        elif sell_frac >= self.min_agreement:
            weights = []
            scores = []
            for sig in sell_signals:
                strategy = next(
                    (s for s in self.strategies if s.name == sig.strategy_name), None
                )
                w = (strategy.weight if strategy else 1.0) * sig.confidence
                weights.append(w)
                scores.append(sig.score)

            total_w = sum(weights)
            score = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0
            confidence = sell_frac * np.mean([s.confidence for s in sell_signals])

            if confidence >= self.min_ensemble_confidence:
                return Signal(
                    symbol=symbol,
                    action="sell",
                    score=float(min(-0.3, max(-1.0, score))),
                    confidence=float(min(confidence, 1.0)),
                    strategy_name="ensemble",
                    reasoning=f"Conviction sell ({sell_frac:.0%} agree): {'; '.join(reasons)}",
                    metadata={"agreement": sell_frac},
                )

        # No agreement — hold
        avg_score = np.mean([s.score for s in signals])
        return Signal(
            symbol=symbol,
            action="hold",
            score=float(avg_score),
            confidence=float(np.mean([s.confidence for s in signals]) * 0.5),
            strategy_name="ensemble",
            reasoning=f"No conviction (buy={buy_frac:.0%}, sell={sell_frac:.0%}): {'; '.join(reasons)}",
            metadata={"buy_agreement": buy_frac, "sell_agreement": sell_frac},
        )
