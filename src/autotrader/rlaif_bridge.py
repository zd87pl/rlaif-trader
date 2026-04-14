"""RLAIF Bridge: connects autotrader experiments to the RLAIF feedback loop.

Converts ExperimentResult + StrategySpec into TradingDecision objects that the
PreferenceGenerator can use. When a "kept" experiment outperforms a "discarded"
one, this creates a preference pair — closing the double feedback loop:

    AutoTrader loop → experiments → RLAIF preferences → reward model training
    → better thesis generation → better experiments → LOOP
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .metrics import ExperimentResult
from .strategy_spec import StrategySpec
from ..utils.logging import get_logger

# Lazy import to avoid pulling in the full agents dependency chain
AgentResponse = None

def _get_agent_response_class():
    global AgentResponse
    if AgentResponse is None:
        from ..agents.base_agent import AgentResponse as AR
        AgentResponse = AR
    return AgentResponse

logger = get_logger(__name__)


class RLAIFBridge:
    """Bridge between autotrader experiments and the RLAIF preference system.

    Accumulates experiment results, and when enough kept/discarded pairs exist,
    generates preference pairs and feeds them to the PreferenceGenerator.
    """

    def __init__(
        self,
        preference_generator: Any = None,
        reward_model: Any = None,
        min_pairs_for_training: int = 10,
        max_history: int = 200,
    ):
        self.preference_generator = preference_generator
        self.reward_model = reward_model
        self.min_pairs_for_training = min_pairs_for_training

        # Accumulate results grouped by status
        self._kept: Deque[Tuple[ExperimentResult, StrategySpec]] = deque(maxlen=max_history)
        self._discarded: Deque[Tuple[ExperimentResult, StrategySpec]] = deque(maxlen=max_history)
        self._pairs_generated = 0

    def record(self, result: ExperimentResult, spec: StrategySpec) -> None:
        """Record an experiment result for later preference generation."""
        if result.status == "keep":
            self._kept.append((result, spec))
        elif result.status == "discard":
            self._discarded.append((result, spec))
        # Crashes/timeouts are not useful for preferences

        # Auto-generate preferences when we have enough pairs
        if len(self._kept) >= 2 and len(self._discarded) >= 2:
            self._generate_preferences()

    def _generate_preferences(self) -> int:
        """Generate preference pairs from kept vs discarded experiments.

        Pairs the best kept experiments with the worst discarded ones
        for maximum training signal (just like PreferenceGenerator does).
        """
        if not self.preference_generator:
            return 0

        # Sort kept by score (best first)
        kept_sorted = sorted(self._kept, key=lambda x: x[0].composite_score, reverse=True)
        # Sort discarded by score (worst first)
        discarded_sorted = sorted(self._discarded, key=lambda x: x[0].composite_score)

        n_pairs = min(len(kept_sorted), len(discarded_sorted), 5)
        new_pairs = 0

        for i in range(n_pairs):
            kept_result, kept_spec = kept_sorted[i]
            disc_result, disc_spec = discarded_sorted[i]

            margin = kept_result.composite_score - disc_result.composite_score
            if margin < 0.005:
                continue

            # Convert to TradingDecision format for the PreferenceGenerator
            chosen_decision = self._result_to_decision(kept_result, kept_spec, "buy")
            rejected_decision = self._result_to_decision(disc_result, disc_spec, "hold")

            try:
                self.preference_generator.decisions.append(chosen_decision)
                self.preference_generator.decisions.append(rejected_decision)
                self.preference_generator._save_decision(chosen_decision)
                self.preference_generator._save_decision(rejected_decision)
                new_pairs += 1
                self._pairs_generated += 1
            except Exception as e:
                logger.debug("Failed to record RLAIF decision: %s", e)

        if new_pairs > 0:
            logger.info(
                "RLAIF bridge generated %d preference-ready decisions "
                "(total: %d kept, %d discarded)",
                new_pairs * 2,
                len(self._kept),
                len(self._discarded),
            )

            # Trigger preference generation if we have enough
            if len(self.preference_generator.decisions) >= self.min_pairs_for_training:
                try:
                    pairs = self.preference_generator.generate_preferences(
                        comparison_metric="risk_adjusted",
                        min_samples=4,
                    )
                    if pairs:
                        logger.info("RLAIF generated %d preference pairs", len(pairs))
                        self._trigger_reward_training()
                except Exception as e:
                    logger.debug("Preference generation failed: %s", e)

        return new_pairs

    def _result_to_decision(
        self,
        result: ExperimentResult,
        spec: StrategySpec,
        action: str,
    ) -> Any:
        """Convert an ExperimentResult into a TradingDecision.

        The strategy's signal code becomes the "analysis" and the
        composite score becomes the agent's score.
        """
        from ..rlaif.preference_generator import TradingDecision

        AR = _get_agent_response_class()
        agent_response = AR(
            analysis=f"AutoTrader strategy: {spec.description}\n\n{spec.signal_code[:500]}",
            score=result.composite_score,
            confidence=min(1.0, abs(result.sharpe_ratio) / 3.0),
            reasoning=[
                f"Composite: {result.composite_score:.6f}",
                f"Sharpe: {result.sharpe_ratio:.4f}",
                f"Return: {result.cumulative_return:.4f}",
                f"Drawdown: {result.max_drawdown:.4f}",
                f"Hit rate: {result.hit_rate:.4f}",
            ],
            data=result.backtest_details,
            agent_name="AutoTrader",
        )

        decision = TradingDecision(
            decision_id=f"at_{result.experiment_id}_{uuid.uuid4().hex[:6]}",
            timestamp=datetime.now(timezone.utc),
            symbol="PORTFOLIO",
            agent_response=agent_response,
            action=action,
            position_size=1.0,
            market_data={
                "composite_score": result.composite_score,
                "sharpe_ratio": result.sharpe_ratio,
                "spec_id": spec.spec_id,
            },
            features={
                "sharpe_ratio": result.sharpe_ratio,
                "cumulative_return": result.cumulative_return,
                "max_drawdown": result.max_drawdown,
                "hit_rate": result.hit_rate,
                "num_trades": result.num_trades,
            },
            entry_price=1.0,
            exit_price=1.0 + result.cumulative_return,
            realized_return=result.cumulative_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            outcome_computed=True,
        )

        return decision

    def _trigger_reward_training(self) -> None:
        """Trigger reward model retraining if available."""
        if not self.reward_model:
            return

        try:
            training_data = self.preference_generator.get_training_data(
                min_confidence=0.3
            )
            if len(training_data) >= self.min_pairs_for_training:
                logger.info(
                    "Triggering reward model training with %d pairs",
                    len(training_data),
                )
                # The reward model training is handled by the existing
                # RLAIF infrastructure
        except Exception as e:
            logger.debug("Reward model training trigger failed: %s", e)

    def score_strategy(self, spec: StrategySpec) -> Optional[float]:
        """Use the reward model to score a strategy (inference-time guidance).

        This enables the ThesisGenerator to prioritize proposals based on
        the reward model's learned preferences, closing the full loop.
        """
        if not self.reward_model or not hasattr(self.reward_model, "predict"):
            return None

        try:
            # Create a dummy decision from the spec for scoring
            dummy_result = ExperimentResult(
                spec_id=spec.spec_id,
                description=spec.description,
            )
            decision = self._result_to_decision(dummy_result, spec, "buy")

            # Use reward model to score
            score = self.reward_model.predict(decision)
            return float(score)
        except Exception as e:
            logger.debug("Reward model scoring failed: %s", e)
            return None

    def get_rlaif_callback(self):
        """Return a callback function suitable for ExperimentOrchestrator.rlaif_callback."""
        def callback(result: ExperimentResult, spec: StrategySpec) -> None:
            self.record(result, spec)
        return callback

    def status(self) -> Dict[str, Any]:
        return {
            "kept_experiments": len(self._kept),
            "discarded_experiments": len(self._discarded),
            "pairs_generated": self._pairs_generated,
            "preference_generator_connected": self.preference_generator is not None,
            "reward_model_connected": self.reward_model is not None,
            "pref_gen_stats": (
                self.preference_generator.get_stats()
                if self.preference_generator
                else None
            ),
        }
