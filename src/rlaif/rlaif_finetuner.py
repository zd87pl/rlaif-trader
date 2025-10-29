"""
RLAIF Fine-Tuner for Claude

Fine-tunes Claude agents using market outcome feedback via the reward model.

Approaches:
1. DPO (Direct Preference Optimization):
   - Directly optimizes LLM to prefer better decisions
   - More stable than PPO
   - Works directly with preference pairs

2. PPO (Proximal Policy Optimization):
   - Uses reward model to compute rewards
   - Optimizes via policy gradients
   - More flexible but requires more tuning

This implementation prepares training data for Claude's fine-tuning API and provides
local inference with the reward model for continuous improvement.

Key Innovation:
Market outcomes (P&L, Sharpe ratio) → Preference pairs → Reward model → Fine-tuned Claude
This creates a feedback loop where Claude learns from real trading results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .preference_generator import PreferencePair, TradingDecision
from .reward_model import RewardModel
from ..agents.claude_client import ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RLAIFFineTuner:
    """
    Fine-tunes Claude agents using RLAIF

    Process:
    1. Collect preference pairs from market outcomes
    2. Train reward model on preferences
    3. Generate fine-tuning data (chosen vs rejected examples)
    4. Fine-tune Claude via API or use reward model for inference-time guidance

    Two modes:
    - API Mode: Prepare data for Claude's official fine-tuning API
    - Local Mode: Use reward model to re-rank and guide agent outputs
    """

    def __init__(
        self,
        reward_model: RewardModel,
        text_encoder: SentenceTransformer,
        claude_client: Optional[ClaudeClient] = None,
        output_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize RLAIF fine-tuner

        Args:
            reward_model: Trained reward model
            text_encoder: Sentence transformer (same as used in reward model)
            claude_client: Claude API client (for fine-tuning)
            output_dir: Where to save fine-tuning data
            device: Device for reward model inference
        """
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        self.text_encoder = text_encoder
        self.claude_client = claude_client
        self.device = device

        self.output_dir = output_dir or Path("./data/rlaif/finetuning")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RLAIFFineTuner initialized with reward model on {device}")

    def prepare_dpo_training_data(
        self,
        preferences: List[PreferencePair],
        output_file: Optional[Path] = None,
        format: str = "anthropic",  # "anthropic", "openai", or "custom"
    ) -> Path:
        """
        Prepare training data for DPO fine-tuning

        DPO directly trains the model to prefer chosen over rejected completions.

        Args:
            preferences: Preference pairs
            output_file: Where to save (default: output_dir/dpo_training_data.jsonl)
            format: Output format for specific API

        Returns:
            Path to training data file
        """
        if output_file is None:
            output_file = self.output_dir / "dpo_training_data.jsonl"

        logger.info(f"Preparing DPO training data from {len(preferences)} preferences")

        training_examples = []

        for pref in preferences:
            # Create prompt from context
            prompt = self._create_prompt_from_decision(pref.chosen_decision)

            # Chosen completion (the better decision)
            chosen_completion = self._create_completion_from_decision(pref.chosen_decision)

            # Rejected completion (the worse decision)
            rejected_completion = self._create_completion_from_decision(pref.rejected_decision)

            if format == "anthropic":
                # Anthropic format (as per Claude fine-tuning API)
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen_completion},
                    ],
                    "rejected_messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected_completion},
                    ],
                    "metadata": {
                        "chosen_score": float(pref.chosen_score),
                        "rejected_score": float(pref.rejected_score),
                        "margin": float(pref.margin),
                        "confidence": float(pref.confidence),
                    },
                }
            elif format == "openai":
                # OpenAI format
                example = {
                    "prompt": prompt,
                    "chosen": chosen_completion,
                    "rejected": rejected_completion,
                }
            else:
                # Custom format
                example = {
                    "prompt": prompt,
                    "chosen": chosen_completion,
                    "rejected": rejected_completion,
                    "chosen_score": float(pref.chosen_score),
                    "rejected_score": float(pref.rejected_score),
                    "margin": float(pref.margin),
                    "confidence": float(pref.confidence),
                }

            training_examples.append(example)

        # Save to JSONL
        with open(output_file, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")

        logger.info(f"Saved {len(training_examples)} DPO examples to {output_file}")

        return output_file

    def prepare_supervised_training_data(
        self,
        preferences: List[PreferencePair],
        output_file: Optional[Path] = None,
        use_only_best: bool = True,
    ) -> Path:
        """
        Prepare training data for supervised fine-tuning

        Uses only the chosen (better) decisions as positive examples.

        Args:
            preferences: Preference pairs
            output_file: Where to save
            use_only_best: If True, only use chosen decisions; if False, use both with weights

        Returns:
            Path to training data file
        """
        if output_file is None:
            output_file = self.output_dir / "supervised_training_data.jsonl"

        logger.info(f"Preparing supervised training data from {len(preferences)} preferences")

        training_examples = []

        for pref in preferences:
            if use_only_best:
                # Only use chosen decisions
                prompt = self._create_prompt_from_decision(pref.chosen_decision)
                completion = self._create_completion_from_decision(pref.chosen_decision)

                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "metadata": {
                        "outcome_score": float(pref.chosen_score),
                        "confidence": float(pref.confidence),
                    },
                }
                training_examples.append(example)
            else:
                # Use both, weighted by outcome
                # Chosen (positive weight)
                prompt_chosen = self._create_prompt_from_decision(pref.chosen_decision)
                completion_chosen = self._create_completion_from_decision(pref.chosen_decision)

                training_examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt_chosen},
                            {"role": "assistant", "content": completion_chosen},
                        ],
                        "metadata": {
                            "outcome_score": float(pref.chosen_score),
                            "weight": 1.0,
                        },
                    }
                )

                # Rejected (negative example - some APIs support this)
                prompt_rejected = self._create_prompt_from_decision(pref.rejected_decision)
                completion_rejected = self._create_completion_from_decision(pref.rejected_decision)

                training_examples.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt_rejected},
                            {"role": "assistant", "content": completion_rejected},
                        ],
                        "metadata": {
                            "outcome_score": float(pref.rejected_score),
                            "weight": 0.1,  # Lower weight for bad examples
                        },
                    }
                )

        # Save to JSONL
        with open(output_file, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")

        logger.info(f"Saved {len(training_examples)} supervised examples to {output_file}")

        return output_file

    def score_decision(
        self,
        decision: TradingDecision,
    ) -> float:
        """
        Score a trading decision using the reward model

        This can be used for:
        - Inference-time guidance (re-ranking multiple candidates)
        - Validation
        - Online learning feedback

        Args:
            decision: Trading decision to score

        Returns:
            Predicted quality score
        """
        self.reward_model.eval()

        with torch.no_grad():
            # Extract features (same as training)
            features = self._extract_features(decision)

            numerical = features["numerical"].unsqueeze(0).to(self.device)  # [1, num_dim]
            text_emb = features["text_embedding"].unsqueeze(0).to(self.device)  # [1, text_dim]

            # Forward pass
            score = self.reward_model(numerical, text_emb)

            return float(score.item())

    def rerank_decisions(
        self,
        decisions: List[TradingDecision],
    ) -> List[Tuple[TradingDecision, float]]:
        """
        Re-rank multiple decisions using reward model

        Use case: Generate multiple candidate decisions from agents, then select best.

        Args:
            decisions: List of candidate decisions

        Returns:
            List of (decision, score) tuples, sorted by score (best first)
        """
        scored = [(d, self.score_decision(d)) for d in decisions]
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Re-ranked {len(decisions)} decisions: "
            f"best={scored[0][1]:.3f}, worst={scored[-1][1]:.3f}"
        )

        return scored

    def guided_sampling(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str],
        agent_name: str,
        num_samples: int = 5,
        temperature: float = 0.8,
    ) -> TradingDecision:
        """
        Generate multiple agent responses and select best using reward model

        This is a form of inference-time optimization - we don't need to fine-tune Claude,
        we just generate multiple candidates and pick the best.

        Args:
            symbol: Stock symbol
            data: Market data and features
            context: RAG context
            agent_name: Which agent to use
            num_samples: Number of candidates to generate
            temperature: Sampling temperature (higher = more diversity)

        Returns:
            Best decision according to reward model
        """
        if self.claude_client is None:
            raise ValueError("Claude client required for guided sampling")

        logger.info(f"Guided sampling: generating {num_samples} candidates")

        # Generate multiple candidates
        # NOTE: This requires access to the specific agent
        # For now, we'll create a placeholder
        # In production, you'd pass the actual agent and call it multiple times

        candidates = []
        # TODO: Implement actual multi-sampling with specific agent
        # This would require refactoring agents to support multiple samples

        logger.warning(
            "Guided sampling not fully implemented - requires agent access. "
            "Use rerank_decisions() with manually generated candidates."
        )

        return None

    # ===================================================================================
    # Helper Methods
    # ===================================================================================

    def _create_prompt_from_decision(self, decision: TradingDecision) -> str:
        """
        Create prompt from decision context

        This reconstructs the prompt that would have been sent to the agent
        """
        prompt = f"Analyze {decision.symbol} and provide trading recommendation.\n\n"

        # Market data
        prompt += "Market Data:\n"
        for key, value in decision.market_data.items():
            if isinstance(value, (int, float)):
                prompt += f"- {key}: {value:.2f}\n"
            else:
                prompt += f"- {key}: {value}\n"

        # Features
        if decision.features:
            prompt += "\nTechnical Indicators:\n"
            # Sample key features
            for key, value in list(decision.features.items())[:10]:
                if isinstance(value, (int, float)):
                    prompt += f"- {key}: {value:.2f}\n"

        # RAG context
        if decision.rag_context:
            prompt += f"\nRelevant Context:\n{decision.rag_context[:500]}\n"

        prompt += "\nProvide your analysis and recommendation."

        return prompt

    def _create_completion_from_decision(self, decision: TradingDecision) -> str:
        """
        Create completion from agent response

        This is what the agent actually output
        """
        completion = f"{decision.analysis}\n\n"

        completion += "Reasoning:\n"
        for i, step in enumerate(decision.agent_response.reasoning, 1):
            completion += f"{i}. {step}\n"

        completion += f"\nScore: {decision.agent_response.score:.2f}\n"
        completion += f"Confidence: {decision.agent_response.confidence:.0%}\n"

        return completion

    def _extract_features(self, decision: TradingDecision) -> Dict[str, torch.Tensor]:
        """
        Extract features from decision (same as reward model dataset)
        """
        # Numerical features
        numerical = []

        numerical.append(decision.agent_response.score)
        numerical.append(decision.agent_response.confidence)
        numerical.append(decision.position_size / 100.0)

        if "current_price" in decision.market_data:
            numerical.append(decision.market_data["current_price"] / 1000.0)

        if decision.features:
            for key, value in decision.features.items():
                if isinstance(value, (int, float)):
                    numerical.append(float(value))

        # Pad to fixed size
        target_size = 100
        if len(numerical) < target_size:
            numerical.extend([0.0] * (target_size - len(numerical)))
        else:
            numerical = numerical[:target_size]

        numerical_tensor = torch.tensor(numerical, dtype=torch.float32)

        # Text encoding
        text = f"{decision.agent_response.analysis}\n\nReasoning:\n"
        text += "\n".join(decision.agent_response.reasoning[:5])

        text_embedding = self.text_encoder.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        return {
            "numerical": numerical_tensor,
            "text_embedding": text_embedding,
        }

    def get_training_stats(self, preferences: List[PreferencePair]) -> Dict[str, Any]:
        """
        Get statistics about training data quality

        Args:
            preferences: Preference pairs

        Returns:
            Statistics dict
        """
        if not preferences:
            return {}

        margins = [p.margin for p in preferences]
        confidences = [p.confidence for p in preferences]
        chosen_scores = [p.chosen_score for p in preferences]
        rejected_scores = [p.rejected_score for p in preferences]

        return {
            "num_preferences": len(preferences),
            "avg_margin": float(np.mean(margins)),
            "std_margin": float(np.std(margins)),
            "avg_confidence": float(np.mean(confidences)),
            "avg_chosen_score": float(np.mean(chosen_scores)),
            "avg_rejected_score": float(np.mean(rejected_scores)),
            "score_separation": float(np.mean(chosen_scores) - np.mean(rejected_scores)),
        }
