#!/usr/bin/env python3
"""
Example: Reward Model Guided Inference

Instead of fine-tuning Claude, use the trained reward model to guide decisions
at inference time. This is faster and doesn't require API fine-tuning.

Process:
1. Generate multiple candidate decisions from agents (vary temperature/prompts)
2. Score each candidate with the reward model
3. Select the best candidate
4. Execute and track outcome

Benefits:
- No fine-tuning needed
- Immediate deployment
- Can update reward model anytime
- Lower cost than fine-tuning

Trade-offs:
- Slightly higher inference cost (multiple calls)
- Not as deeply integrated as fine-tuning
- Requires keeping reward model in memory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import numpy as np
import torch

from src.agents import ClaudeClient, ManagerAgent, AgentResponse
from src.rlaif import RLAIFFineTuner
from src.rlaif.reward_model import create_reward_model, RewardModelTrainer
from src.rlaif.preference_generator import TradingDecision
from src.utils import setup_logging, set_seed

load_dotenv()
logger = setup_logging(log_level="INFO")


def generate_mock_decision(symbol: str, temperature: float = 0.7) -> TradingDecision:
    """
    Generate a mock trading decision

    In production, this would call the actual multi-agent system with
    varied temperature/prompts to get different candidates.
    """
    # Simulate different agent outputs based on temperature
    base_score = np.random.uniform(-0.5, 0.5)
    noise = np.random.randn() * temperature * 0.3
    score = np.clip(base_score + noise, -1, 1)

    confidence = np.random.uniform(0.5, 0.9)

    current_price = 175.0

    agent_response = AgentResponse(
        analysis=f"Analysis for {symbol} with temperature {temperature:.2f}. "
        f"Market conditions show {'bullish' if score > 0 else 'bearish'} signals.",
        score=float(score),
        confidence=float(confidence),
        reasoning=[
            "Technical indicators show momentum",
            "Fundamental valuations are reasonable",
            "Sentiment is moderately positive",
        ],
        data={},
        agent_name="ManagerAgent",
    )

    action = "buy" if score > 0.2 else ("sell" if score < -0.2 else "hold")

    decision = TradingDecision(
        decision_id=f"{symbol}_{np.random.randint(10000)}",
        timestamp=pd.Timestamp.now(),
        symbol=symbol,
        agent_response=agent_response,
        action=action,
        position_size=100,
        market_data={"close": current_price, "volume": 50_000_000},
        features={
            "rsi": np.random.uniform(30, 70),
            "macd": np.random.uniform(-5, 5),
        },
        entry_price=current_price,
    )

    return decision


def main():
    """
    Demonstrate reward model guided inference
    """
    logger.info("=" * 80)
    logger.info("REWARD MODEL GUIDED INFERENCE")
    logger.info("=" * 80)

    set_seed(42)

    # =========================================================================
    # Step 1: Load Trained Reward Model
    # =========================================================================
    logger.info("\n[Step 1] Loading Reward Model")
    logger.info("-" * 80)

    # Check if model exists
    model_path = Path("./models/reward_model.pt")

    if model_path.exists():
        logger.info(f"Loading trained model from {model_path}")

        # Create model
        reward_model, text_encoder = create_reward_model()

        # Load weights
        checkpoint = torch.load(model_path, map_location="cpu")
        reward_model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("✓ Reward model loaded")
    else:
        logger.warning(f"No trained model found at {model_path}")
        logger.info("Creating new model (untrained - for demonstration only)")

        reward_model, text_encoder = create_reward_model()

        logger.info("⚠ Using untrained model - scores will be random")

    # Create fine-tuner (for scoring functionality)
    finetuner = RLAIFFineTuner(
        reward_model=reward_model,
        text_encoder=text_encoder,
    )

    # =========================================================================
    # Step 2: Generate Multiple Candidate Decisions
    # =========================================================================
    logger.info("\n[Step 2] Generating Candidate Decisions")
    logger.info("-" * 80)

    symbol = "AAPL"
    num_candidates = 5

    logger.info(f"Generating {num_candidates} candidate decisions for {symbol}")
    logger.info("(In production, this would call Claude with different temperatures/prompts)")

    candidates = []
    import pandas as pd  # Import for timestamp

    for i in range(num_candidates):
        # Vary temperature to get different outputs
        temperature = 0.5 + i * 0.2  # 0.5, 0.7, 0.9, 1.1, 1.3
        decision = generate_mock_decision(symbol, temperature=temperature)
        candidates.append(decision)

        logger.info(
            f"  Candidate {i + 1}: {decision.action.upper()} "
            f"(agent_score: {decision.agent_response.score:.2f}, "
            f"confidence: {decision.agent_response.confidence:.0%})"
        )

    # =========================================================================
    # Step 3: Score Candidates with Reward Model
    # =========================================================================
    logger.info("\n[Step 3] Scoring Candidates with Reward Model")
    logger.info("-" * 80)

    scored_candidates = []

    for i, decision in enumerate(candidates):
        reward_score = finetuner.score_decision(decision)
        scored_candidates.append((decision, reward_score))

        logger.info(
            f"  Candidate {i + 1}: Reward Score = {reward_score:.4f} "
            f"(agent_score: {decision.agent_response.score:.2f})"
        )

    # =========================================================================
    # Step 4: Select Best Candidate
    # =========================================================================
    logger.info("\n[Step 4] Selecting Best Candidate")
    logger.info("-" * 80)

    # Sort by reward score (highest first)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    best_decision, best_score = scored_candidates[0]
    worst_decision, worst_score = scored_candidates[-1]

    logger.info(f"\n✓ Best Decision Selected:")
    logger.info(f"  Action: {best_decision.action.upper()}")
    logger.info(f"  Reward Score: {best_score:.4f}")
    logger.info(f"  Agent Score: {best_decision.agent_response.score:.2f}")
    logger.info(f"  Agent Confidence: {best_decision.agent_response.confidence:.0%}")

    logger.info(f"\nFor comparison, worst candidate:")
    logger.info(f"  Reward Score: {worst_score:.4f}")
    logger.info(f"  Score Difference: {best_score - worst_score:.4f}")

    # =========================================================================
    # Step 5: Analysis
    # =========================================================================
    logger.info("\n[Step 5] Analysis")
    logger.info("-" * 80)

    logger.info("\nBenefits of Reward Model Guidance:")
    logger.info("  ✓ No fine-tuning required")
    logger.info("  ✓ Immediate deployment")
    logger.info("  ✓ Can update reward model anytime")
    logger.info("  ✓ Leverages market outcome feedback")

    logger.info("\nNext Steps in Production:")
    logger.info("  1. Execute best_decision in market")
    logger.info("  2. Track outcome with OutcomeTracker")
    logger.info("  3. Update PreferenceGenerator with result")
    logger.info("  4. Periodically retrain reward model")
    logger.info("  5. Improved reward model → better candidate selection")

    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)

    logger.info("\nKey Insight:")
    logger.info("  Reward model learns from past market outcomes to predict")
    logger.info("  which decisions will perform well. Use it to select best")
    logger.info("  from multiple candidates WITHOUT fine-tuning Claude.")

    logger.info(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
