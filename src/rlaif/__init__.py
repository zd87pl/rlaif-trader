"""
RLAIF (Reinforcement Learning from AI Feedback) Module

This module implements the core RLAIF feedback loop:
1. PreferenceGenerator: Creates preference pairs from market outcomes
2. RewardModel: Learns to predict good vs bad trading decisions
3. RLAIFFineTuner: Fine-tunes Claude agents using PPO/DPO
4. OutcomeTracker: Monitors live trading results

Key Innovation:
Uses actual market outcomes (profit/loss, risk-adjusted returns) as ground truth
to create preference pairs, then fine-tunes the multi-agent system to make
better decisions over time.

Research shows 20-40% improvement with RLAIF over static LLM approaches.
"""

from .preference_generator import PreferenceGenerator
from .reward_model import RewardModel
from .rlaif_finetuner import RLAIFFineTuner
from .outcome_tracker import OutcomeTracker

__all__ = [
    "PreferenceGenerator",
    "RewardModel",
    "RLAIFFineTuner",
    "OutcomeTracker",
]
