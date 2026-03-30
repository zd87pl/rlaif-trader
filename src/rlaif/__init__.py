"""
RLAIF (Reinforcement Learning from AI Feedback) Module

Core loop: Market outcomes -> Preference pairs -> Reward model -> Fine-tuned agents
v2.1: Added process-level reasoning verification, composite rewards, options outcomes
"""

from .preference_generator import PreferenceGenerator
from .reward_model import RewardModel
from .rlaif_finetuner import RLAIFFineTuner

try:
    from .outcome_tracker import OutcomeTracker
except ImportError:
    OutcomeTracker = None

try:
    from .options_outcome_tracker import OptionsOutcomeTracker
except ImportError:
    OptionsOutcomeTracker = None

from .reasoning_verifier import ProcessRewardModel, DynamicSemanticReward, ReasoningChain
from .composite_reward import CompositeRewardFunction

__all__ = [
    "PreferenceGenerator", "RewardModel", "RLAIFFineTuner",
    "OutcomeTracker", "OptionsOutcomeTracker",
    "ProcessRewardModel", "DynamicSemanticReward", "ReasoningChain",
    "CompositeRewardFunction",
]
