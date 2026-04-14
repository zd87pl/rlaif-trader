"""AutoTrader: Autonomous strategy experimentation loop.

Applies the autoresearch-mlx pattern (edit code -> run -> measure -> keep/discard -> repeat)
to trading strategies. The LLM generates Python signal functions, backtests them in a sandbox,
and hot-swaps winners into live trading.
"""

from .strategy_spec import StrategySpec
from .metrics import CompositeMetric, ExperimentResult
from .experiment_log import ExperimentLog
from .safety import SafetyGuard
from .experiment_runner import ExperimentRunner
from .sentinel import MarketSentinel, MarketEvent
from .thesis_generator import ThesisGenerator, StrategyThesis
from .strategy_swapper import StrategyHotSwapper, SwapResult
from .orchestrator import ExperimentOrchestrator

# Lazy import to avoid pulling in heavy deps (faiss, torch, etc.)
def _get_rlaif_bridge():
    from .rlaif_bridge import RLAIFBridge
    return RLAIFBridge

__all__ = [
    "StrategySpec",
    "CompositeMetric",
    "ExperimentResult",
    "ExperimentLog",
    "SafetyGuard",
    "ExperimentRunner",
    "MarketSentinel",
    "MarketEvent",
    "ThesisGenerator",
    "StrategyThesis",
    "StrategyHotSwapper",
    "SwapResult",
    "ExperimentOrchestrator",
]
