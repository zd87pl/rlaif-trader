"""Model modules — foundation models, conformal prediction, regime detection, world models"""

from .foundation.timesfm_wrapper import TimesFMPredictor
from .foundation.ttm_wrapper import TTMPredictor

# New v2.1 models
try:
    from .foundation.kronos_wrapper import KronosPredictor
except ImportError:
    KronosPredictor = None

try:
    from .foundation.mamba_wrapper import MambaPredictor, EnsemblePredictor
except ImportError:
    MambaPredictor = None
    EnsemblePredictor = None

try:
    from .mlx_manager import MLXModelManager
except ImportError:
    MLXModelManager = None

from .conformal import ConformalPredictor, TemporalConformalPredictor, SignalConformalWrapper, PortfolioConformalPredictor
from .regime_detector import RegimeDetector, AdaptiveStrategySelector, MarketRegime
from .world_model import MarketWorldModel, SyntheticDataGenerator

__all__ = [
    "TimesFMPredictor", "TTMPredictor",
    "KronosPredictor", "MambaPredictor", "EnsemblePredictor",
    "MLXModelManager",
    "ConformalPredictor", "TemporalConformalPredictor", "SignalConformalWrapper", "PortfolioConformalPredictor",
    "RegimeDetector", "AdaptiveStrategySelector", "MarketRegime",
    "MarketWorldModel", "SyntheticDataGenerator",
]
