"""
Strategy base class — every strategy returns a Signal.

Signal is the universal currency of this system.
Any strategy that produces a Signal can be composed, ensembled, or backtested.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Signal:
    """
    Universal signal output from any strategy.
    Every strategy must produce this — it's what the risk manager and executor consume.
    """
    symbol: str
    action: str  # "buy", "sell", "hold"
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class Strategy(ABC):
    """
    Base strategy class. Every trading strategy inherits from this.

    Contract:
    - generate_signal() takes a symbol + data, returns a Signal
    - Each strategy has a name and configurable parameters
    - Strategies are stateless per call (no hidden position tracking)
    """

    def __init__(self, name: str, weight: float = 1.0):
        """
        Args:
            name: Strategy identifier
            weight: Default weight in ensemble (0.0 to 1.0)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        news_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        """
        Generate a trading signal.

        Args:
            symbol: Stock symbol
            price_data: OHLCV DataFrame
            features: DataFrame with technical indicators computed
            news_data: News/sentiment data dict
            fundamental_data: Fundamental data dict

        Returns:
            Signal with action, score, confidence
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"
