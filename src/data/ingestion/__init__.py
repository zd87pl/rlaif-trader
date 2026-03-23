"""Data ingestion modules"""

from .market_data import AlpacaDataClient
from .news_client import (
    FinnhubNewsClient,
    PolygonNewsClient,
    AlpacaNewsClient,
    NewsAggregator,
)
from .fundamental_client import (
    YFinanceFundamentals,
    SECEdgarClient,
    FundamentalDataAggregator,
)

__all__ = [
    "AlpacaDataClient",
    "FinnhubNewsClient",
    "PolygonNewsClient",
    "AlpacaNewsClient",
    "NewsAggregator",
    "YFinanceFundamentals",
    "SECEdgarClient",
    "FundamentalDataAggregator",
]
