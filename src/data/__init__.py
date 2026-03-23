"""Data ingestion and processing modules"""

from .ingestion.market_data import AlpacaDataClient
from .ingestion.news_client import NewsAggregator
from .ingestion.fundamental_client import FundamentalDataAggregator
from .processing.preprocessor import DataPreprocessor

__all__ = [
    "AlpacaDataClient",
    "NewsAggregator",
    "FundamentalDataAggregator",
    "DataPreprocessor",
]
