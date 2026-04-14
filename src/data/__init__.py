"""Data ingestion and processing modules"""

from .ingestion.market_data import AlpacaDataClient
from .processing.preprocessor import DataPreprocessor

try:
    from .ingestion.ccxt_client import CCXTDataClient
except ImportError:
    CCXTDataClient = None  # ccxt not installed

__all__ = ["AlpacaDataClient", "CCXTDataClient", "DataPreprocessor"]
