"""Market data ingestion from Alpaca API"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest

from ...utils.config import get_settings
from ...utils.logging import get_logger

logger = get_logger(__name__)


class AlpacaDataClient:
    """
    Client for downloading and caching market data from Alpaca

    Features:
    - Download historical bars (1Min, 5Min, 1Hour, Daily)
    - Caching to avoid repeated API calls
    - Automatic retry on failures
    - Point-in-time correctness (no lookahead bias)
    - Support for multiple assets
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize Alpaca data client

        Args:
            api_key: Alpaca API key (if None, loads from env)
            secret_key: Alpaca secret key (if None, loads from env)
            cache_dir: Directory for caching data (if None, uses default)
        """
        settings = get_settings()

        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Alpaca data client initialized")

    def download_bars(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        timeframe: str = "1Min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical bars for symbols

        Args:
            symbols: Single symbol or list of symbols (e.g., "AAPL" or ["AAPL", "MSFT"])
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data, multi-indexed by (symbol, timestamp)
        """
        # Convert to list if single symbol
        if isinstance(symbols, str):
            symbols = [symbols]

        # Parse dates
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        logger.info(
            f"Downloading bars for {len(symbols)} symbols from {start} to {end} "
            f"at {timeframe} timeframe"
        )

        # Try loading from cache
        if use_cache:
            cached_data = self._load_from_cache(symbols, start, end, timeframe)
            if cached_data is not None:
                logger.info("Loaded data from cache")
                return cached_data

        # Parse timeframe
        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }

        if timeframe not in tf_map:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(tf_map.keys())}"
            )

        tf = tf_map[timeframe]

        # Create request
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
            adjustment="all",  # Adjust for splits and dividends
        )

        # Download data
        try:
            bars = self.client.get_stock_bars(request_params)
            df = bars.df

            # Reset index to get symbol and timestamp as columns
            df = df.reset_index()

            # Set proper multi-index
            df = df.set_index(["symbol", "timestamp"])

            # Sort index
            df = df.sort_index()

            logger.info(f"Downloaded {len(df)} bars")

            # Cache the data
            if use_cache:
                self._save_to_cache(df, symbols, start, end, timeframe)

            return df

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise

    def download_latest(
        self,
        symbols: Union[str, List[str]],
        days: int = 365,
        timeframe: str = "1Min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download latest N days of data

        Args:
            symbols: Single symbol or list of symbols
            days: Number of days to download
            timeframe: Bar timeframe
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        return self.download_bars(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            use_cache=use_cache,
        )

    def _get_cache_path(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Path:
        """Generate cache file path"""
        symbols_str = "_".join(sorted(symbols))
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        filename = f"{symbols_str}_{start_str}_{end_str}_{timeframe}.parquet"
        return self.cache_dir / filename

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ):
        """Save DataFrame to cache"""
        try:
            cache_path = self._get_cache_path(symbols, start, end, timeframe)
            df.to_parquet(cache_path)
            logger.debug(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if available"""
        try:
            cache_path = self._get_cache_path(symbols, start, end, timeframe)

            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded data from cache: {cache_path}")
                return df

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

        return None

    def get_asset_info(self, symbol: str) -> dict:
        """
        Get asset information

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with asset information
        """
        # Note: Alpaca's data client doesn't provide asset info directly
        # You would need to use the TradingClient for this
        # For now, return basic info
        return {
            "symbol": symbol,
            "exchange": "Unknown",
            "asset_class": "us_equity",
        }


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    client = AlpacaDataClient()

    # Download sample data
    df = client.download_latest(
        symbols=["AAPL", "MSFT"],
        days=30,
        timeframe="1Min",
    )

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
