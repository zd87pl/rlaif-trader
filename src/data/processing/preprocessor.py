"""Data preprocessing and cleaning"""

from typing import Optional

import numpy as np
import pandas as pd

from ...utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Data preprocessing for financial time series

    Features:
    - Fill missing values (forward fill for prices)
    - Remove outliers
    - Handle market microstructure effects
    - Ensure temporal ordering (critical for avoiding lookahead bias)
    - Generate basic returns and volatility features
    """

    def __init__(
        self,
        fill_method: str = "forward",
        outlier_threshold: float = 5.0,
        min_trading_days: int = 252,
    ):
        """
        Initialize preprocessor

        Args:
            fill_method: Method for filling missing values ("forward", "interpolate")
            outlier_threshold: Standard deviations for outlier detection
            min_trading_days: Minimum number of trading days required
        """
        self.fill_method = fill_method
        self.outlier_threshold = outlier_threshold
        self.min_trading_days = min_trading_days

    def preprocess(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Preprocess raw OHLCV data

        Args:
            df: DataFrame with OHLCV columns, indexed by timestamp
            symbol: Symbol name (if multi-symbol DataFrame, will be extracted)

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing data: {len(df)} rows")

        # If multi-index with symbols, extract single symbol
        if isinstance(df.index, pd.MultiIndex) and symbol:
            df = df.xs(symbol, level="symbol")
        elif isinstance(df.index, pd.MultiIndex):
            # Get first symbol
            symbol = df.index.get_level_values("symbol")[0]
            df = df.xs(symbol, level="symbol")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by timestamp (critical!)
        df = df.sort_index()

        # Check minimum data requirement
        if len(df) < self.min_trading_days:
            logger.warning(
                f"Insufficient data: {len(df)} < {self.min_trading_days} days"
            )

        # Fill missing timestamps (create complete time series)
        df = self._fill_missing_timestamps(df)

        # Fill missing values
        df = self._fill_missing_values(df)

        # Remove outliers
        df = self._remove_outliers(df)

        # Add basic features
        df = self._add_basic_features(df)

        # Drop any remaining NaN rows
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")

        logger.info(f"Preprocessing complete: {len(df)} rows remaining")

        return df

    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing timestamps in time series"""
        # Infer frequency
        freq = pd.infer_freq(df.index[:100])  # Use first 100 rows to infer

        if freq is None:
            logger.warning("Could not infer frequency, using '1min'")
            freq = "1min"

        # Create complete date range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
        )

        # Reindex to include all timestamps
        df = df.reindex(full_range)

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values appropriately"""
        if self.fill_method == "forward":
            # Forward fill prices (use last known price)
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()

            # Volume: fill with 0 (no trading)
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)

        elif self.fill_method == "interpolate":
            # Interpolate prices
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear")

            # Volume: fill with 0
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or cap outliers based on returns"""
        if "close" not in df.columns:
            return df

        # Calculate returns
        returns = df["close"].pct_change()

        # Calculate z-scores
        mean_ret = returns.mean()
        std_ret = returns.std()
        z_scores = (returns - mean_ret) / std_ret

        # Identify outliers
        outliers = np.abs(z_scores) > self.outlier_threshold

        n_outliers = outliers.sum()
        if n_outliers > 0:
            logger.info(
                f"Detected {n_outliers} outliers "
                f"({n_outliers/len(df)*100:.2f}% of data)"
            )

            # Cap outliers instead of removing (to preserve time series)
            # Cap at +/- threshold standard deviations
            cap_value = self.outlier_threshold * std_ret

            returns_capped = returns.clip(
                lower=mean_ret - cap_value,
                upper=mean_ret + cap_value,
            )

            # Reconstruct prices from capped returns
            df["close"] = df["close"].iloc[0] * (1 + returns_capped).cumprod()

            # Adjust OHLC proportionally
            if all(col in df.columns for col in ["open", "high", "low"]):
                ratio = df["close"] / df["close"].shift(1)
                df["open"] = df["open"] * ratio
                df["high"] = df["high"] * ratio
                df["low"] = df["low"] * ratio

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""
        if "close" not in df.columns:
            return df

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Rolling volatility (20-period)
        df["volatility_20"] = df["returns"].rolling(window=20).std()

        # Volume features (if available)
        if "volume" in df.columns:
            df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)

        return df

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (temporal split, no shuffle!)

        Args:
            df: DataFrame to split
            test_size: Fraction for test set (default 0.2 = 20%)

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(
            f"Split data: train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
            f"test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)"
        )

        return train_df, test_df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="1D")
    df = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 102,
            "low": np.random.randn(len(dates)).cumsum() + 98,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Preprocess
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess(df)

    print(processed_df.head())
    print(f"\nColumns: {processed_df.columns.tolist()}")
    print(f"\nShape: {processed_df.shape}")
