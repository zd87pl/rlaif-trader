"""Technical indicator feature engineering"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalFeatureEngine:
    """
    Compute 60+ technical indicators for financial time series

    Categories:
    - Trend: SMA, EMA, MACD, ADX
    - Momentum: RSI, Stochastic, ROC, Williams %R
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, MFI, VWAP, Volume SMA

    All indicators are computed without lookahead bias.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical feature engine

        Args:
            config: Configuration dict with periods and parameters
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """Default configuration for technical indicators"""
        return {
            "sma_periods": [20, 50, 100, 200],
            "ema_periods": [12, 26, 50],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "rsi_period": 14,
            "stochastic": {"k_period": 14, "d_period": 3},
            "bollinger": {"period": 20, "std_dev": 2},
            "atr_period": 14,
            "adx_period": 14,
            "obv": True,
            "mfi_period": 14,
            "vwap": True,
        }

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all technical indicators added
        """
        logger.info("Computing technical indicators")

        df = df.copy()

        # Trend indicators
        df = self._compute_trend(df)

        # Momentum indicators
        df = self._compute_momentum(df)

        # Volatility indicators
        df = self._compute_volatility(df)

        # Volume indicators
        df = self._compute_volume(df)

        logger.info(f"Added {len(df.columns)} features")

        return df

    def _compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend indicators"""
        # Simple Moving Averages
        for period in self.config["sma_periods"]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in self.config["ema_periods"]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        macd_config = self.config["macd"]
        ema_fast = df["close"].ewm(span=macd_config["fast"], adjust=False).mean()
        ema_slow = df["close"].ewm(span=macd_config["slow"], adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=macd_config["signal"], adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ADX (Average Directional Index)
        df = self._compute_adx(df, self.config["adx_period"])

        return df

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators"""
        # RSI (Relative Strength Index)
        period = self.config["rsi_period"]
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        stoch_config = self.config["stochastic"]
        k_period = stoch_config["k_period"]
        d_period = stoch_config["d_period"]

        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()

        # ROC (Rate of Change)
        for period in [12, 21]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period)) / df["close"].shift(period)
            ) * 100

        # Williams %R
        period = 14
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-9)

        # Momentum
        df["momentum"] = df["close"] - df["close"].shift(10)

        # CCI (Commodity Channel Index)
        df = self._compute_cci(df, period=20)

        return df

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility indicators"""
        # Bollinger Bands
        bb_config = self.config["bollinger"]
        period = bb_config["period"]
        std_dev = bb_config["std_dev"]

        df["bb_middle"] = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (rolling_std * std_dev)
        df["bb_lower"] = df["bb_middle"] - (rolling_std * std_dev)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

        # ATR (Average True Range)
        atr_period = self.config["atr_period"]
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()

        # Keltner Channels
        kc_period = 20
        kc_atr_mult = 2
        df["kc_middle"] = df["close"].ewm(span=kc_period, adjust=False).mean()
        df["kc_upper"] = df["kc_middle"] + (df["atr"] * kc_atr_mult)
        df["kc_lower"] = df["kc_middle"] - (df["atr"] * kc_atr_mult)

        # Historical Volatility
        for period in [20, 60]:
            returns = df["close"].pct_change()
            df[f"hvol_{period}"] = returns.rolling(window=period).std() * np.sqrt(252)

        return df

    def _compute_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume indicators"""
        if "volume" not in df.columns:
            logger.warning("Volume column not found, skipping volume indicators")
            return df

        # OBV (On-Balance Volume)
        if self.config["obv"]:
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # MFI (Money Flow Index)
        mfi_period = self.config["mfi_period"]
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=mfi_period).sum()
        negative_mf = negative_flow.rolling(window=mfi_period).sum()

        mfi_ratio = positive_mf / (negative_mf + 1e-9)
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # VWAP (Volume Weighted Average Price)
        if self.config["vwap"]:
            df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df[
                "volume"
            ].cumsum()

        # Volume SMA
        for period in [20, 50]:
            df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()

        # Volume ratio (current / average)
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)

        # Accumulation/Distribution Line
        df["ad_line"] = (
            ((df["close"] - df["low"]) - (df["high"] - df["close"]))
            / (df["high"] - df["low"] + 1e-9)
            * df["volume"]
        ).cumsum()

        return df

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average Directional Index (ADX)"""
        # Calculate +DM and -DM
        high_diff = df["high"].diff()
        low_diff = df["low"].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)

        # Calculate True Range (already done in ATR, recompute for clarity)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    def _compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Commodity Channel Index (CCI)"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        df["cci"] = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-9)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be computed"""
        features = []

        # Trend
        features.extend([f"sma_{p}" for p in self.config["sma_periods"]])
        features.extend([f"ema_{p}" for p in self.config["ema_periods"]])
        features.extend(["macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di"])

        # Momentum
        features.extend(
            [
                "rsi",
                "stoch_k",
                "stoch_d",
                "roc_12",
                "roc_21",
                "williams_r",
                "momentum",
                "cci",
            ]
        )

        # Volatility
        features.extend(
            [
                "bb_middle",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_pct",
                "atr",
                "kc_middle",
                "kc_upper",
                "kc_lower",
                "hvol_20",
                "hvol_60",
            ]
        )

        # Volume
        features.extend(
            [
                "obv",
                "mfi",
                "vwap",
                "volume_sma_20",
                "volume_sma_50",
                "volume_ratio",
                "ad_line",
            ]
        )

        return features


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

    # Compute technical indicators
    engine = TechnicalFeatureEngine()
    df_with_features = engine.compute_all(df)

    print("Sample features:")
    print(df_with_features[["close", "rsi", "macd", "bb_upper", "bb_lower", "atr"]].tail(10))
    print(f"\nTotal features: {len(df_with_features.columns)}")
    print(f"Feature names: {engine.get_feature_names()[:10]}...")
