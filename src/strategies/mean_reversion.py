"""
Mean reversion strategy — buy the dip, sell the rip.

Profits when price reverts to its mean after overextension.
Works best in range-bound markets.
Opposite of momentum — they're designed to be uncorrelated.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import Strategy, Signal


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands, RSI extremes, and z-scores.

    Signals:
    - BUY: Price below lower Bollinger Band, RSI < 30, high z-score deviation
    - SELL: Price above upper Bollinger Band, RSI > 70
    - Requires volatility to be within normal range (not during crashes)
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        zscore_threshold: float = 2.0,
        max_volatility_percentile: float = 0.85,
        weight: float = 1.0,
    ):
        super().__init__(name="mean_reversion", weight=weight)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.zscore_threshold = zscore_threshold
        self.max_volatility_percentile = max_volatility_percentile

    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        news_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        latest = features.iloc[-1]
        score = 0.0
        reasons = []

        price = float(latest.get("close", 0))

        # Safety check: don't mean-revert in high volatility (crash conditions)
        returns = features["close"].pct_change().dropna()
        if len(returns) > 20:
            current_vol = float(returns.iloc[-20:].std())
            vol_percentile = float((returns.rolling(20).std() < current_vol).mean())
            if vol_percentile > self.max_volatility_percentile:
                return Signal(
                    symbol=symbol,
                    action="hold",
                    score=0.0,
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning="Volatility too high for mean reversion — sitting out",
                )

        # 1. Bollinger Band position
        bb_upper = float(latest.get("bb_upper", price * 1.02))
        bb_lower = float(latest.get("bb_lower", price * 0.98))
        bb_middle = float(latest.get("bb_middle", price))

        if bb_upper != bb_lower:
            bb_position = (price - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5

        if bb_position < 0:  # Below lower band
            score += 0.4
            reasons.append(f"Price below lower Bollinger Band (position={bb_position:.2f})")
        elif bb_position > 1:  # Above upper band
            score -= 0.4
            reasons.append(f"Price above upper Bollinger Band (position={bb_position:.2f})")
        elif bb_position < 0.2:
            score += 0.2
            reasons.append("Near lower Bollinger Band")
        elif bb_position > 0.8:
            score -= 0.2
            reasons.append("Near upper Bollinger Band")

        # 2. RSI extremes (contrarian)
        rsi = float(latest.get("rsi", 50))
        if rsi < self.rsi_oversold:
            score += 0.3
            reasons.append(f"RSI oversold ({rsi:.0f} < {self.rsi_oversold})")
        elif rsi > self.rsi_overbought:
            score -= 0.3
            reasons.append(f"RSI overbought ({rsi:.0f} > {self.rsi_overbought})")

        # 3. Z-score of price from moving average
        sma_20 = float(latest.get("sma_20", price))
        if len(features) >= 20:
            std_20 = float(features["close"].iloc[-20:].std())
            if std_20 > 0:
                zscore = (price - sma_20) / std_20
                if zscore < -self.zscore_threshold:
                    score += 0.3
                    reasons.append(f"Z-score very negative ({zscore:.2f}) — oversold")
                elif zscore > self.zscore_threshold:
                    score -= 0.3
                    reasons.append(f"Z-score very positive ({zscore:.2f}) — overbought")

        # 4. Stochastic oversold/overbought
        stoch_k = float(latest.get("stoch_k", 50))
        if stoch_k < 20:
            score += 0.15
            reasons.append(f"Stochastic oversold ({stoch_k:.0f})")
        elif stoch_k > 80:
            score -= 0.15
            reasons.append(f"Stochastic overbought ({stoch_k:.0f})")

        # Normalize
        score = max(-1.0, min(1.0, score))
        confidence = min(abs(score) + 0.15, 1.0)

        if score >= 0.3:
            action = "buy"
        elif score <= -0.3:
            action = "sell"
        else:
            action = "hold"

        return Signal(
            symbol=symbol,
            action=action,
            score=score,
            confidence=confidence,
            strategy_name=self.name,
            reasoning="; ".join(reasons),
        )
