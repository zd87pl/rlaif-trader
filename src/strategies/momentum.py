"""
Momentum strategy — trend following.

Buys when price is above key moving averages with strong momentum.
Sells when momentum breaks down.

This is the workhorse strategy for trending markets.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import Strategy, Signal


class MomentumStrategy(Strategy):
    """
    Trend-following momentum strategy.

    Signals:
    - BUY: Price > SMA50 > SMA200, RSI 40-70, MACD positive, strong volume
    - SELL: Price < SMA50, RSI > 75 or < 25, MACD crossover down
    - HOLD: Mixed signals or low conviction

    Parameters tunable via config.
    """

    def __init__(
        self,
        fast_sma: int = 50,
        slow_sma: int = 200,
        rsi_buy_range: tuple = (40, 70),
        rsi_overbought: float = 75,
        rsi_oversold: float = 25,
        volume_multiplier: float = 1.2,
        weight: float = 1.0,
    ):
        super().__init__(name="momentum", weight=weight)
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.rsi_buy_range = rsi_buy_range
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_multiplier = volume_multiplier

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
        signals_counted = 0

        price = float(latest.get("close", 0))

        # 1. Trend alignment (SMA crossover)
        sma_fast = float(latest.get(f"sma_{self.fast_sma}", price))
        sma_slow = float(latest.get(f"sma_{self.slow_sma}", price))

        if price > sma_fast > sma_slow:
            score += 0.4
            reasons.append(f"Price > SMA{self.fast_sma} > SMA{self.slow_sma} (bullish trend)")
        elif price < sma_fast < sma_slow:
            score -= 0.4
            reasons.append(f"Price < SMA{self.fast_sma} < SMA{self.slow_sma} (bearish trend)")
        signals_counted += 1

        # 2. RSI momentum
        rsi = float(latest.get("rsi", 50))
        if self.rsi_buy_range[0] <= rsi <= self.rsi_buy_range[1]:
            score += 0.2
            reasons.append(f"RSI {rsi:.0f} in buy zone ({self.rsi_buy_range[0]}-{self.rsi_buy_range[1]})")
        elif rsi > self.rsi_overbought:
            score -= 0.3
            reasons.append(f"RSI {rsi:.0f} overbought (>{self.rsi_overbought})")
        elif rsi < self.rsi_oversold:
            score += 0.3  # Oversold can mean bounce in trend context
            reasons.append(f"RSI {rsi:.0f} oversold (<{self.rsi_oversold})")
        signals_counted += 1

        # 3. MACD momentum
        macd = float(latest.get("macd", 0))
        macd_signal = float(latest.get("macd_signal", 0))
        macd_hist = float(latest.get("macd_hist", 0))

        if macd > macd_signal and macd_hist > 0:
            score += 0.25
            reasons.append("MACD bullish crossover")
        elif macd < macd_signal and macd_hist < 0:
            score -= 0.25
            reasons.append("MACD bearish crossover")
        signals_counted += 1

        # 4. Rate of change (price momentum)
        if "roc_12" in latest.index:
            roc = float(latest["roc_12"])
            if roc > 5:
                score += 0.15
                reasons.append(f"12-period ROC strong ({roc:.1f}%)")
            elif roc < -5:
                score -= 0.15
                reasons.append(f"12-period ROC weak ({roc:.1f}%)")
            signals_counted += 1

        # 5. Volume confirmation
        if "volume" in latest.index and "volume" in features.columns:
            vol = float(latest["volume"])
            avg_vol = float(features["volume"].rolling(20).mean().iloc[-1])
            if avg_vol > 0 and vol > avg_vol * self.volume_multiplier:
                # Volume confirms the move
                score *= 1.2
                reasons.append(f"Volume confirms ({vol / avg_vol:.1f}x avg)")

        # Normalize score to [-1, 1]
        score = max(-1.0, min(1.0, score))

        # Confidence: higher when signals agree
        confidence = min(abs(score) + 0.2, 1.0)

        # Action
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
