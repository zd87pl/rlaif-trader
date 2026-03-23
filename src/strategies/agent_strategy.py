"""
Agent strategy — wraps the multi-agent Claude system as a Strategy.

This is the most expensive strategy (API calls), but the most adaptive.
Use alongside cheaper strategies for signal confirmation.
"""

from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

from .base import Strategy, Signal
from ..agents import ManagerAgent, ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentStrategy(Strategy):
    """
    Wraps the multi-agent Claude system as a composable Strategy.

    Cost: ~$0.02-0.10 per signal (depending on model choice).
    Latency: 10-30 seconds per signal.
    Strength: Adapts to novel situations, reads sentiment/fundamentals.
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        debate_rounds: int = 2,
        weight: float = 1.5,  # Higher default weight — it's the most informed
    ):
        super().__init__(name="agent", weight=weight)
        load_dotenv()
        self.claude_client = claude_client or ClaudeClient()
        self.manager = ManagerAgent(
            claude_client=self.claude_client, debate_rounds=debate_rounds
        )

    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        news_data: Optional[Dict[str, Any]] = None,
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        latest = features.iloc[-1]

        def safe_float(val, default=0.0):
            try:
                v = float(val)
                return v if v == v else default
            except (ValueError, TypeError):
                return default

        # Build agent data in the format ManagerAgent expects
        price = safe_float(latest.get("close"))
        sma_50 = safe_float(latest.get("sma_50", price))

        technical = {
            "current_price": price,
            "rsi": safe_float(latest.get("rsi", 50)),
            "macd": safe_float(latest.get("macd", 0)),
            "macd_signal": safe_float(latest.get("macd_signal", 0)),
            "macd_hist": safe_float(latest.get("macd_hist", 0)),
            "trend": "uptrend" if price > sma_50 else "downtrend",
        }

        for period in [20, 50, 200]:
            key = f"sma_{period}"
            if key in latest.index:
                technical[key] = safe_float(latest[key])

        returns = features["close"].pct_change().dropna()
        risk = {
            "volatility": safe_float(returns.std() * (252 ** 0.5)),
            "max_drawdown": 0.15,
            "sharpe_ratio": safe_float(
                returns.mean() / (returns.std() + 1e-9) * (252 ** 0.5)
            ),
            "beta": fundamental_data.get("market_data", {}).get("beta", 1.0) if fundamental_data else 1.0,
        }

        sentiment = news_data or {
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "analyst_ratings": "Not available",
        }

        fundamentals = {}
        if fundamental_data:
            fundamentals = fundamental_data.get("fundamentals", {})
            analyst_est = fundamental_data.get("analyst_estimates", {})
            if analyst_est:
                fundamentals["target_price_mean"] = analyst_est.get("target_mean", 0)
                fundamentals["analyst_recommendation"] = analyst_est.get("recommendation", "none")

        if not fundamentals:
            fundamentals = {"symbol": symbol}

        agent_data = {
            "technical": technical,
            "risk": risk,
            "sentiment": sentiment,
            "fundamentals": fundamentals,
        }

        try:
            response = self.manager.analyze(symbol=symbol, data=agent_data)

            if response.score >= 0.3:
                action = "buy"
            elif response.score <= -0.3:
                action = "sell"
            else:
                action = "hold"

            return Signal(
                symbol=symbol,
                action=action,
                score=response.score,
                confidence=response.confidence,
                strategy_name=self.name,
                reasoning=response.analysis[:300],
                metadata={
                    "specialist_scores": response.data.get("specialist_scores", {}),
                },
            )

        except Exception as e:
            logger.error(f"Agent strategy failed for {symbol}: {e}")
            return Signal(
                symbol=symbol,
                action="hold",
                score=0.0,
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Agent error: {e}",
            )
