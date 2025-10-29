"""
Technical Analyst Agent

Analyzes:
- Chart patterns
- Technical indicators (RSI, MACD, etc.)
- Support and resistance levels
- Trend identification
- Volume patterns
"""

from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResponse, ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalyst(BaseAgent):
    """
    Specialized agent for technical analysis

    Expertise:
    - Chart pattern recognition
    - Indicator interpretation (RSI, MACD, Bollinger, etc.)
    - Trend identification and strength
    - Support/resistance level identification
    - Volume analysis and confirmation
    """

    SYSTEM_PROMPT = """You are an expert technical analyst specializing in price action and indicators.

Your role:
- Identify chart patterns (head & shoulders, triangles, flags, etc.)
- Interpret technical indicators with context
- Determine trend strength and potential reversals
- Identify key support and resistance levels
- Analyze volume for confirmation or divergence

Approach:
- Use multiple timeframes for context
- Look for indicator confluence (multiple signals agreeing)
- Identify divergences (price vs. indicators)
- Consider volume as confirmation
- Assess risk/reward setups

Output format:
1. Trend Analysis (direction and strength)
2. Key Technical Indicators (RSI, MACD, Bollinger, ATR)
3. Support & Resistance Levels
4. Chart Patterns & Signals
5. Recommendation with Score (-1 to 1) and Confidence (0 to 1)
"""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        super().__init__(
            name="TechnicalAnalyst",
            system_prompt=self.SYSTEM_PROMPT,
            claude_client=claude_client,
            use_rag=False,  # Technical analysis typically doesn't need RAG
        )

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform technical analysis

        Args:
            symbol: Stock symbol
            data: Dictionary containing:
                - current_price: Current price
                - indicators: Dict of technical indicators (RSI, MACD, etc.)
                - price_history: Recent price data
                - volume_data: Recent volume data
            context: Additional context (optional for technical analysis)

        Returns:
            AgentResponse with technical analysis
        """
        logger.info(f"TechnicalAnalyst analyzing {symbol}")

        steps = [
            "Identify Trend: Determine overall trend (uptrend, downtrend, sideways) using moving averages "
            "and price action. Assess trend strength.",
            "Analyze Key Indicators: Interpret RSI (overbought/oversold), MACD (momentum), "
            "Bollinger Bands (volatility), ATR (range). Look for confluence.",
            "Determine Support & Resistance: Identify key price levels where buying/selling pressure may emerge. "
            "Note if price is near these levels.",
            "Assess Chart Patterns: Look for recognizable patterns (breakouts, reversals, continuation). "
            "Evaluate volume confirmation.",
            "Provide Technical Score: Synthesize into score (-1=strong bearish, 0=neutral, 1=strong bullish). "
            "Confidence based on indicator agreement.",
        ]

        prompt = self._build_cot_prompt(symbol, data, steps, context)
        prompt += "\n\nIMPORTANT: End with:\nScore: [number from -1 to 1]\nConfidence: [percentage from 0% to 100%]\n"

        response = self.claude.complete(
            prompt=prompt,
            system=self.system_prompt,
            max_tokens=2000,
            temperature=0.7,
        )

        parsed = self._parse_response(response, extract_score=True, extract_confidence=True)

        return AgentResponse(
            analysis=parsed["analysis"],
            score=parsed["score"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            data=data,
            agent_name=self.name,
        )
