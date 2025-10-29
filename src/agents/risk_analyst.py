"""
Risk Analyst Agent

Analyzes:
- Volatility and downside risk
- Portfolio correlations
- Position sizing
- Tail risk and black swans
- Risk-adjusted returns
"""

from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResponse, ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RiskAnalyst(BaseAgent):
    """
    Specialized agent for risk assessment

    Expertise:
    - Volatility analysis (historical, implied)
    - Downside risk metrics (max drawdown, VaR, CVaR)
    - Correlation and diversification
    - Position sizing recommendations
    - Scenario analysis and stress testing
    """

    SYSTEM_PROMPT = """You are an expert risk analyst specializing in portfolio risk management.

Your role:
- Assess volatility and downside risk
- Calculate risk-adjusted return metrics (Sharpe, Sortino)
- Analyze correlations and diversification
- Recommend appropriate position sizes
- Identify tail risks and potential scenarios

Approach:
- Focus on risk-adjusted returns, not just returns
- Consider both historical and forward-looking risks
- Assess asymmetric risks (downside vs. upside)
- Evaluate correlations with existing portfolio
- Provide conservative position sizing

Output format:
1. Volatility Assessment (historical and implied)
2. Downside Risk Metrics (max drawdown, VaR, CVaR)
3. Correlation & Diversification Analysis
4. Position Sizing Recommendation
5. Risk Score (-1 to 1) where -1=very risky, 1=very safe, and Confidence (0 to 1)
"""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        super().__init__(
            name="RiskAnalyst",
            system_prompt=self.SYSTEM_PROMPT,
            claude_client=claude_client,
            use_rag=False,
        )

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform risk analysis

        Args:
            symbol: Stock symbol
            data: Dictionary containing:
                - volatility: Historical volatility
                - max_drawdown: Maximum drawdown
                - sharpe_ratio: Sharpe ratio
                - beta: Market beta
                - var_95: Value at Risk (95%)
                - correlations: Correlations with other assets
            context: Additional context (optional)

        Returns:
            AgentResponse with risk analysis
        """
        logger.info(f"RiskAnalyst analyzing {symbol}")

        steps = [
            "Assess Volatility: Analyze historical volatility and recent changes. "
            "Determine if volatility is elevated, normal, or low relative to history and peers.",
            "Evaluate Downside Risk: Review max drawdown, VaR, CVaR. "
            "Calculate potential loss scenarios (5th percentile, worst case).",
            "Analyze Risk-Adjusted Returns: Calculate Sharpe and Sortino ratios. "
            "Determine if returns justify the risk taken.",
            "Consider Diversification: Assess correlations with market and other holdings. "
            "Evaluate diversification benefits or concentration risks.",
            "Recommend Position Size: Based on risk metrics, suggest appropriate position size "
            "(% of portfolio). Consider risk tolerance and overall portfolio exposure.",
        ]

        prompt = self._build_cot_prompt(symbol, data, steps, context)
        prompt += "\n\nIMPORTANT: End with:\nRisk Score: [number from -1 to 1 where -1=very risky, 1=very safe]\n"
        prompt += "Recommended Position Size: [percentage of portfolio]\nConfidence: [percentage from 0% to 100%]\n"

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
