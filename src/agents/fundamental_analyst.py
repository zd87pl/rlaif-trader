"""
Fundamental Analyst Agent

Analyzes:
- Financial statements (10-K, 10-Q)
- Earnings reports
- Revenue and profit growth
- Balance sheet health
- Management quality
- Competitive positioning
"""

from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResponse, ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FundamentalAnalyst(BaseAgent):
    """
    Specialized agent for fundamental analysis

    Expertise:
    - Financial statement analysis
    - Valuation metrics (P/E, P/B, P/S, EV/EBITDA)
    - Profitability ratios (ROE, ROA, margins)
    - Growth rates (revenue, earnings, EPS)
    - Balance sheet strength (debt, liquidity)
    """

    SYSTEM_PROMPT = """You are an expert fundamental analyst specializing in equity research.

Your role:
- Analyze financial statements with precision
- Evaluate business fundamentals and competitive positioning
- Assess management quality and corporate governance
- Calculate and interpret financial ratios
- Identify value creation and risks

Approach:
- Be quantitative: use specific numbers and ratios
- Compare to industry peers and historical performance
- Consider both short-term results and long-term trends
- Flag accounting red flags or unusual items
- Provide actionable insights with clear reasoning

Output format:
1. Financial Health Assessment
2. Growth Analysis
3. Valuation Analysis
4. Key Risks and Opportunities
5. Recommendation with Score (-1 to 1) and Confidence (0 to 1)
"""

    def __init__(self, claude_client: Optional[ClaudeClient] = None, use_rag: bool = True):
        super().__init__(
            name="FundamentalAnalyst",
            system_prompt=self.SYSTEM_PROMPT,
            claude_client=claude_client,
            use_rag=use_rag,
        )

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform fundamental analysis

        Args:
            symbol: Stock symbol
            data: Dictionary containing:
                - fundamentals: Dict of financial ratios
                - financials: Recent financial statements
                - market_data: Current price, market cap
                - industry_averages: Peer comparison data
            context: Additional context from RAG (e.g., 10-K excerpts)

        Returns:
            AgentResponse with fundamental analysis
        """
        logger.info(f"FundamentalAnalyst analyzing {symbol}")

        # Build Chain-of-Thought prompt
        steps = [
            "Analyze Financial Health: Review profitability (ROE, ROA, margins), "
            "liquidity (current ratio, quick ratio), and leverage (debt-to-equity). "
            "Assess balance sheet strength.",
            "Evaluate Growth: Calculate revenue, earnings, and EPS growth rates YoY. "
            "Compare to historical trends and industry averages. "
            "Identify growth drivers and sustainability.",
            "Assess Valuation: Evaluate P/E, P/B, P/S, EV/EBITDA relative to peers "
            "and historical multiples. Determine if fairly valued, undervalued, or overvalued.",
            "Identify Risks: Flag accounting concerns, debt levels, margin compression, "
            "competitive threats, regulatory risks, management issues.",
            "Provide Recommendation: Synthesize analysis into overall score (-1 to 1) "
            "where -1=strong sell, 0=hold, 1=strong buy. Include confidence (0-1).",
        ]

        prompt = self._build_cot_prompt(symbol, data, steps, context)

        # Add specific instructions for output format
        prompt += "\n\nIMPORTANT: End your analysis with:\n"
        prompt += "Score: [number from -1 to 1]\n"
        prompt += "Confidence: [percentage from 0% to 100%]\n"

        # Get response from Claude
        response = self.claude.complete(
            prompt=prompt,
            system=self.system_prompt,
            max_tokens=2000,
            temperature=0.7,
        )

        # Parse response
        parsed = self._parse_response(response, extract_score=True, extract_confidence=True)

        # Create structured response
        return AgentResponse(
            analysis=parsed["analysis"],
            score=parsed["score"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            data=data,
            agent_name=self.name,
        )

    def analyze_with_tools(
        self,
        symbol: str,
        data: Dict[str, Any],
        enable_calculations: bool = True,
    ) -> AgentResponse:
        """
        Analyze with tool use (calculator, ratio calculator)

        Args:
            symbol: Stock symbol
            data: Financial data
            enable_calculations: Whether to enable calculator tools

        Returns:
            AgentResponse with analysis
        """
        if not enable_calculations:
            return self.analyze(symbol, data)

        # Define tools
        tools = [
            {
                "name": "calculator",
                "description": "Perform financial calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "The calculation to perform (e.g., 'ROE', 'P/E', 'growth_rate')",
                        },
                        "values": {
                            "type": "object",
                            "description": "Values needed for calculation",
                        },
                    },
                    "required": ["operation", "values"],
                },
            }
        ]

        # Build prompt
        prompt = f"""Analyze {symbol}'s fundamentals using the provided data and calculator tool.

Data: {self._format_data(data)}

Use the calculator tool to compute relevant ratios and metrics. Then provide your analysis."""

        # Get response with tools
        response = self.claude.complete_with_tools(
            prompt=prompt,
            tools=tools,
            system=self.system_prompt,
            max_tokens=2000,
        )

        # Extract text from response
        text_content = ""
        for block in response["content"]:
            if hasattr(block, "text"):
                text_content += block.text
            elif block.type == "text":
                text_content += block.text

        parsed = self._parse_response(text_content)

        return AgentResponse(
            analysis=text_content,
            score=parsed["score"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            data=data,
            agent_name=self.name,
        )


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Sample data
    sample_data = {
        "fundamentals": {
            "roe": 28.5,
            "roa": 15.2,
            "profit_margin": 22.5,
            "current_ratio": 1.8,
            "debt_to_equity": 0.65,
            "pe_ratio": 28.3,
            "pb_ratio": 8.5,
            "revenue_growth_yoy": 12.5,
            "earnings_growth_yoy": 15.8,
            "eps_growth_yoy": 16.2,
        },
        "market_data": {
            "current_price": 175.50,
            "market_cap": "2.8T",
        },
        "industry_averages": {
            "pe_ratio": 25.0,
            "roe": 22.0,
            "revenue_growth": 8.5,
        },
    }

    # Create analyst
    analyst = FundamentalAnalyst()

    # Analyze
    result = analyst.analyze(
        symbol="AAPL",
        data=sample_data,
        context="Recent 10-K highlights strong iPhone and Services revenue growth.",
    )

    print(f"\n{'='*60}")
    print(f"Fundamental Analysis: {result.agent_name}")
    print(f"{'='*60}")
    print(f"\nScore: {result.score:.2f} (Range: -1 to 1)")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nAnalysis:\n{result.analysis[:500]}...")
    print(f"\nReasoning Steps: {len(result.reasoning)}")
