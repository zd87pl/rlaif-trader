"""
Sentiment Analyst Agent

Analyzes:
- News sentiment
- Social media trends
- Earnings call transcripts
- Analyst reports
- Market psychology
"""

from typing import Any, Dict, Optional

from .base_agent import BaseAgent, AgentResponse, ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyst(BaseAgent):
    """
    Specialized agent for sentiment analysis

    Expertise:
    - News sentiment interpretation
    - Social media trend analysis
    - Earnings call tone and confidence
    - Analyst opinion synthesis
    - Market psychology and momentum
    """

    SYSTEM_PROMPT = """You are an expert sentiment analyst for financial markets.

Your role:
- Interpret news sentiment beyond surface-level positive/negative
- Identify narrative shifts and emerging themes
- Assess management tone and confidence in earnings calls
- Synthesize analyst opinions and market psychology
- Detect hype, fear, euphoria, or panic

Approach:
- Look beyond headlines to underlying implications
- Consider sentiment context (is "good" news already priced in?)
- Identify sentiment divergence (news vs. price action)
- Assess sentiment momentum and potential reversals
- Weight recent sentiment more heavily

Output format:
1. News Sentiment Assessment
2. Social Media & Analyst Sentiment
3. Earnings Call Tone Analysis
4. Market Psychology & Momentum
5. Recommendation with Score (-1 to 1) and Confidence (0 to 1)
"""

    def __init__(self, claude_client: Optional[ClaudeClient] = None, use_rag: bool = True):
        super().__init__(
            name="SentimentAnalyst",
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
        Perform sentiment analysis

        Args:
            symbol: Stock symbol
            data: Dictionary containing:
                - news_sentiment: FinBERT scores from recent news
                - social_sentiment: Social media sentiment
                - analyst_ratings: Recent analyst opinions
                - earnings_transcript: Recent earnings call (if available)
            context: Additional context from RAG (e.g., historical sentiment patterns)

        Returns:
            AgentResponse with sentiment analysis
        """
        logger.info(f"SentimentAnalyst analyzing {symbol}")

        steps = [
            "Analyze News Sentiment: Review recent news sentiment scores and major headlines. "
            "Assess if sentiment is improving, deteriorating, or stable. Consider novelty.",
            "Evaluate Social & Analyst Sentiment: Interpret social media trends and analyst rating changes. "
            "Look for consensus vs. contrarian signals.",
            "Assess Earnings Call Tone: If available, analyze management tone, confidence level, "
            "and forward guidance sentiment.",
            "Identify Market Psychology: Determine overall market mood (euphoric, fearful, rational). "
            "Assess if sentiment is aligned with fundamentals or diverging.",
            "Provide Sentiment Score: Synthesize into score (-1=very negative, 0=neutral, 1=very positive). "
            "Include confidence based on data quality and consistency.",
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
