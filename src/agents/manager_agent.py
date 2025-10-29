"""
Manager Agent

Orchestrates multi-agent analysis:
- Coordinates specialist agents
- Facilitates structured debate
- Synthesizes consensus
- Makes final trading decisions
- Provides explainable reasoning
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentResponse, ClaudeClient
from .fundamental_analyst import FundamentalAnalyst
from .sentiment_analyst import SentimentAnalyst
from .technical_analyst import TechnicalAnalyst
from .risk_analyst import RiskAnalyst
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ManagerAgent(BaseAgent):
    """
    Portfolio manager agent that synthesizes specialist insights

    Coordinates:
    - FundamentalAnalyst
    - SentimentAnalyst
    - TechnicalAnalyst
    - RiskAnalyst

    Process:
    1. Gather analyses from all specialists
    2. Facilitate structured debate (2 rounds)
    3. Resolve conflicts and weigh perspectives
    4. Make final decision with full reasoning chain
    5. Provide actionable trading signal
    """

    SYSTEM_PROMPT = """You are an experienced portfolio manager synthesizing insights from specialist analysts.

Your role:
- Integrate perspectives from fundamental, sentiment, technical, and risk analysts
- Facilitate structured debate to challenge assumptions
- Resolve conflicts between conflicting signals
- Weight analyst inputs based on market context and confidence
- Make final trading decisions with clear reasoning

Approach:
- Consider all perspectives, but weight based on current market regime
- Challenge consensus when appropriate (contrarian thinking)
- Prioritize risk management and capital preservation
- Provide actionable signals: BUY, HOLD, SELL
- Explain reasoning clearly for transparency

Decision framework:
- Strong fundamental + positive sentiment + bullish technicals + acceptable risk = BUY
- Mixed signals or high uncertainty = HOLD
- Weak fundamentals + negative sentiment + bearish technicals = SELL
- Elevated risk always reduces position size or triggers SELL

Output format:
1. Summary of Analyst Perspectives
2. Key Agreements and Disagreements
3. Debate Resolution and Synthesis
4. Final Decision: BUY/HOLD/SELL
5. Score (-1 to 1), Confidence (0 to 1), Position Size (% of portfolio)
"""

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        debate_rounds: int = 2,
    ):
        """
        Initialize manager agent

        Args:
            claude_client: Claude client (shared across agents)
            debate_rounds: Number of debate rounds
        """
        super().__init__(
            name="ManagerAgent",
            system_prompt=self.SYSTEM_PROMPT,
            claude_client=claude_client,
            use_rag=False,
        )

        self.debate_rounds = debate_rounds

        # Initialize specialist agents
        self.fundamental = FundamentalAnalyst(claude_client=self.claude)
        self.sentiment = SentimentAnalyst(claude_client=self.claude)
        self.technical = TechnicalAnalyst(claude_client=self.claude)
        self.risk = RiskAnalyst(claude_client=self.claude)

        logger.info(f"Initialized {self.name} with {debate_rounds} debate rounds")

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Orchestrate multi-agent analysis and make final decision

        Args:
            symbol: Stock symbol
            data: Dictionary containing data for all analysts:
                - fundamentals: For FundamentalAnalyst
                - sentiment: For SentimentAnalyst
                - technical: For TechnicalAnalyst
                - risk: For RiskAnalyst
            context: Additional context

        Returns:
            AgentResponse with final decision
        """
        logger.info(f"ManagerAgent orchestrating analysis for {symbol}")

        # Step 1: Gather initial analyses from specialists
        specialist_responses = self._gather_specialist_analyses(symbol, data, context)

        # Step 2: Facilitate debate
        debate_summary = self._facilitate_debate(symbol, specialist_responses)

        # Step 3: Synthesize and make final decision
        final_decision = self._make_final_decision(
            symbol, specialist_responses, debate_summary
        )

        return final_decision

    def _gather_specialist_analyses(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, AgentResponse]:
        """Gather analyses from all specialist agents"""
        logger.info("Gathering specialist analyses...")

        responses = {}

        # Fundamental analysis
        if "fundamentals" in data:
            logger.info("Running fundamental analysis...")
            responses["fundamental"] = self.fundamental.analyze(
                symbol, data["fundamentals"], context
            )

        # Sentiment analysis
        if "sentiment" in data:
            logger.info("Running sentiment analysis...")
            responses["sentiment"] = self.sentiment.analyze(
                symbol, data["sentiment"], context
            )

        # Technical analysis
        if "technical" in data:
            logger.info("Running technical analysis...")
            responses["technical"] = self.technical.analyze(
                symbol, data["technical"], None
            )

        # Risk analysis
        if "risk" in data:
            logger.info("Running risk analysis...")
            responses["risk"] = self.risk.analyze(symbol, data["risk"], None)

        logger.info(f"Gathered {len(responses)} specialist analyses")
        return responses

    def _facilitate_debate(
        self,
        symbol: str,
        specialist_responses: Dict[str, AgentResponse],
    ) -> str:
        """
        Facilitate structured debate between specialists

        Args:
            symbol: Stock symbol
            specialist_responses: Responses from all specialists

        Returns:
            Debate summary
        """
        if len(specialist_responses) < 2:
            return "Insufficient analysts for debate."

        logger.info(f"Facilitating {self.debate_rounds} debate rounds...")

        # Build debate prompt
        debate_prompt = f"Facilitate a structured debate about {symbol} among the following analysts:\n\n"

        for name, response in specialist_responses.items():
            debate_prompt += f"**{name.upper()} (Score: {response.score:.2f}, Confidence: {response.confidence:.0%})**\n"
            debate_prompt += f"{response.analysis[:300]}...\n\n"

        debate_prompt += f"""
Conduct {self.debate_rounds} rounds of debate:

Round 1 - Challenge Assumptions:
- Identify areas where analysts disagree
- Have each analyst defend their position
- Challenge weak or contradictory reasoning

Round 2 - Synthesize:
- Find common ground and key insights
- Weigh conflicting signals based on strength of evidence
- Identify the most important factors for the decision

Provide a summary of the debate outcome.
"""

        debate_response = self.claude.complete(
            prompt=debate_prompt,
            system="You are facilitating a structured debate among financial analysts.",
            max_tokens=1500,
            temperature=0.7,
        )

        logger.info("Debate facilitation complete")
        return debate_response

    def _make_final_decision(
        self,
        symbol: str,
        specialist_responses: Dict[str, AgentResponse],
        debate_summary: str,
    ) -> AgentResponse:
        """
        Make final trading decision based on specialist inputs and debate

        Args:
            symbol: Stock symbol
            specialist_responses: All specialist responses
            debate_summary: Summary of debate

        Returns:
            Final AgentResponse with decision
        """
        logger.info("Making final trading decision...")

        # Build synthesis prompt
        synthesis_prompt = f"""As portfolio manager, make final trading decision for {symbol}.

SPECIALIST ANALYSES:
"""

        for name, response in specialist_responses.items():
            synthesis_prompt += f"\n{name.upper()}:\n"
            synthesis_prompt += f"- Score: {response.score:.2f}\n"
            synthesis_prompt += f"- Confidence: {response.confidence:.0%}\n"
            synthesis_prompt += f"- Key points: {response.analysis[:200]}...\n"

        synthesis_prompt += f"\n\nDEBATE SUMMARY:\n{debate_summary}\n\n"

        synthesis_prompt += """
Based on all analyses and the debate, provide your final decision:

1. Weigh all perspectives considering:
   - Strength of evidence and confidence
   - Agreement vs. disagreement among analysts
   - Current market context
   - Risk management principles

2. Make decision: BUY, HOLD, or SELL

3. Provide:
   - Overall Score: -1 (strong sell) to 1 (strong buy)
   - Confidence: 0% to 100%
   - Recommended Position Size: % of portfolio
   - Clear reasoning for decision

Format:
Decision: [BUY/HOLD/SELL]
Score: [number from -1 to 1]
Confidence: [percentage]
Position Size: [percentage]
Reasoning: [your explanation]
"""

        final_response = self.claude.complete(
            prompt=synthesis_prompt,
            system=self.system_prompt,
            max_tokens=2000,
            temperature=0.7,
        )

        # Parse final decision
        parsed = self._parse_response(final_response, extract_score=True, extract_confidence=True)

        # Combine all data
        combined_data = {
            "specialist_scores": {
                name: resp.score for name, resp in specialist_responses.items()
            },
            "specialist_confidence": {
                name: resp.confidence for name, resp in specialist_responses.items()
            },
            "debate_summary": debate_summary,
        }

        logger.info(f"Final decision: Score={parsed['score']:.2f}, Confidence={parsed['confidence']:.0%}")

        return AgentResponse(
            analysis=final_response,
            score=parsed["score"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            data=combined_data,
            agent_name=self.name,
        )


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Sample data for all analysts
    sample_data = {
        "fundamentals": {
            "roe": 28.5,
            "roa": 15.2,
            "profit_margin": 22.5,
            "pe_ratio": 28.3,
            "revenue_growth_yoy": 12.5,
        },
        "sentiment": {
            "news_sentiment": 0.75,
            "social_sentiment": 0.65,
            "analyst_ratings": "12 Buy, 3 Hold, 1 Sell",
        },
        "technical": {
            "current_price": 175.50,
            "rsi": 68,
            "macd": 2.5,
            "trend": "uptrend",
        },
        "risk": {
            "volatility": 0.25,
            "max_drawdown": 0.18,
            "sharpe_ratio": 1.8,
            "beta": 1.1,
        },
    }

    # Create manager
    manager = ManagerAgent(debate_rounds=2)

    # Perform multi-agent analysis
    result = manager.analyze(symbol="AAPL", data=sample_data)

    print(f"\n{'='*60}")
    print(f"MULTI-AGENT DECISION: {result.agent_name}")
    print(f"{'='*60}")
    print(f"\nFinal Score: {result.score:.2f} (Range: -1 to 1)")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nDecision Summary:\n{result.analysis[:400]}...")
    print(f"\nSpecialist Scores: {result.data.get('specialist_scores')}")
    print(f"{'='*60}")
