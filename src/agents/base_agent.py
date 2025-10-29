"""
Base Agent Class

All specialized agents inherit from this base class,
which provides:
- Claude API integration
- Tool use capabilities
- Chain-of-Thought reasoning
- Structured output
- RAG integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .claude_client import ClaudeClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Structured response from an agent

    Contains:
    - analysis: Main analysis text
    - score: Numerical score (e.g., -1 to 1 for sentiment)
    - confidence: Confidence level (0-1)
    - reasoning: Step-by-step reasoning chain
    - data: Additional structured data
    """

    analysis: str
    score: float
    confidence: float
    reasoning: List[str]
    data: Dict[str, Any]
    agent_name: str


class BaseAgent(ABC):
    """
    Base class for all financial analysis agents

    Provides common functionality:
    - Claude API integration
    - Tool use
    - Chain-of-Thought reasoning
    - Structured output formatting
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        claude_client: Optional[ClaudeClient] = None,
        tools: Optional[List[Dict]] = None,
        use_rag: bool = False,
    ):
        """
        Initialize base agent

        Args:
            name: Agent name
            system_prompt: System prompt defining agent's role
            claude_client: Claude client (creates new if None)
            tools: Tool definitions for Claude
            use_rag: Whether to use RAG for context
        """
        self.name = name
        self.system_prompt = system_prompt
        self.claude = claude_client or ClaudeClient()
        self.tools = tools or []
        self.use_rag = use_rag

        logger.info(f"Initialized {name} agent")

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform analysis (implemented by each agent)

        Args:
            symbol: Stock symbol
            data: Relevant data for analysis
            context: Additional context (e.g., from RAG)

        Returns:
            AgentResponse with analysis
        """
        pass

    def _build_cot_prompt(
        self,
        symbol: str,
        data: Dict[str, Any],
        steps: List[str],
        context: Optional[str] = None,
    ) -> str:
        """
        Build Chain-of-Thought prompt

        Args:
            symbol: Stock symbol
            data: Analysis data
            steps: Reasoning steps to follow
            context: Additional context

        Returns:
            Formatted prompt
        """
        prompt = f"Analyze {symbol} following these steps:\n\n"

        for i, step in enumerate(steps, 1):
            prompt += f"{i}. {step}\n"

        prompt += f"\nData:\n{self._format_data(data)}\n"

        if context:
            prompt += f"\nRelevant Context:\n{context}\n"

        prompt += "\nProvide your analysis following the steps above. "
        prompt += "For each step, explain your reasoning clearly."

        return prompt

    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for prompt"""
        lines = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value}")
            elif isinstance(value, str):
                lines.append(f"- {key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"- {key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                if len(value) <= 5:
                    lines.append(f"- {key}: {value}")
                else:
                    lines.append(f"- {key}: {value[:3]}... ({len(value)} items)")
        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        extract_score: bool = True,
        extract_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse Claude's response to extract structured data

        Args:
            response: Raw response text
            extract_score: Whether to extract numerical score
            extract_confidence: Whether to extract confidence

        Returns:
            Parsed data dictionary
        """
        parsed = {
            "analysis": response,
            "score": 0.0,
            "confidence": 0.5,
            "reasoning": [],
        }

        # Extract reasoning steps (lines starting with numbers)
        lines = response.split("\n")
        for line in lines:
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith("-")):
                parsed["reasoning"].append(line.strip())

        # Extract score if requested
        if extract_score:
            score = self._extract_score(response)
            if score is not None:
                parsed["score"] = score

        # Extract confidence if requested
        if extract_confidence:
            confidence = self._extract_confidence(response)
            if confidence is not None:
                parsed["confidence"] = confidence

        return parsed

    def _extract_score(self, text: str) -> Optional[float]:
        """
        Extract numerical score from text

        Looks for patterns like:
        - "Score: 0.75"
        - "Rating: 8/10"
        - "Buy (7/10)"
        """
        import re

        # Pattern 1: "Score: X" or "Rating: X"
        pattern1 = r"(?:score|rating):\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Pattern 2: "X/10" or "X out of 10"
        pattern2 = r"(\d+)\s*(?:/|out of)\s*10"
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 10.0

        # Pattern 3: Buy/Hold/Sell keywords
        if "strong buy" in text.lower():
            return 1.0
        elif "buy" in text.lower():
            return 0.7
        elif "hold" in text.lower():
            return 0.0
        elif "sell" in text.lower():
            return -0.7
        elif "strong sell" in text.lower():
            return -1.0

        return None

    def _extract_confidence(self, text: str) -> Optional[float]:
        """
        Extract confidence level from text

        Looks for patterns like:
        - "Confidence: 80%"
        - "High confidence"
        - "Low confidence"
        """
        import re

        # Pattern: "Confidence: X%"
        pattern = r"confidence:\s*(\d+)%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0

        # Keywords
        if "high confidence" in text.lower():
            return 0.9
        elif "medium confidence" in text.lower() or "moderate confidence" in text.lower():
            return 0.6
        elif "low confidence" in text.lower():
            return 0.3

        return None

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Execute a tool

        Override this method to implement tool execution logic

        Args:
            tool_name: Name of tool to execute
            tool_input: Tool input parameters

        Returns:
            Tool execution result
        """
        logger.warning(f"Tool execution not implemented for {tool_name}")
        return None

    def get_context_from_rag(
        self,
        symbol: str,
        query: str,
        rag_system: Optional[Any] = None,
        top_k: int = 5,
    ) -> str:
        """
        Get context from RAG system

        Args:
            symbol: Stock symbol
            query: Query for retrieval
            rag_system: RAG system instance
            top_k: Number of documents to retrieve

        Returns:
            Retrieved context as string
        """
        if not self.use_rag or not rag_system:
            return ""

        try:
            results = rag_system.retrieve(query, symbol=symbol, top_k=top_k)
            context = "\n\n".join([r["text"] for r in results])
            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""


# Example agent implementation
class ExampleAnalyst(BaseAgent):
    """Example agent demonstrating BaseAgent usage"""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        super().__init__(
            name="ExampleAnalyst",
            system_prompt="You are a financial analyst providing investment recommendations.",
            claude_client=claude_client,
        )

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """Perform analysis"""
        # Build Chain-of-Thought prompt
        steps = [
            "Analyze the current price and recent trends",
            "Evaluate the data provided",
            "Determine overall outlook (bullish/bearish)",
            "Provide a recommendation with confidence level",
        ]

        prompt = self._build_cot_prompt(symbol, data, steps, context)

        # Get response from Claude
        response = self.claude.complete(
            prompt=prompt,
            system=self.system_prompt,
            max_tokens=1500,
            temperature=0.7,
        )

        # Parse response
        parsed = self._parse_response(response)

        # Create structured response
        return AgentResponse(
            analysis=parsed["analysis"],
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

    # Create agent
    agent = ExampleAnalyst()

    # Analyze
    response = agent.analyze(
        symbol="AAPL",
        data={
            "current_price": 175.50,
            "52week_high": 199.62,
            "52week_low": 164.08,
            "pe_ratio": 28.5,
            "recent_news": "Strong iPhone sales reported",
        },
    )

    print(f"Agent: {response.agent_name}")
    print(f"Analysis: {response.analysis[:200]}...")
    print(f"Score: {response.score}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning steps: {len(response.reasoning)}")
