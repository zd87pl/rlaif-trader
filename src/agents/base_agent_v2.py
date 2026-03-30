"""
Base Agent V2

Extends BaseAgent to accept either ClaudeClient or MLXClient transparently.
Uses duck typing -- both clients expose .complete() with the same signature,
so this class does not care which backend is in use.

Usage:
    # Create from config (auto-selects backend)
    agent = MyAgent.from_config()

    # Or pass any client manually
    agent = MyAgent(name="test", system_prompt="...", llm_client=my_client)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base_agent import AgentResponse, BaseAgent
from .llm_client_factory import create_client, get_default_client
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for either client (duck typed)
LLMClient = Any


class BaseAgentV2(ABC):
    """
    Base class for all financial analysis agents (V2).

    Like BaseAgent but accepts any LLM client (ClaudeClient or MLXClient)
    via duck typing. Both clients implement .complete() with the same
    interface, so they are interchangeable.

    Provides common functionality:
    - LLM integration (Claude API or local MLX)
    - Tool use
    - Chain-of-Thought reasoning
    - Structured output formatting
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: Optional[LLMClient] = None,
        tools: Optional[List[Dict]] = None,
        use_rag: bool = False,
    ):
        """
        Initialize base agent v2.

        Args:
            name: Agent name
            system_prompt: System prompt defining agent's role
            llm_client: LLM client (ClaudeClient or MLXClient).
                        If None, creates one from config via factory.
            tools: Tool definitions for the LLM
            use_rag: Whether to use RAG for context
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm_client or get_default_client()
        self.tools = tools or []
        self.use_rag = use_rag

        # Keep backward-compatible alias
        self.claude = self.llm

        logger.info(
            f"Initialized {name} agent (backend: {type(self.llm).__name__})"
        )

    @classmethod
    def from_config(cls, **kwargs) -> "BaseAgentV2":
        """
        Create an agent with an LLM client chosen by configuration.

        Reads LLM_BACKEND env var or config.yaml to pick the backend,
        then passes the client into __init__.

        Args:
            **kwargs: Forwarded to the subclass __init__.

        Returns:
            Agent instance with the configured LLM client.
        """
        client = create_client()
        kwargs.setdefault("llm_client", client)
        return cls(**kwargs)

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform analysis (implemented by each agent).

        Args:
            symbol: Stock symbol
            data: Relevant data for analysis
            context: Additional context (e.g., from RAG)

        Returns:
            AgentResponse with analysis
        """
        pass

    # ------------------------------------------------------------------
    # Prompt building helpers (same as BaseAgent)
    # ------------------------------------------------------------------

    def _build_cot_prompt(
        self,
        symbol: str,
        data: Dict[str, Any],
        steps: List[str],
        context: Optional[str] = None,
    ) -> str:
        """
        Build Chain-of-Thought prompt.

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
        """Format data dictionary for prompt."""
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
        Parse LLM response to extract structured data.

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
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
                parsed["reasoning"].append(stripped)

        if extract_score:
            score = self._extract_score(response)
            if score is not None:
                parsed["score"] = score

        if extract_confidence:
            confidence = self._extract_confidence(response)
            if confidence is not None:
                parsed["confidence"] = confidence

        return parsed

    def _extract_score(self, text: str) -> Optional[float]:
        """Extract numerical score from text."""
        import re

        pattern1 = r"(?:score|rating):\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        pattern2 = r"(\d+)\s*(?:/|out of)\s*10"
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 10.0

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
        """Extract confidence level from text."""
        import re

        pattern = r"confidence:\s*(\d+)%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0

        if "high confidence" in text.lower():
            return 0.9
        elif "medium confidence" in text.lower() or "moderate confidence" in text.lower():
            return 0.6
        elif "low confidence" in text.lower():
            return 0.3

        return None

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Execute a tool.

        Override this method to implement tool execution logic.

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
        Get context from RAG system.

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
