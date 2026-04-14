"""Multi-agent LLM system for financial analysis"""

from .claude_client import ClaudeClient
from .base_agent import BaseAgent, AgentResponse
from .base_agent_v2 import BaseAgentV2
from .llm_client_factory import create_client, get_default_client
from .fundamental_analyst import FundamentalAnalyst
from .sentiment_analyst import SentimentAnalyst
from .technical_analyst import TechnicalAnalyst
from .risk_analyst import RiskAnalyst
from .manager_agent import ManagerAgent

# Lazy imports for heavy dependencies (faiss, mlx, etc.)
try:
    from .rag_system import RAGSystem
except ImportError:
    RAGSystem = None

try:
    from .mlx_client import MLXClient
except ImportError:
    MLXClient = None

__all__ = [
    "ClaudeClient",
    "MLXClient",
    "BaseAgent",
    "BaseAgentV2",
    "AgentResponse",
    "create_client",
    "get_default_client",
    "FundamentalAnalyst",
    "SentimentAnalyst",
    "TechnicalAnalyst",
    "RiskAnalyst",
    "ManagerAgent",
    "RAGSystem",
]
