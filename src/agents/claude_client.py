"""
Claude API Client Wrapper

Provides robust integration with Anthropic's Claude API:
- Rate limiting and backoff
- Token counting and cost tracking
- Retry logic with exponential backoff
- Streaming support
- Context window management
"""

import time
from typing import Dict, List, Optional, Union

import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeClient:
    """
    Robust Claude API client with rate limiting and retry logic

    Features:
    - Automatic retry with exponential backoff
    - Token counting and cost estimation
    - Rate limiting (RPM/TPM)
    - Context window management
    - Streaming support
    """

    # Model specifications
    MODELS = {
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_tokens": 8192,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
        },
        "claude-3-opus-20240229": {
            "context_window": 200000,
            "max_tokens": 4096,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
        },
        "claude-3-sonnet-20240229": {
            "context_window": 200000,
            "max_tokens": 4096,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
        },
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "max_tokens": 4096,
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_retries: int = 3,
        rate_limit_rpm: int = 50,  # Requests per minute
    ):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key (if None, loads from env)
            model: Model to use
            max_retries: Maximum retry attempts
            rate_limit_rpm: Rate limit (requests per minute)
        """
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_rpm = rate_limit_rpm

        # Rate limiting
        self.request_times: List[float] = []
        self.min_interval = 60.0 / rate_limit_rpm  # Seconds between requests

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Validate model
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Choose from {list(self.MODELS.keys())}")

        logger.info(f"Initialized Claude client with model: {model}")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, Dict]:
        """
        Generate completion with Claude

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stop_sequences: Stop sequences
            stream: Whether to stream response

        Returns:
            Completion text or full response dict
        """
        # Rate limiting
        self._wait_if_needed()

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Claude API (attempt {attempt + 1}/{self.max_retries})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system if system else anthropic.NOT_GIVEN,
                    messages=messages,
                    stop_sequences=stop_sequences if stop_sequences else anthropic.NOT_GIVEN,
                    stream=stream,
                )

                if stream:
                    return response  # Return stream object

                # Extract text
                text = response.content[0].text

                # Track usage
                self._track_usage(response)

                logger.info(
                    f"Claude API success - Input: {response.usage.input_tokens} tokens, "
                    f"Output: {response.usage.output_tokens} tokens, "
                    f"Cost: ${self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens):.4f}"
                )

                return text

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        raise RuntimeError("Max retries exceeded")

    def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate completion with tool use

        Args:
            prompt: User prompt
            tools: Tool definitions
            system: System prompt
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            Full response with tool calls
        """
        self._wait_if_needed()

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system if system else anthropic.NOT_GIVEN,
                messages=messages,
                tools=tools,
            )

            self._track_usage(response)

            return {
                "content": response.content,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Tool completion error: {e}")
            raise

    def _wait_if_needed(self):
        """Rate limiting: wait if necessary"""
        current_time = time.time()

        # Remove old request times (>1 minute ago)
        self.request_times = [
            t for t in self.request_times if current_time - t < 60
        ]

        # Check if at rate limit
        if len(self.request_times) >= self.rate_limit_rpm:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        # Add current request
        self.request_times.append(current_time)

    def _track_usage(self, response):
        """Track token usage and cost"""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += self._calculate_cost(input_tokens, output_tokens)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for tokens"""
        model_spec = self.MODELS[self.model]

        input_cost = (input_tokens / 1000) * model_spec["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * model_spec["cost_per_1k_output"]

        return input_cost + output_cost

    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Rule of thumb: ~4 characters per token
        """
        return len(text) // 4

    def truncate_to_context(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Truncate text to fit in context window

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (if None, uses model's context window)

        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.MODELS[self.model]["context_window"]

        estimated_tokens = self.estimate_tokens(text)

        if estimated_tokens <= max_tokens:
            return text

        # Truncate (rough approximation)
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token

        return text[:max_chars]


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize client
    client = ClaudeClient(model="claude-3-5-sonnet-20241022")

    # Simple completion
    response = client.complete(
        prompt="Analyze the financial health of Apple Inc. based on recent earnings.",
        system="You are an expert financial analyst.",
        max_tokens=1000,
        temperature=0.7,
    )

    print("Response:", response)
    print("\nUsage stats:", client.get_usage_stats())
