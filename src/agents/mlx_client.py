"""
MLX-LM Local Inference Client

Drop-in replacement for ClaudeClient that uses MLX-LM for local inference
on Apple Silicon instead of the Anthropic API.

Features:
- Same interface as ClaudeClient (complete(), complete_with_tools(), etc.)
- Local inference via mlx-lm on Apple Silicon
- Model registry with common mlx-community models
- Token usage tracking
- Graceful fallback with clear errors if MLX not available
"""

import time
from typing import Dict, List, Optional, Union

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Registry – common mlx-community HuggingFace models
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict] = {
    # ── Qwen ──────────────────────────────────────────────────────────────
    "qwen-72b-4bit": {
        "repo": "mlx-community/Qwen2.5-72B-Instruct-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "qwen",
        "description": "Qwen 2.5 72B Instruct – 4-bit quantised",
    },
    "qwen-32b-4bit": {
        "repo": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "qwen",
        "description": "Qwen 2.5 32B Instruct – 4-bit quantised",
    },
    "qwen-7b-4bit": {
        "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "qwen",
        "description": "Qwen 2.5 7B Instruct – 4-bit quantised",
    },
    "qwen-3b-4bit": {
        "repo": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "context_window": 32768,
        "max_tokens": 4096,
        "chat_template": "qwen",
        "description": "Qwen 2.5 3B Instruct – 4-bit quantised",
    },
    # ── Llama ─────────────────────────────────────────────────────────────
    "llama-70b-4bit": {
        "repo": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "llama3",
        "description": "Llama 3.1 70B Instruct – 4-bit quantised",
    },
    "llama-8b-4bit": {
        "repo": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "llama3",
        "description": "Llama 3.1 8B Instruct – 4-bit quantised",
    },
    # ── Mistral ───────────────────────────────────────────────────────────
    "mistral-7b-4bit": {
        "repo": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "context_window": 32768,
        "max_tokens": 4096,
        "chat_template": "mistral",
        "description": "Mistral 7B Instruct v0.3 – 4-bit quantised",
    },
    "mixtral-8x7b-4bit": {
        "repo": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        "context_window": 32768,
        "max_tokens": 4096,
        "chat_template": "mistral",
        "description": "Mixtral 8x7B Instruct – 4-bit quantised",
    },
    # ── Gemma ─────────────────────────────────────────────────────────────
    "gemma-27b-4bit": {
        "repo": "mlx-community/gemma-2-27b-it-4bit",
        "context_window": 8192,
        "max_tokens": 4096,
        "chat_template": "gemma",
        "description": "Gemma 2 27B IT – 4-bit quantised",
    },
    "gemma-9b-4bit": {
        "repo": "mlx-community/gemma-2-9b-it-4bit",
        "context_window": 8192,
        "max_tokens": 4096,
        "chat_template": "gemma",
        "description": "Gemma 2 9B IT – 4-bit quantised",
    },
    # ── Phi ────────────────────────────────────────────────────────────────
    "phi-4-4bit": {
        "repo": "mlx-community/phi-4-4bit",
        "context_window": 16384,
        "max_tokens": 4096,
        "chat_template": "phi",
        "description": "Phi 4 – 4-bit quantised",
    },
    # ── DeepSeek ──────────────────────────────────────────────────────────
    "deepseek-r1-8b-4bit": {
        "repo": "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        "context_window": 131072,
        "max_tokens": 8192,
        "chat_template": "llama3",
        "description": "DeepSeek R1 Distill Llama 8B – 4-bit quantised",
    },
}

# Aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    "qwen-72b": "qwen-72b-4bit",
    "qwen-32b": "qwen-32b-4bit",
    "qwen-7b": "qwen-7b-4bit",
    "qwen-3b": "qwen-3b-4bit",
    "llama-70b": "llama-70b-4bit",
    "llama-8b": "llama-8b-4bit",
    "mistral-7b": "mistral-7b-4bit",
    "mixtral": "mixtral-8x7b-4bit",
    "gemma-27b": "gemma-27b-4bit",
    "gemma-9b": "gemma-9b-4bit",
    "phi-4": "phi-4-4bit",
    "deepseek-r1": "deepseek-r1-8b-4bit",
}

# Default model if none specified
DEFAULT_MODEL = "qwen-7b-4bit"


class MLXClient:
    """
    Drop-in replacement for ClaudeClient using MLX-LM for local inference.

    Same interface as ClaudeClient so it can be swapped transparently:
        client = MLXClient(model='qwen-7b-4bit')
        response = client.complete("Analyse AAPL", system="You are a financial analyst.")

    You can also pass a raw mlx-community repo path as the model name:
        client = MLXClient(model='mlx-community/Qwen2.5-7B-Instruct-4bit')
    """

    MODELS = MODEL_REGISTRY

    def __init__(
        self,
        api_key: Optional[str] = None,         # ignored – kept for interface compat
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,                   # ignored – kept for interface compat
        rate_limit_rpm: int = 50,               # ignored – kept for interface compat
    ):
        if not MLX_AVAILABLE:
            raise ImportError(
                "mlx-lm is not installed. Install it with:\n"
                "  pip install mlx-lm\n"
                "MLX requires Apple Silicon (M1/M2/M3/M4)."
            )

        # Resolve aliases
        resolved = MODEL_ALIASES.get(model, model)

        # Check if it's a registry key or a raw HuggingFace repo path
        if resolved in MODEL_REGISTRY:
            self._model_key = resolved
            self._model_spec = MODEL_REGISTRY[resolved]
            self._repo = self._model_spec["repo"]
        elif "/" in model:
            # Treat as raw HuggingFace repo (e.g. 'mlx-community/SomeModel')
            self._model_key = model
            self._repo = model
            self._model_spec = {
                "repo": model,
                "context_window": 32768,
                "max_tokens": 4096,
                "chat_template": "chatml",
                "description": f"Custom model: {model}",
            }
        else:
            available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
            raise ValueError(
                f"Unknown model: {model}. "
                f"Choose from {available} or pass a full HuggingFace repo path."
            )

        self.model = self._model_key
        self.max_retries = max_retries

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0  # always 0 for local inference
        self._request_count = 0

        # Load model & tokenizer
        logger.info(f"Loading MLX model: {self._repo} ...")
        _t0 = time.time()
        self._model, self._tokenizer = load(self._repo)
        _elapsed = time.time() - _t0
        logger.info(f"Model loaded in {_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def _format_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """
        Format prompt + system message using the tokenizer's built-in
        chat template (preferred), falling back to ChatML if unavailable.
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Most HF tokenizers ship with apply_chat_template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception as exc:
                logger.debug(f"apply_chat_template failed ({exc}), using fallback")

        # Fallback: ChatML
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

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
        Generate a completion using MLX-LM.

        Interface matches ClaudeClient.complete() exactly.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stop_sequences: Stop sequences (best-effort post-processing)
            stream: Streaming stub – not yet implemented, returns string

        Returns:
            Generated text string (or dict stub if stream=True)
        """
        if stream:
            logger.warning(
                "Streaming is not yet implemented for MLXClient. "
                "Returning full response."
            )

        formatted_prompt = self._format_prompt(prompt, system)

        # Count input tokens
        input_token_ids = self._tokenize(formatted_prompt)
        input_token_count = len(input_token_ids)

        # Clamp max_tokens to model limit
        model_max = self._model_spec.get("max_tokens", 4096)
        effective_max_tokens = min(max_tokens, model_max)

        logger.info(
            f"MLX generate – input tokens: {input_token_count}, "
            f"max_tokens: {effective_max_tokens}, temp: {temperature}"
        )

        _t0 = time.time()

        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=effective_max_tokens,
            temp=temperature,
        )

        _elapsed = time.time() - _t0

        # Post-process: apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                idx = response_text.find(seq)
                if idx != -1:
                    response_text = response_text[:idx]

        # Count output tokens
        output_token_ids = self._tokenize(response_text)
        output_token_count = len(output_token_ids)

        # Track usage
        self.total_input_tokens += input_token_count
        self.total_output_tokens += output_token_count
        self._request_count += 1

        tokens_per_sec = output_token_count / _elapsed if _elapsed > 0 else 0
        logger.info(
            f"MLX generate done – output tokens: {output_token_count}, "
            f"time: {_elapsed:.2f}s, speed: {tokens_per_sec:.1f} tok/s"
        )

        return response_text

    def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate a completion that may include tool calls.

        For local models this works by injecting tool descriptions into the
        system prompt and asking the model to output structured JSON when it
        wants to call a tool.  The caller is responsible for parsing the
        response and executing tools.

        Interface matches ClaudeClient.complete_with_tools() exactly.
        """
        # Build tool description block
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("input_schema", tool.get("parameters", {}))
            tool_descriptions.append(
                f"Tool: {name}\nDescription: {desc}\n"
                f"Parameters: {params}"
            )

        tool_block = (
            "You have access to the following tools. To use a tool, respond "
            "with a JSON object containing \"tool_name\" and \"tool_input\" keys.\n\n"
            + "\n---\n".join(tool_descriptions)
        )

        combined_system = f"{system}\n\n{tool_block}" if system else tool_block

        response_text = self.complete(
            prompt=prompt,
            system=combined_system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Return in the same shape as ClaudeClient
        return {
            "content": [{"type": "text", "text": response_text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list:
        """Tokenize text using the loaded tokenizer, returning token ids."""
        if hasattr(self._tokenizer, "encode"):
            ids = self._tokenizer.encode(text)
            if isinstance(ids, list):
                return ids
            # Some tokenizers return a BatchEncoding or similar
            if hasattr(ids, "input_ids"):
                return ids.input_ids
            return list(ids)
        # Fallback heuristic
        return list(range(len(text) // 4))

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses the actual tokenizer when loaded, matching ClaudeClient's
        interface (which uses a ~4 chars/token heuristic).
        """
        return len(self._tokenize(text))

    def truncate_to_context(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Truncate text to fit within the model's context window.

        Args:
            text: Text to truncate
            max_tokens: Max tokens (defaults to model context window)

        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self._model_spec.get("context_window", 32768)

        estimated = self.estimate_tokens(text)
        if estimated <= max_tokens:
            return text

        # Use tokenizer for precise truncation
        tokens = self._tokenize(text)
        truncated_tokens = tokens[:max_tokens]

        if hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(truncated_tokens)

        # Fallback: character-level approximation
        chars_per_token = 4
        return text[: max_tokens * chars_per_token]

    # ------------------------------------------------------------------
    # Usage & stats
    # ------------------------------------------------------------------

    def get_usage_stats(self) -> Dict:
        """
        Get usage statistics.

        Returns dict with same keys as ClaudeClient.get_usage_stats().
        total_cost is always 0.0 since inference is local.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": 0.0,
            "model": self.model,
            "request_count": self._request_count,
            "backend": "mlx-lm",
        }

    # ------------------------------------------------------------------
    # Utility class methods
    # ------------------------------------------------------------------

    @staticmethod
    def list_models() -> Dict[str, str]:
        """Return {key: description} for all models in the registry."""
        return {k: v["description"] for k, v in MODEL_REGISTRY.items()}

    @staticmethod
    def is_available() -> bool:
        """Check whether MLX-LM is importable on this system."""
        return MLX_AVAILABLE

    def __repr__(self) -> str:
        return (
            f"MLXClient(model={self.model!r}, repo={self._repo!r}, "
            f"requests={self._request_count})"
        )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available MLX models:")
    for key, desc in MLXClient.list_models().items():
        print(f"  {key:25s} – {desc}")

    if MLXClient.is_available():
        client = MLXClient(model="qwen-7b-4bit")

        response = client.complete(
            prompt="Analyse the financial health of Apple Inc. based on recent earnings.",
            system="You are an expert financial analyst.",
            max_tokens=1024,
            temperature=0.7,
        )

        print("\nResponse:", response)
        print("\nUsage stats:", client.get_usage_stats())
    else:
        print("\nMLX-LM not available on this system.")
