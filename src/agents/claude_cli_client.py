"""Claude CLI client — uses the `claude` CLI as an LLM proxy.

Instead of calling the Anthropic API directly (which requires an API key),
this client shells out to the `claude` CLI tool. This uses your existing
Claude Code authentication — no separate API key needed.

Default model: claude-opus-4-6 (Opus 4.6)

The client implements the same .complete() interface as ClaudeClient,
so it's a drop-in replacement anywhere in the pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeCliClient:
    """LLM client that uses the `claude` CLI for completions.

    Parameters
    ----------
    model : str
        Model to use. Default: claude-opus-4-6
    claude_path : str, optional
        Path to the claude binary. Auto-detected if not provided.
    timeout : int
        Max seconds to wait for a response. Default: 120.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        claude_path: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any,
    ):
        self.model = model
        self.timeout = timeout
        self.claude_path = claude_path or shutil.which("claude")

        if not self.claude_path:
            raise FileNotFoundError(
                "claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code"
            )

        # Usage tracking (compatible with ClaudeClient interface)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        logger.info("ClaudeCliClient initialised (model=%s, cli=%s)", model, self.claude_path)

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, Dict]:
        """Generate a completion via the claude CLI.

        Compatible with ClaudeClient.complete() interface.
        """
        cmd = [
            self.claude_path,
            "-p", prompt,
            "--output-format", "text",
            "--model", self.model,
        ]

        if system:
            cmd.extend(["-s", system])

        if max_tokens:
            cmd.extend(["--max-turns", "1"])

        logger.debug("ClaudeCliClient calling: %s %s", self.model, prompt[:80])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("claude CLI error (rc=%d): %s", result.returncode, stderr[:200])
                # Still return stdout if there's any output
                if result.stdout.strip():
                    return result.stdout.strip()
                raise RuntimeError(f"claude CLI failed: {stderr[:200]}")

            response = result.stdout.strip()
            logger.debug("ClaudeCliClient response: %d chars", len(response))
            return response

        except subprocess.TimeoutExpired:
            logger.warning("claude CLI timed out after %ds", self.timeout)
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4

    def get_usage(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
        }
