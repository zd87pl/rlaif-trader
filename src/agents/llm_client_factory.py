"""
LLM Client Factory

Returns either ClaudeClient or MLXClient based on configuration.
Reads from config.yaml (llm.backend) and supports LLM_BACKEND env var override.

Usage:
    from .llm_client_factory import create_client, get_default_client

    # Create a new client based on config
    client = create_client()

    # Get the singleton default client
    client = get_default_client()

    # Force a specific backend
    client = create_client(backend="mlx", model="mlx-community/Llama-3-8B-4bit")
"""

import os
from typing import Optional, Union

from ..utils.config import get_config, get_nested
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid requiring both backends installed
_ClaudeClient = None
_MLXClient = None


def _get_claude_client_class():
    global _ClaudeClient
    if _ClaudeClient is None:
        from .claude_client import ClaudeClient
        _ClaudeClient = ClaudeClient
    return _ClaudeClient


def _get_mlx_client_class():
    global _MLXClient
    if _MLXClient is None:
        from .mlx_client import MLXClient
        _MLXClient = MLXClient
    return _MLXClient


def _resolve_backend() -> str:
    """
    Determine which LLM backend to use.

    Priority:
    1. LLM_BACKEND env var (mlx | claude)
    2. config.yaml llm.backend field
    3. Default: 'claude'
    """
    # Check env var first
    env_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if env_backend in ("mlx", "claude"):
        logger.info(f"LLM backend from env var LLM_BACKEND: {env_backend}")
        return env_backend

    # Check config
    try:
        config = get_config()
        config_backend = get_nested(config, "llm.backend", default=None)
        if config_backend and str(config_backend).lower() in ("mlx", "claude"):
            backend = str(config_backend).lower()
            logger.info(f"LLM backend from config: {backend}")
            return backend
    except Exception as e:
        logger.debug(f"Could not read config for backend: {e}")

    # Default
    logger.info("LLM backend defaulting to 'claude'")
    return "claude"


def create_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Union["ClaudeClient", "MLXClient"]:  # noqa: F821
    """
    Create an LLM client based on configuration.

    Args:
        backend: Force a specific backend ('claude' or 'mlx').
                 If None, reads from env/config.
        model: Model name/path override. If None, reads from config.
        **kwargs: Additional keyword arguments passed to the client constructor.

    Returns:
        ClaudeClient or MLXClient instance.
    """
    backend = backend or _resolve_backend()

    # Resolve model from config if not provided
    if model is None:
        try:
            config = get_config()
            if backend == "mlx":
                model = get_nested(config, "llm.mlx_model", default=None)
            else:
                model = get_nested(config, "llm.model", default=None)
        except Exception:
            pass

    if backend == "mlx":
        logger.info(f"Creating MLXClient (model={model})")
        MLXClient = _get_mlx_client_class()
        client_kwargs = {}
        if model:
            client_kwargs["model"] = model
        client_kwargs.update(kwargs)
        return MLXClient(**client_kwargs)

    elif backend == "claude":
        logger.info(f"Creating ClaudeClient (model={model})")
        ClaudeClient = _get_claude_client_class()
        client_kwargs = {}
        if model:
            client_kwargs["model"] = model
        client_kwargs.update(kwargs)
        return ClaudeClient(**client_kwargs)

    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. Must be 'claude' or 'mlx'."
        )


# Singleton default client
_default_client = None


def get_default_client() -> Union["ClaudeClient", "MLXClient"]:  # noqa: F821
    """
    Get the singleton default LLM client.

    Creates the client on first call, then returns the same instance.
    Uses configuration to determine backend and model.

    Returns:
        ClaudeClient or MLXClient instance.
    """
    global _default_client
    if _default_client is None:
        _default_client = create_client()
    return _default_client


def reset_default_client():
    """Reset the singleton so the next get_default_client() creates a fresh one."""
    global _default_client
    _default_client = None
    logger.debug("Default LLM client reset")
