"""
MLX Model Manager for RLAIF Trading System.

Manages downloading, loading, and serving MLX-optimized language models
for local inference on Apple Silicon. Implements singleton pattern to
ensure only one model is loaded at a time for memory management.
"""

import gc
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Conditional MLX imports
try:
    import mlx.core as mx
    import mlx_lm

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning(
        "MLX libraries not available. Install with: pip install mlx mlx-lm"
    )

try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning(
        "huggingface_hub not available. Install with: pip install huggingface-hub"
    )


class RAMTier(str, Enum):
    """RAM tier categories for model selection."""

    TIER_24GB = "24GB"
    TIER_48GB = "48GB"
    TIER_64GB = "64GB"
    TIER_128GB = "128GB"


@dataclass
class ModelInfo:
    """Metadata for a registered MLX model."""

    model_id: str
    display_name: str
    parameter_count: str
    quantization: str
    min_ram_gb: int
    recommended_ram_gb: int
    context_length: int
    ram_tier: RAMTier
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "min_ram_gb": self.min_ram_gb,
            "recommended_ram_gb": self.recommended_ram_gb,
            "context_length": self.context_length,
            "ram_tier": self.ram_tier.value,
            "description": self.description,
            "tags": self.tags,
        }


# Model registry with recommended models for different RAM tiers
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # 24GB RAM tier
    "mlx-community/Qwen2.5-7B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        display_name="Qwen 2.5 7B Instruct (4-bit)",
        parameter_count="7B",
        quantization="4-bit",
        min_ram_gb=8,
        recommended_ram_gb=24,
        context_length=32768,
        ram_tier=RAMTier.TIER_24GB,
        description="Compact and efficient model suitable for systems with 24GB RAM.",
        tags=["qwen", "7b", "4bit", "instruct", "lightweight"],
    ),
    "mlx-community/Llama-3.1-8B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Llama-3.1-8B-Instruct-4bit",
        display_name="Llama 3.1 8B Instruct (4-bit)",
        parameter_count="8B",
        quantization="4-bit",
        min_ram_gb=8,
        recommended_ram_gb=24,
        context_length=131072,
        ram_tier=RAMTier.TIER_24GB,
        description="Meta's Llama 3.1 optimized for instruction following.",
        tags=["llama", "8b", "4bit", "instruct", "lightweight"],
    ),
    # 48GB RAM tier
    "mlx-community/Qwen2.5-14B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
        display_name="Qwen 2.5 14B Instruct (4-bit)",
        parameter_count="14B",
        quantization="4-bit",
        min_ram_gb=16,
        recommended_ram_gb=48,
        context_length=32768,
        ram_tier=RAMTier.TIER_48GB,
        description="Mid-range model with strong reasoning for financial analysis.",
        tags=["qwen", "14b", "4bit", "instruct", "midrange"],
    ),
    # 64GB RAM tier
    "mlx-community/Qwen2.5-32B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Qwen2.5-32B-Instruct-4bit",
        display_name="Qwen 2.5 32B Instruct (4-bit)",
        parameter_count="32B",
        quantization="4-bit",
        min_ram_gb=32,
        recommended_ram_gb=64,
        context_length=32768,
        ram_tier=RAMTier.TIER_64GB,
        description="Large model with excellent reasoning for complex trading analysis.",
        tags=["qwen", "32b", "4bit", "instruct", "large"],
    ),
    # 128GB RAM tier
    "mlx-community/Qwen2.5-72B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Qwen2.5-72B-Instruct-4bit",
        display_name="Qwen 2.5 72B Instruct (4-bit)",
        parameter_count="72B",
        quantization="4-bit",
        min_ram_gb=64,
        recommended_ram_gb=128,
        context_length=32768,
        ram_tier=RAMTier.TIER_128GB,
        description="Flagship Qwen model for maximum quality financial analysis.",
        tags=["qwen", "72b", "4bit", "instruct", "flagship"],
    ),
    "mlx-community/Llama-3.3-70B-Instruct-4bit": ModelInfo(
        model_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
        display_name="Llama 3.3 70B Instruct (4-bit)",
        parameter_count="70B",
        quantization="4-bit",
        min_ram_gb=64,
        recommended_ram_gb=128,
        context_length=131072,
        ram_tier=RAMTier.TIER_128GB,
        description="Meta's flagship Llama model with 128K context window.",
        tags=["llama", "70b", "4bit", "instruct", "flagship"],
    ),
}


class MLXModelManager:
    """
    Singleton manager for MLX model lifecycle.

    Handles downloading, loading, and serving MLX-optimized language models.
    Only one model can be loaded at a time to manage memory on Apple Silicon.

    Usage:
        manager = MLXModelManager.get_instance()
        model, tokenizer = manager.load_model("mlx-community/Qwen2.5-72B-Instruct-4bit")
        # ... use model ...
        manager.unload_model()
    """

    _instance: Optional["MLXModelManager"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "MLXModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        if self._initialized:
            return

        self._cache_dir = Path(
            cache_dir or os.environ.get("MLX_CACHE_DIR", "~/.cache/mlx-models")
        ).expanduser()
        self._default_model = (
            default_model or "mlx-community/Qwen2.5-72B-Instruct-4bit"
        )

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_model_id: Optional[str] = None

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "MLXModelManager initialized",
            extra={
                "cache_dir": str(self._cache_dir),
                "default_model": self._default_model,
                "mlx_available": MLX_AVAILABLE,
            },
        )

        self._initialized = True

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> "MLXModelManager":
        """Get or create the singleton instance."""
        return cls(cache_dir=cache_dir, default_model=default_model)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (mainly for testing)."""
        if cls._instance is not None:
            cls._instance.unload_model()
        cls._instance = None
        cls._initialized = False

    # ------------------------------------------------------------------
    # Model registry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_models(
        ram_tier: Optional[RAMTier] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all registered models, optionally filtered by RAM tier.

        Args:
            ram_tier: Filter by RAM tier (e.g., RAMTier.TIER_24GB).

        Returns:
            List of model info dictionaries.
        """
        models = MODEL_REGISTRY.values()
        if ram_tier is not None:
            models = [m for m in models if m.ram_tier == ram_tier]
        return [m.to_dict() for m in models]

    @staticmethod
    def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info for a specific model.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Model info dict or None if not in registry.
        """
        info = MODEL_REGISTRY.get(model_id)
        return info.to_dict() if info else None

    @staticmethod
    def get_models_for_ram(ram_gb: int) -> List[Dict[str, Any]]:
        """
        Get models suitable for a given amount of RAM.

        Args:
            ram_gb: Available system RAM in gigabytes.

        Returns:
            List of compatible model info dicts, sorted by parameter count.
        """
        compatible = [
            m for m in MODEL_REGISTRY.values() if m.min_ram_gb <= ram_gb
        ]
        compatible.sort(key=lambda m: m.recommended_ram_gb, reverse=True)
        return [m.to_dict() for m in compatible]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_model(self, model_id: str) -> Path:
        """
        Download a model from HuggingFace mlx-community.

        Args:
            model_id: Full HuggingFace model identifier
                      (e.g. 'mlx-community/Qwen2.5-72B-Instruct-4bit').

        Returns:
            Path to the downloaded model directory.

        Raises:
            RuntimeError: If huggingface_hub is not installed.
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is required. Install with: pip install huggingface-hub"
            )

        model_dir = self._cache_dir / model_id.replace("/", "--")

        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Model already cached at {model_dir}")
            return model_dir

        logger.info(f"Downloading model {model_id} to {model_dir} ...")

        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded successfully: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as exc:
            logger.error(f"Failed to download model {model_id}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        Load an MLX model and tokenizer.

        If a different model is already loaded it will be unloaded first.

        Args:
            model_id: HuggingFace model identifier. Defaults to the
                      configured default model.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            RuntimeError: If MLX libraries are not available.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX libraries are required. Install with: pip install mlx mlx-lm"
            )

        model_id = model_id or self._default_model

        # Already loaded
        if self._loaded_model_id == model_id and self._model is not None:
            logger.info(f"Model {model_id} is already loaded")
            return self._model, self._tokenizer

        # Unload previous model if different
        if self._model is not None:
            logger.info(
                f"Unloading current model {self._loaded_model_id} "
                f"before loading {model_id}"
            )
            self.unload_model()

        logger.info(f"Loading MLX model: {model_id}")

        # Check local cache first
        cached_path = self._cache_dir / model_id.replace("/", "--")
        load_path = str(cached_path) if cached_path.exists() else model_id

        try:
            model, tokenizer = mlx_lm.load(load_path)
            self._model = model
            self._tokenizer = tokenizer
            self._loaded_model_id = model_id

            logger.info(f"Model {model_id} loaded successfully")
            return self._model, self._tokenizer

        except Exception as exc:
            logger.error(f"Failed to load model {model_id}: {exc}")
            self._model = None
            self._tokenizer = None
            self._loaded_model_id = None
            raise

    def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self._model is None:
            logger.debug("No model loaded, nothing to unload")
            return

        model_id = self._loaded_model_id
        logger.info(f"Unloading model: {model_id}")

        self._model = None
        self._tokenizer = None
        self._loaded_model_id = None

        # Force garbage collection to reclaim memory
        gc.collect()

        if MLX_AVAILABLE:
            try:
                mx.metal.reset_peak_memory()
            except (AttributeError, Exception):
                pass  # Not all MLX versions support this

        logger.info(f"Model {model_id} unloaded, memory freed")

    def get_loaded_model(self) -> Optional[Tuple[Any, Any]]:
        """
        Return the currently loaded model and tokenizer.

        Returns:
            Tuple of (model, tokenizer) or None if no model is loaded.
        """
        if self._model is None:
            return None
        return self._model, self._tokenizer

    @property
    def loaded_model_id(self) -> Optional[str]:
        """Return the ID of the currently loaded model, or None."""
        return self._loaded_model_id

    @property
    def is_model_loaded(self) -> bool:
        """Check whether a model is currently loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the loaded model.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation kwargs forwarded to mlx_lm.generate.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If no model is loaded or MLX is unavailable.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX libraries are not available")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "No model loaded. Call load_model() first."
            )

        try:
            response = mlx_lm.generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                **kwargs,
            )
            return response
        except Exception as exc:
            logger.error(f"Generation failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current manager status for diagnostics."""
        status: Dict[str, Any] = {
            "mlx_available": MLX_AVAILABLE,
            "hf_hub_available": HF_HUB_AVAILABLE,
            "cache_dir": str(self._cache_dir),
            "default_model": self._default_model,
            "model_loaded": self.is_model_loaded,
            "loaded_model_id": self._loaded_model_id,
            "registered_models": len(MODEL_REGISTRY),
        }

        if MLX_AVAILABLE:
            try:
                status["metal_device"] = str(mx.default_device())
            except Exception:
                status["metal_device"] = "unknown"

        return status
