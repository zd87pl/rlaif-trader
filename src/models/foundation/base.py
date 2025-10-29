"""Base class for foundation models"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ...utils.logging import get_logger

logger = get_logger(__name__)


class FoundationModelBase(ABC):
    """
    Abstract base class for time series foundation models

    All foundation models (TimesFM, TTM, etc.) should inherit from this
    and implement the abstract methods.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize foundation model

        Args:
            model_name: Name/identifier of the model
            device: Device to use (cuda/cpu/mps)
            checkpoint_path: Path to model checkpoint
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Initializing {model_name} on {device}")

    @abstractmethod
    def load_model(self):
        """Load the foundation model"""
        pass

    @abstractmethod
    def predict(
        self,
        time_series: np.ndarray,
        horizon: int,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions

        Args:
            time_series: Historical time series data (shape: [batch, sequence, features])
            horizon: Number of steps to predict into the future
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainty) where uncertainty is None if not requested
        """
        pass

    @abstractmethod
    def get_embeddings(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract learned embeddings from the model

        Args:
            time_series: Historical time series data

        Returns:
            Embeddings array
        """
        pass

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess DataFrame into model input format

        Args:
            df: DataFrame with time series data

        Returns:
            Preprocessed numpy array
        """
        # Default implementation: convert to numpy and normalize
        values = df.values

        # Handle NaN values
        if np.any(np.isnan(values)):
            logger.warning("NaN values detected, forward filling")
            df_filled = df.ffill().bfill()
            values = df_filled.values

        return values

    def postprocess(
        self,
        predictions: np.ndarray,
        original_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Postprocess model predictions into DataFrame format

        Args:
            predictions: Model predictions
            original_df: Original DataFrame (for index/columns)

        Returns:
            DataFrame with predictions
        """
        # Create future index
        last_date = original_df.index[-1]
        freq = pd.infer_freq(original_df.index)

        if freq is None:
            logger.warning("Could not infer frequency, using '1D'")
            freq = "1D"

        future_index = pd.date_range(
            start=last_date,
            periods=len(predictions) + 1,
            freq=freq,
        )[1:]  # Exclude start date

        # Create DataFrame
        pred_df = pd.DataFrame(
            predictions,
            index=future_index,
            columns=[f"pred_{col}" for col in original_df.columns],
        )

        return pred_df

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        if hasattr(self, "model"):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "model_name": self.model_name,
                },
                path,
            )
            logger.info(f"Checkpoint saved to {path}")
        else:
            logger.warning("No model to save")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if hasattr(self, "model"):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Checkpoint loaded from {path}")
        else:
            logger.warning("Model not initialized")
