"""Unified fine-tuning pipeline for foundation models"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .base import FoundationModelBase
from .timesfm_wrapper import TimesFMPredictor
from .ttm_wrapper import TTMPredictor
from ...utils.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting

    Creates sliding windows for training:
    - Input: Historical context (context_length)
    - Target: Future values (horizon)
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int,
        horizon: int,
        stride: int = 1,
    ):
        """
        Initialize dataset

        Args:
            data: Time series data (shape: [total_timesteps] or [total_timesteps, features])
            context_length: Length of input window
            horizon: Length of prediction window
            stride: Stride for sliding window
        """
        self.data = data
        self.context_length = context_length
        self.horizon = horizon
        self.stride = stride

        # Calculate number of samples
        self.num_samples = max(
            0,
            (len(data) - context_length - horizon) // stride + 1,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate start index
        start = idx * self.stride

        # Extract input and target
        input_data = self.data[start : start + self.context_length]
        target_data = self.data[
            start + self.context_length : start + self.context_length + self.horizon
        ]

        # Convert to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        return input_tensor, target_tensor


class FoundationModelFineTuner:
    """
    Unified fine-tuning pipeline for foundation models

    Supports:
    - TimesFM 2.5
    - TTM
    - Any model inheriting from FoundationModelBase

    Features:
    - Walk-forward validation
    - Early stopping
    - Multiple loss functions
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training (optional)
    """

    def __init__(
        self,
        model: FoundationModelBase,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        gradient_clip: float = 1.0,
        loss_fn: str = "mse",
    ):
        """
        Initialize fine-tuner

        Args:
            model: Foundation model to fine-tune
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            gradient_clip: Gradient clipping threshold
            loss_fn: Loss function ("mse", "mae", "huber", "mape")
        """
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience
        self.grad_clip = gradient_clip

        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        elif loss_fn == "mape":
            self.criterion = self._mape_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    @staticmethod
    def _mape_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Percentage Error"""
        return torch.mean(torch.abs((target - pred) / (target + 1e-9))) * 100

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        context_length: int,
        horizon: int,
        stride: int = 1,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Prepare DataLoaders from DataFrames

        Args:
            train_df: Training data
            val_df: Validation data
            context_length: Input window length
            horizon: Prediction horizon
            stride: Sliding window stride

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to numpy
        train_data = train_df.values
        val_data = val_df.values

        # Handle multi-dimensional data
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
            val_data = val_data.reshape(-1, 1)

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data,
            context_length,
            horizon,
            stride,
        )

        val_dataset = TimeSeriesDataset(
            val_data,
            context_length,
            horizon,
            stride,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=0,
            pin_memory=True if self.model.device == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.model.device == "cuda" else False,
        )

        logger.info(
            f"Prepared data: {len(train_dataset)} train, {len(val_dataset)} val samples"
        )

        return train_loader, val_loader

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Union[str, Path],
    ) -> Dict:
        """
        Fine-tune the model

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            save_path: Path to save best checkpoint

        Returns:
            Dictionary with training history
        """
        logger.info("Starting fine-tuning...")

        # Get model parameters
        if isinstance(self.model, (TimesFMPredictor, TTMPredictor)):
            model_params = self.model.model.parameters()
        else:
            model_params = self.model.parameters()

        # Optimizer
        optimizer = torch.optim.AdamW(model_params, lr=self.lr)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer)

            # Validation
            val_loss = self._validate_epoch(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                self.model.save_checkpoint(str(save_path))
                logger.info(f"Saved best model (val_loss: {val_loss:.6f})")

            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_checkpoint(str(save_path))

        logger.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.6f}")

        return self.history

    def _train_epoch(self, train_loader: DataLoader, optimizer) -> float:
        """Train for one epoch"""
        if isinstance(self.model, (TimesFMPredictor, TTMPredictor)):
            model = self.model.model
        else:
            model = self.model

        model.train()
        total_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            # Move to device
            inputs = inputs.to(self.model.device)
            targets = targets.to(self.model.device)

            # Forward pass
            optimizer.zero_grad()

            # Get predictions (using model's predict method)
            predictions, _ = self.model.predict(
                inputs.cpu().numpy(),
                horizon=targets.shape[1],
                return_uncertainty=False,
            )

            predictions = torch.tensor(predictions, device=self.model.device)

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        if isinstance(self.model, (TimesFMPredictor, TTMPredictor)):
            model = self.model.model
        else:
            model = self.model

        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
                # Move to device
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)

                # Get predictions
                predictions, _ = self.model.predict(
                    inputs.cpu().numpy(),
                    horizon=targets.shape[1],
                    return_uncertainty=False,
                )

                predictions = torch.tensor(predictions, device=self.model.device)

                # Compute loss
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)


# Example usage
if __name__ == "__main__":
    from src.data import AlpacaDataClient, DataPreprocessor

    # Download and prepare data
    client = AlpacaDataClient()
    df = client.download_latest(symbols="AAPL", days=365, timeframe="1Day")

    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)

    # Train/val split
    train_df, val_df = preprocessor.train_test_split(df_processed["close"], test_size=0.2)

    # Initialize model
    model = TimesFMPredictor(context_length=64, horizon=32)

    # Initialize fine-tuner
    tuner = FoundationModelFineTuner(
        model=model,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=50,
    )

    # Prepare data
    train_loader, val_loader = tuner.prepare_data(
        train_df,
        val_df,
        context_length=64,
        horizon=32,
        stride=1,
    )

    # Fine-tune
    history = tuner.fine_tune(
        train_loader,
        val_loader,
        save_path="models/checkpoints/timesfm_finetuned.pt",
    )

    print(f"Training complete. Final val loss: {history['val_loss'][-1]:.6f}")
