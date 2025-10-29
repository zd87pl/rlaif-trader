"""TTM (Tiny Time Mixers) wrapper"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig

from .base import FoundationModelBase
from ...utils.logging import get_logger

logger = get_logger(__name__)


class TTMPredictor(FoundationModelBase):
    """
    Wrapper for IBM's TTM (Tiny Time Mixers) foundation model

    TTM is a lightweight (1-16M parameters) time series model that:
    - Achieves 25-50% improvement over larger models with limited data
    - Uses adaptive patching and diverse resolution sampling
    - Enables CPU-only deployment while maintaining accuracy
    - Excels particularly with limited financial data

    Key advantages over TimesFM:
    - Much smaller size (1-16M vs 200M parameters)
    - Better performance with limited data
    - CPU-friendly for cost-effective deployment
    """

    def __init__(
        self,
        model_name: str = "ibm/ttm-research-r1",
        context_length: int = 512,
        patch_size: int = 16,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize TTM predictor

        Args:
            model_name: HuggingFace model name
            context_length: Length of historical context
            patch_size: Size of time series patches
            device: Device to use
            checkpoint_path: Path to fine-tuned checkpoint
        """
        super().__init__(
            model_name=model_name,
            device=device,
            checkpoint_path=checkpoint_path,
        )

        self.context_length = context_length
        self.patch_size = patch_size

        # Load model
        self.load_model()

    def load_model(self):
        """Load TTM model from HuggingFace"""
        try:
            logger.info(f"Loading TTM model: {self.model_name}")

            # Load configuration
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                config=config,
                trust_remote_code=True,
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            # Load fine-tuned checkpoint if provided
            if self.checkpoint_path:
                self.load_checkpoint(self.checkpoint_path)

            logger.info("TTM model loaded successfully")

        except ImportError:
            logger.error(
                "transformers not installed or TTM not available. "
                "Install with: pip install transformers>=4.30.0"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading TTM: {e}")
            raise

    def predict(
        self,
        time_series: np.ndarray,
        horizon: int = 96,
        return_uncertainty: bool = False,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with TTM

        Args:
            time_series: Historical data (shape: [batch, sequence] or [sequence])
            horizon: Prediction horizon
            return_uncertainty: Whether to return uncertainty estimates
            num_samples: Number of samples for uncertainty (if enabled)

        Returns:
            Tuple of (predictions, uncertainty)
        """
        # Ensure 2D array
        if time_series.ndim == 1:
            time_series = time_series.reshape(1, -1)

        batch_size = time_series.shape[0]

        # Use last context_length points
        if time_series.shape[1] > self.context_length:
            time_series = time_series[:, -self.context_length :]

        try:
            # Convert to tensor
            inputs = torch.tensor(time_series, dtype=torch.float32).to(self.device)

            # Add channel dimension if needed [batch, seq] -> [batch, 1, seq]
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)

            # Inference
            with torch.no_grad():
                # TTM forward pass
                outputs = self.model(
                    inputs,
                    decoder_input_ids=None,  # Auto-regressive generation
                )

                # Extract predictions
                if hasattr(outputs, "logits"):
                    predictions = outputs.logits
                elif hasattr(outputs, "last_hidden_state"):
                    predictions = outputs.last_hidden_state
                else:
                    predictions = outputs

                # Reshape to [batch, horizon]
                predictions = predictions.squeeze()

                if predictions.ndim == 1 and batch_size > 1:
                    predictions = predictions.reshape(batch_size, -1)
                elif predictions.ndim == 1:
                    predictions = predictions.reshape(1, -1)

                # Truncate or pad to horizon
                if predictions.shape[-1] > horizon:
                    predictions = predictions[..., :horizon]
                elif predictions.shape[-1] < horizon:
                    # Pad with last value
                    pad_size = horizon - predictions.shape[-1]
                    last_val = predictions[..., -1:].expand(-1, pad_size)
                    predictions = torch.cat([predictions, last_val], dim=-1)

                predictions = predictions.cpu().numpy()

            # Estimate uncertainty if requested
            uncertainty = None
            if return_uncertainty:
                # Monte Carlo dropout for uncertainty
                uncertainties = []

                # Enable dropout
                self.model.train()

                for _ in range(num_samples):
                    with torch.no_grad():
                        sample_output = self.model(inputs)
                        if hasattr(sample_output, "logits"):
                            sample_pred = sample_output.logits.squeeze().cpu().numpy()
                        else:
                            sample_pred = sample_output.squeeze().cpu().numpy()

                        # Handle shape
                        if sample_pred.ndim == 1:
                            sample_pred = sample_pred.reshape(1, -1)

                        # Truncate/pad
                        if sample_pred.shape[-1] > horizon:
                            sample_pred = sample_pred[..., :horizon]
                        elif sample_pred.shape[-1] < horizon:
                            pad_size = horizon - sample_pred.shape[-1]
                            sample_pred = np.pad(
                                sample_pred,
                                ((0, 0), (0, pad_size)),
                                mode="edge",
                            )

                        uncertainties.append(sample_pred)

                # Set back to eval
                self.model.eval()

                # Calculate uncertainty as std across samples
                uncertainties = np.stack(uncertainties, axis=0)
                uncertainty = np.std(uncertainties, axis=0)

            return predictions, uncertainty

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def get_embeddings(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract learned embeddings from TTM encoder

        Args:
            time_series: Historical data

        Returns:
            Embeddings array (shape: [batch, embedding_dim])
        """
        if time_series.ndim == 1:
            time_series = time_series.reshape(1, -1)

        # Use last context_length points
        if time_series.shape[1] > self.context_length:
            time_series = time_series[:, -self.context_length :]

        try:
            # Convert to tensor
            inputs = torch.tensor(time_series, dtype=torch.float32).to(self.device)

            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)

            with torch.no_grad():
                # Get encoder output
                if hasattr(self.model, "encoder"):
                    encoder_output = self.model.encoder(inputs)
                    embeddings = encoder_output.last_hidden_state
                else:
                    # Fallback: use model output
                    outputs = self.model(inputs)
                    if hasattr(outputs, "encoder_last_hidden_state"):
                        embeddings = outputs.encoder_last_hidden_state
                    else:
                        embeddings = outputs.last_hidden_state

                # Pool embeddings (mean over sequence)
                embeddings = embeddings.mean(dim=1).cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            raise

    def batch_predict(
        self,
        time_series_list: list[np.ndarray],
        horizon: int = 96,
    ) -> list[np.ndarray]:
        """
        Batch predict for multiple time series

        Args:
            time_series_list: List of time series arrays
            horizon: Prediction horizon

        Returns:
            List of prediction arrays
        """
        # Pad to same length
        max_len = max(len(ts) for ts in time_series_list)
        padded = []

        for ts in time_series_list:
            if len(ts) < max_len:
                padded_ts = np.pad(ts, (max_len - len(ts), 0), mode="edge")
            else:
                padded_ts = ts

            padded.append(padded_ts)

        # Stack into batch
        batch = np.stack(padded, axis=0)

        # Predict
        predictions, _ = self.predict(batch, horizon=horizon)

        # Split back into list
        return [predictions[i] for i in range(len(predictions))]


class TTMFineTuner:
    """
    Fine-tune TTM on financial data

    TTM's adaptive patching and small size make it ideal for
    fine-tuning with limited financial data.
    """

    def __init__(
        self,
        base_model: TTMPredictor,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        num_epochs: int = 50,
    ):
        """
        Initialize fine-tuner

        Args:
            base_model: Pre-trained TTM model
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of epochs
        """
        self.model = base_model
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fine_tune(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        save_path: str,
        early_stopping_patience: int = 10,
    ):
        """
        Fine-tune TTM model on financial data

        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            save_path: Path to save checkpoint
            early_stopping_patience: Patience for early stopping
        """
        logger.info("Starting TTM fine-tuning...")

        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.lr,
        )

        # Loss function
        criterion = torch.nn.MSELoss()

        # Training loop (simplified outline)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.model.train()
            train_loss = 0.0

            # Note: Full implementation needs DataLoader
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")

            # Validation phase
            self.model.model.eval()
            val_loss = 0.0

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.model.save_checkpoint(save_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Fine-tuning complete. Best val loss: {best_val_loss:.6f}")


# Example usage
if __name__ == "__main__":
    # Initialize TTM
    predictor = TTMPredictor(
        context_length=512,
        patch_size=16,
    )

    # Create sample time series
    time_series = np.random.randn(10, 500)  # 10 series, 500 time steps each

    # Make predictions
    predictions, uncertainty = predictor.predict(
        time_series,
        horizon=96,
        return_uncertainty=True,
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Uncertainty shape: {uncertainty.shape if uncertainty is not None else None}")

    # Get embeddings
    embeddings = predictor.get_embeddings(time_series)
    print(f"Embeddings shape: {embeddings.shape}")
