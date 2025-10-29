"""
Reward Model for RLAIF

Learns to predict trading decision quality from preference pairs.

Architecture:
- Numerical encoder: For market data and features
- Text encoder: For agent analysis and reasoning (uses sentence transformers)
- Fusion: Combines numerical and text representations
- Output: Single score predicting decision quality

Training:
- Pairwise ranking loss (Bradley-Terry model)
- Given preference (chosen, rejected), maximize P(chosen > rejected)
- Loss = -log(sigmoid(score_chosen - score_rejected))

This model provides rewards for RL fine-tuning of Claude agents.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .preference_generator import PreferencePair, TradingDecision
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DecisionDataset(Dataset):
    """
    Dataset for preference pairs

    Each sample is a preference pair: (chosen_features, rejected_features, margin)
    """

    def __init__(
        self,
        preferences: List[PreferencePair],
        text_encoder: SentenceTransformer,
        feature_keys: Optional[List[str]] = None,
    ):
        """
        Initialize dataset

        Args:
            preferences: List of preference pairs
            text_encoder: Sentence transformer for encoding text
            feature_keys: Which numerical features to use (None = all)
        """
        self.preferences = preferences
        self.text_encoder = text_encoder
        self.feature_keys = feature_keys

        # Extract features
        self.chosen_features = []
        self.rejected_features = []
        self.margins = []

        logger.info("Encoding preference pairs...")
        for pref in tqdm(preferences, desc="Encoding"):
            chosen_feat = self._extract_features(pref.chosen_decision)
            rejected_feat = self._extract_features(pref.rejected_decision)

            self.chosen_features.append(chosen_feat)
            self.rejected_features.append(rejected_feat)
            self.margins.append(pref.margin)

    def _extract_features(self, decision: TradingDecision) -> Dict[str, torch.Tensor]:
        """
        Extract features from a trading decision

        Returns:
            Dict with:
                - numerical: Tensor of numerical features
                - text_embedding: Tensor from text encoder
        """
        # Numerical features
        numerical = []

        # Agent score and confidence
        numerical.append(decision.agent_response.score)
        numerical.append(decision.agent_response.confidence)

        # Position size (normalized)
        numerical.append(decision.position_size / 100.0)

        # Market data features (extract key values)
        if "current_price" in decision.market_data:
            numerical.append(decision.market_data["current_price"] / 1000.0)  # Normalize

        # Technical indicators from features dict
        if decision.features:
            for key, value in decision.features.items():
                if self.feature_keys is None or key in self.feature_keys:
                    if isinstance(value, (int, float)):
                        numerical.append(float(value))

        # Pad or truncate to fixed size (e.g., 100 features)
        target_size = 100
        if len(numerical) < target_size:
            numerical.extend([0.0] * (target_size - len(numerical)))
        else:
            numerical = numerical[:target_size]

        numerical_tensor = torch.tensor(numerical, dtype=torch.float32)

        # Text encoding
        # Combine analysis and reasoning into one text
        text = f"{decision.agent_response.analysis}\n\nReasoning:\n"
        text += "\n".join(decision.agent_response.reasoning[:5])  # Top 5 reasoning steps

        # Encode with sentence transformer
        text_embedding = self.text_encoder.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        return {
            "numerical": numerical_tensor,
            "text_embedding": text_embedding,
        }

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "chosen_numerical": self.chosen_features[idx]["numerical"],
            "chosen_text": self.chosen_features[idx]["text_embedding"],
            "rejected_numerical": self.rejected_features[idx]["numerical"],
            "rejected_text": self.rejected_features[idx]["text_embedding"],
            "margin": torch.tensor(self.margins[idx], dtype=torch.float32),
        }


class RewardModel(nn.Module):
    """
    Neural network reward model

    Predicts quality score for a trading decision based on:
    - Numerical features (market data, indicators, agent scores)
    - Text features (agent analysis and reasoning)

    Architecture:
    - Numerical encoder: MLP
    - Text encoder: Pre-trained sentence transformer (frozen or fine-tuned)
    - Fusion: Concatenate + MLP
    - Output: Single scalar score
    """

    def __init__(
        self,
        numerical_dim: int = 100,
        text_dim: int = 768,  # Sentence transformer default
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        """
        Initialize reward model

        Args:
            numerical_dim: Dimension of numerical features
            text_dim: Dimension of text embeddings
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()

        self.numerical_dim = numerical_dim
        self.text_dim = text_dim

        # Numerical encoder
        numerical_layers = []
        in_dim = numerical_dim
        for h_dim in hidden_dims:
            numerical_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        self.numerical_encoder = nn.Sequential(*numerical_layers)
        numerical_output_dim = hidden_dims[-1]

        # Text projection (project text embedding to smaller dimension)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        text_output_dim = 128

        # Fusion layer
        fusion_input_dim = numerical_output_dim + text_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Output: single score
        )

        logger.info(
            f"RewardModel initialized: num_dim={numerical_dim}, text_dim={text_dim}, "
            f"hidden={hidden_dims}"
        )

    def forward(
        self,
        numerical: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            numerical: [batch_size, numerical_dim]
            text_embedding: [batch_size, text_dim]

        Returns:
            scores: [batch_size, 1]
        """
        # Encode numerical
        numerical_repr = self.numerical_encoder(numerical)  # [batch, hidden_dims[-1]]

        # Project text
        text_repr = self.text_projection(text_embedding)  # [batch, 128]

        # Fuse
        combined = torch.cat([numerical_repr, text_repr], dim=1)
        scores = self.fusion(combined)  # [batch, 1]

        return scores.squeeze(-1)  # [batch]


class RewardModelTrainer:
    """
    Trainer for reward model using preference pairs

    Uses Bradley-Terry pairwise ranking loss
    """

    def __init__(
        self,
        model: RewardModel,
        text_encoder: SentenceTransformer,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer

        Args:
            model: Reward model
            text_encoder: Sentence transformer for encoding
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model.to(device)
        self.text_encoder = text_encoder
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        logger.info(f"RewardModelTrainer initialized on {device}")

    def train(
        self,
        preferences: List[PreferencePair],
        val_preferences: Optional[List[PreferencePair]] = None,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """
        Train reward model

        Args:
            preferences: Training preference pairs
            val_preferences: Validation preference pairs
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        # Create datasets
        train_dataset = DecisionDataset(preferences, self.text_encoder)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_preferences:
            val_dataset = DecisionDataset(val_preferences, self.text_encoder)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Training history
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        logger.info(f"Training reward model for {epochs} epochs")

        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.2%}"
                )

                self.scheduler.step(val_loss)
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return history

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move to device
            chosen_num = batch["chosen_numerical"].to(self.device)
            chosen_text = batch["chosen_text"].to(self.device)
            rejected_num = batch["rejected_numerical"].to(self.device)
            rejected_text = batch["rejected_text"].to(self.device)

            # Forward pass
            chosen_scores = self.model(chosen_num, chosen_text)
            rejected_scores = self.model(rejected_num, rejected_text)

            # Bradley-Terry loss: -log(sigmoid(chosen - rejected))
            # This maximizes P(chosen > rejected)
            loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                chosen_num = batch["chosen_numerical"].to(self.device)
                chosen_text = batch["chosen_text"].to(self.device)
                rejected_num = batch["rejected_numerical"].to(self.device)
                rejected_text = batch["rejected_text"].to(self.device)

                # Forward pass
                chosen_scores = self.model(chosen_num, chosen_text)
                rejected_scores = self.model(rejected_num, rejected_text)

                # Loss
                loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
                total_loss += loss.item()

                # Accuracy (how often model ranks chosen > rejected)
                correct += (chosen_scores > rejected_scores).sum().item()
                total += len(chosen_scores)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def save(self, path: Path) -> None:
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        logger.info(f"Saved reward model to {path}")

    def load(self, path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded reward model from {path}")


def create_reward_model(
    text_encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[RewardModel, SentenceTransformer]:
    """
    Factory function to create reward model and text encoder

    Args:
        text_encoder_name: Sentence transformer model name
        device: Device

    Returns:
        (reward_model, text_encoder)
    """
    # Load text encoder
    logger.info(f"Loading text encoder: {text_encoder_name}")
    text_encoder = SentenceTransformer(text_encoder_name, device=device)
    text_dim = text_encoder.get_sentence_embedding_dimension()

    # Create reward model
    reward_model = RewardModel(
        numerical_dim=100,
        text_dim=text_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.1,
    )

    logger.info(
        f"Reward model created: {sum(p.numel() for p in reward_model.parameters()):,} parameters"
    )

    return reward_model, text_encoder
