"""Kronos Financial Foundation Model wrapper

Kronos is the first foundation model trained on 12 billion K-line (candlestick)
records from 45 global exchanges. It features a specialized tokenizer for
OHLCVA (Open, High, Low, Close, Volume, Amount) data with hierarchical discrete
tokens optimized for financial candlestick patterns.

Key advantages over generic time series foundation models:
- Native OHLCVA tokenization (not just close prices)
- Trained exclusively on financial candlestick data (12B K-lines)
- 93% improvement in price forecasting RankIC vs best TSFM baselines
- Hierarchical discrete tokens capture candlestick pattern semantics

Reference: https://github.com/Donaghue/Kronos
Available on HuggingFace: Donaghue/kronos-base, kronos-small, kronos-large
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .base import FoundationModelBase
from ...utils.logging import get_logger

logger = get_logger(__name__)

# Model size configurations
KRONOS_MODEL_CONFIGS = {
    "small": {
        "repo_id": "Donaghue/kronos-small",
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "description": "Kronos-Small (~50M params) — fast inference, suitable for real-time",
    },
    "base": {
        "repo_id": "Donaghue/kronos-base",
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "description": "Kronos-Base (~150M params) — balanced speed/accuracy",
    },
    "large": {
        "repo_id": "Donaghue/kronos-large",
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "description": "Kronos-Large (~400M params) — highest accuracy",
    },
}

# Standard OHLCVA column names (case-insensitive matching)
OHLCVA_COLUMNS = {
    "open": ["open", "Open", "OPEN", "o"],
    "high": ["high", "High", "HIGH", "h"],
    "low": ["low", "Low", "LOW", "l"],
    "close": ["close", "Close", "CLOSE", "c", "adj_close", "Adj Close"],
    "volume": ["volume", "Volume", "VOLUME", "v", "vol"],
    "amount": ["amount", "Amount", "AMOUNT", "a", "turnover", "Turnover"],
}

# Kronos may not be installable via pip yet; guard all imports
_KRONOS_AVAILABLE = False
_KRONOS_IMPORT_ERROR = None

try:
    import kronos as kronos_lib

    _KRONOS_AVAILABLE = True
except ImportError as e:
    _KRONOS_IMPORT_ERROR = e
    kronos_lib = None


def _resolve_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Resolve a canonical OHLCVA field name to an actual DataFrame column."""
    candidates = OHLCVA_COLUMNS.get(field, [])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


class KronosPredictor(FoundationModelBase):
    """
    Wrapper for the Kronos financial foundation model.

    Kronos is purpose-built for financial candlestick forecasting, trained on
    12 billion K-line records from 45 global exchanges. Unlike generic time
    series models that treat price as a single scalar, Kronos natively
    tokenizes OHLCVA (Open, High, Low, Close, Volume, Amount) tuples using
    a hierarchical discrete tokenizer.

    Features:
        - Native OHLCVA input support (the primary advantage)
        - Close-only fallback for compatibility with generic pipelines
        - Multi-horizon forecasting (1d, 1w, 1m, 3m)
        - Uncertainty estimation via Monte Carlo dropout
        - Embedding extraction from the encoder for downstream tasks
        - Graceful degradation when kronos package is not installed

    Usage::

        predictor = KronosPredictor(model_size="base", context_length=512)
        predictions, uncertainty = predictor.predict(
            time_series, horizon=64, return_uncertainty=True
        )

    If the ``kronos`` package is not installed, the wrapper operates in mock
    mode — returning zero-filled predictions so that the rest of the trading
    system does not break.
    """

    # Trading horizon presets (business days)
    TRADING_HORIZONS: Dict[str, int] = {
        "1d": 1,
        "1w": 5,
        "1m": 21,
        "3m": 63,
    }

    def __init__(
        self,
        context_length: int = 512,
        horizon: int = 64,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_size: str = "base",
        mc_dropout_samples: int = 30,
    ):
        """
        Initialize KronosPredictor.

        Args:
            context_length: Number of historical K-lines to feed the model.
                Kronos supports up to 2048 but 512 is the recommended default.
            horizon: Default prediction horizon (number of future K-lines).
            device: Compute device — 'cuda', 'cpu', or 'mps'. Auto-detected
                if ``None``.
            checkpoint_path: Path to a local fine-tuned checkpoint. When
                ``None`` the model is loaded from HuggingFace.
            model_size: One of 'small', 'base', 'large'. Controls the
                number of parameters and inference cost.
            mc_dropout_samples: Number of forward passes for Monte Carlo
                dropout uncertainty estimation.

        Raises:
            ValueError: If ``model_size`` is not in {small, base, large}.
        """
        if model_size not in KRONOS_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from: {list(KRONOS_MODEL_CONFIGS.keys())}"
            )

        self._config = KRONOS_MODEL_CONFIGS[model_size]
        self.model_size = model_size
        self.context_length = context_length
        self.default_horizon = horizon
        self.mc_dropout_samples = mc_dropout_samples
        self._mock_mode = False

        super().__init__(
            model_name=self._config["repo_id"],
            device=device,
            checkpoint_path=checkpoint_path,
        )

        self.load_model()

    # ------------------------------------------------------------------
    # Core interface (FoundationModelBase)
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the Kronos model from HuggingFace or a local checkpoint.

        If the ``kronos`` package is not installed, the wrapper enters mock
        mode and logs installation instructions. In mock mode all predictions
        return zeros so dependent components remain functional.
        """
        if not _KRONOS_AVAILABLE:
            logger.warning(
                "Kronos package not installed. Entering mock mode.\n"
                "To install Kronos, run one of:\n"
                "  pip install kronos-finance\n"
                "  pip install git+https://github.com/Donaghue/Kronos.git\n"
                "Or clone and install locally:\n"
                "  git clone https://github.com/Donaghue/Kronos.git\n"
                "  cd Kronos && pip install -e .\n"
                f"Original import error: {_KRONOS_IMPORT_ERROR}"
            )
            self._mock_mode = True
            self.model = None
            return

        try:
            logger.info(
                f"Loading Kronos model: {self.model_name} "
                f"({self._config['description']})"
            )

            if self.checkpoint_path:
                # Load from local checkpoint
                logger.info(
                    f"Loading from local checkpoint: {self.checkpoint_path}"
                )
                self.model = kronos_lib.KronosModel.from_pretrained(
                    self.checkpoint_path,
                    device=self.device,
                )
            else:
                # Load from HuggingFace hub
                self.model = kronos_lib.KronosModel.from_pretrained(
                    self._config["repo_id"],
                    device=self.device,
                )

            # Configure model for inference
            self.model.eval()

            # Store the tokenizer (Kronos ships its own OHLCVA tokenizer)
            self.tokenizer = self.model.tokenizer

            logger.info(
                f"Kronos model loaded successfully on {self.device} "
                f"(context_length={self.context_length}, "
                f"hidden_dim={self._config['hidden_dim']})"
            )

        except Exception as e:
            logger.error(
                f"Failed to load Kronos model: {e}. "
                "Falling back to mock mode."
            )
            self._mock_mode = True
            self.model = None

    def predict(
        self,
        time_series: np.ndarray,
        horizon: Optional[int] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate forecasts from historical time series data.

        The input can be either:
        - 1-D or 2-D with a single feature (close-price only), or
        - 2-D with 4–6 features interpreted as OHLC(VA) columns.

        When OHLCVA columns are present, Kronos uses its native candlestick
        tokenizer for significantly better accuracy.

        Args:
            time_series: Historical data. Accepted shapes:
                - ``(sequence,)`` — single univariate series
                - ``(batch, sequence)`` — batch of univariate series
                - ``(batch, sequence, features)`` — batch of multivariate
                  series (features ordered as O, H, L, C, V, A)
            horizon: Steps ahead to forecast. Uses ``self.default_horizon``
                when ``None``.
            return_uncertainty: If ``True``, estimate prediction uncertainty
                via Monte Carlo dropout.

        Returns:
            Tuple of ``(predictions, uncertainty)`` where:
                - predictions: ``np.ndarray`` of shape ``(batch, horizon)``
                - uncertainty: ``np.ndarray`` of same shape if requested,
                  else ``None``
        """
        if horizon is None:
            horizon = self.default_horizon

        # Normalise to 3-D: (batch, sequence, features)
        time_series = self._ensure_3d(time_series)

        # Truncate to context window
        if time_series.shape[1] > self.context_length:
            time_series = time_series[:, -self.context_length :, :]

        # ---- Mock mode ----
        if self._mock_mode:
            logger.warning(
                "Kronos is in mock mode — returning zero predictions. "
                "Install the kronos package for real forecasts."
            )
            batch_size = time_series.shape[0]
            preds = np.zeros((batch_size, horizon), dtype=np.float32)
            unc = (
                np.ones((batch_size, horizon), dtype=np.float32) * 1e-3
                if return_uncertainty
                else None
            )
            return preds, unc

        # ---- Real inference ----
        try:
            input_tensor = torch.tensor(
                time_series, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                output = self.model.forecast(
                    input_tensor,
                    horizon=horizon,
                )

            # Extract point forecast (close-price channel)
            predictions = output.predictions.cpu().numpy()

            # Uncertainty via Monte Carlo dropout
            uncertainty = None
            if return_uncertainty:
                uncertainty = self._estimate_uncertainty(
                    input_tensor, horizon
                )

            return predictions, uncertainty

        except Exception as e:
            logger.error(f"Kronos prediction failed: {e}")
            raise

    def get_embeddings(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract learned representations from Kronos's encoder.

        These embeddings capture the semantic structure of candlestick
        patterns and can be used as features for downstream classifiers
        (e.g., regime detection, anomaly scoring).

        Args:
            time_series: Historical data — same shape conventions as
                :meth:`predict`.

        Returns:
            Embeddings array of shape ``(batch, hidden_dim)`` where
            ``hidden_dim`` depends on the model size.
        """
        time_series = self._ensure_3d(time_series)

        if time_series.shape[1] > self.context_length:
            time_series = time_series[:, -self.context_length :, :]

        # ---- Mock mode ----
        if self._mock_mode:
            logger.warning(
                "Kronos is in mock mode — returning random embeddings."
            )
            batch_size = time_series.shape[0]
            hidden_dim = self._config["hidden_dim"]
            return np.zeros((batch_size, hidden_dim), dtype=np.float32)

        # ---- Real embedding extraction ----
        try:
            input_tensor = torch.tensor(
                time_series, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                encoder_output = self.model.encode(input_tensor)

            # Pool over the sequence dimension (mean pooling)
            # encoder_output shape: (batch, seq_len, hidden_dim)
            embeddings = encoder_output.mean(dim=1).cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Kronos embedding extraction failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Kronos-specific methods
    # ------------------------------------------------------------------

    def preprocess_ohlcva(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess a DataFrame with OHLCV(A) columns into the format
        expected by Kronos's native candlestick tokenizer.

        This is Kronos's key advantage: instead of collapsing candlestick
        data into a single close-price scalar, the model tokenizes the full
        OHLCVA tuple as a single hierarchical token.

        The method automatically resolves common column name variants
        (e.g., 'Close', 'close', 'CLOSE', 'Adj Close').

        Args:
            df: DataFrame with at least Open, High, Low, Close columns.
                Volume and Amount are optional but recommended.

        Returns:
            ``np.ndarray`` of shape ``(1, sequence_length, n_channels)``
            where ``n_channels`` is 4 (OHLC), 5 (OHLCV), or 6 (OHLCVA)
            depending on available columns.

        Raises:
            ValueError: If required OHLC columns are missing.
        """
        # Resolve column names
        resolved: Dict[str, Optional[str]] = {}
        for field in ["open", "high", "low", "close", "volume", "amount"]:
            resolved[field] = _resolve_column(df, field)

        # Validate required columns
        missing = [
            f for f in ["open", "high", "low", "close"] if resolved[f] is None
        ]
        if missing:
            raise ValueError(
                f"Missing required OHLC columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Build ordered channel list
        channels = []
        channel_names = []
        for field in ["open", "high", "low", "close", "volume", "amount"]:
            col = resolved[field]
            if col is not None:
                channels.append(df[col].values.astype(np.float64))
                channel_names.append(field)

        logger.info(
            f"Preprocessed OHLCVA with channels: {channel_names} "
            f"({len(df)} timesteps)"
        )

        # Stack to (sequence, n_channels) then add batch dim
        data = np.stack(channels, axis=-1)

        # Handle NaN values via forward-fill then back-fill
        if np.any(np.isnan(data)):
            logger.warning("NaN values detected in OHLCVA data, forward filling")
            df_temp = pd.DataFrame(data, columns=channel_names)
            df_temp = df_temp.ffill().bfill()
            data = df_temp.values

        # Normalise: per-channel z-score over the context window
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
        data = (data - mean) / std

        return data[np.newaxis, :, :]  # (1, seq, channels)

    def forecast_multi_horizon(
        self,
        time_series: np.ndarray,
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Generate forecasts at multiple trading-relevant horizons.

        This is a convenience method for the common pattern of producing
        1-day, 1-week, 1-month, and 3-month ahead forecasts in a single
        call.

        Args:
            time_series: Historical data (same conventions as :meth:`predict`).
            horizons: List of horizons in business days. Defaults to
                ``[1, 5, 21, 63]`` (1d, 1w, 1m, 3m).

        Returns:
            Dictionary mapping horizon labels to ``(predictions, uncertainty)``
            tuples. Labels follow the pattern ``"Xd"`` for X business days.

        Example::

            results = predictor.forecast_multi_horizon(price_data)
            preds_1w, unc_1w = results["5d"]
        """
        if horizons is None:
            horizons = [1, 5, 21, 63]

        results: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

        # Run the longest horizon first (most compute) then slice for
        # shorter ones — this is more efficient than N separate calls.
        max_horizon = max(horizons)

        logger.info(
            f"Multi-horizon forecast: horizons={horizons}, "
            f"max={max_horizon}"
        )

        predictions, uncertainty = self.predict(
            time_series,
            horizon=max_horizon,
            return_uncertainty=True,
        )

        for h in sorted(horizons):
            label = f"{h}d"
            h_preds = predictions[:, :h]
            h_unc = uncertainty[:, :h] if uncertainty is not None else None
            results[label] = (h_preds, h_unc)

        return results

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        horizon: int = 64,
        return_uncertainty: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        High-level predict that auto-detects OHLCVA columns in a DataFrame.

        If the DataFrame contains OHLC columns, uses the native OHLCVA
        tokenizer path. Otherwise falls back to close-only / generic
        preprocessing.

        Args:
            df: Input DataFrame with price data.
            horizon: Prediction horizon.
            return_uncertainty: Whether to return uncertainty estimates.

        Returns:
            Tuple of (predictions, uncertainty).
        """
        # Try OHLCVA path first
        has_ohlc = all(
            _resolve_column(df, f) is not None
            for f in ["open", "high", "low", "close"]
        )

        if has_ohlc:
            logger.info("OHLC columns detected — using native OHLCVA path")
            data = self.preprocess_ohlcva(df)
        else:
            logger.info(
                "OHLC columns not found — falling back to generic preprocess"
            )
            data = self.preprocess(df)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]

        return self.predict(data, horizon=horizon, return_uncertainty=return_uncertainty)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        """Normalise input to (batch, sequence, features)."""
        if arr.ndim == 1:
            return arr.reshape(1, -1, 1)
        if arr.ndim == 2:
            return arr[:, :, np.newaxis]
        if arr.ndim == 3:
            return arr
        raise ValueError(
            f"Expected 1-D, 2-D, or 3-D array, got shape {arr.shape}"
        )

    def _estimate_uncertainty(
        self,
        input_tensor: torch.Tensor,
        horizon: int,
    ) -> np.ndarray:
        """
        Estimate predictive uncertainty via Monte Carlo dropout.

        Performs ``self.mc_dropout_samples`` stochastic forward passes with
        dropout enabled and returns the standard deviation across samples
        as the uncertainty estimate.

        Args:
            input_tensor: Prepared input tensor on the correct device.
            horizon: Forecast horizon.

        Returns:
            Uncertainty array of shape ``(batch, horizon)``.
        """
        if self._mock_mode or self.model is None:
            batch_size = input_tensor.shape[0]
            return np.ones((batch_size, horizon), dtype=np.float32) * 1e-3

        # Enable dropout for MC sampling
        self.model.train()

        samples = []
        try:
            for _ in range(self.mc_dropout_samples):
                with torch.no_grad():
                    output = self.model.forecast(input_tensor, horizon=horizon)
                samples.append(output.predictions.cpu().numpy())
        finally:
            # Always restore eval mode
            self.model.eval()

        # Stack: (n_samples, batch, horizon)
        samples = np.stack(samples, axis=0)
        uncertainty = np.std(samples, axis=0)

        return uncertainty

    def __repr__(self) -> str:
        status = "mock" if self._mock_mode else "loaded"
        return (
            f"KronosPredictor("
            f"model_size='{self.model_size}', "
            f"context_length={self.context_length}, "
            f"horizon={self.default_horizon}, "
            f"device='{self.device}', "
            f"status='{status}')"
        )


# ------------------------------------------------------------------
# Example / smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Initializing KronosPredictor (will use mock if not installed)...")
    predictor = KronosPredictor(
        model_size="base",
        context_length=512,
        horizon=64,
    )
    print(repr(predictor))

    # Univariate test
    ts = np.random.randn(2, 200)
    preds, unc = predictor.predict(ts, horizon=21, return_uncertainty=True)
    print(f"Predictions shape: {preds.shape}")
    print(f"Uncertainty shape:  {unc.shape if unc is not None else None}")

    # OHLCVA DataFrame test
    n = 300
    df = pd.DataFrame(
        {
            "Open": np.cumsum(np.random.randn(n)) + 100,
            "High": np.cumsum(np.random.randn(n)) + 101,
            "Low": np.cumsum(np.random.randn(n)) + 99,
            "Close": np.cumsum(np.random.randn(n)) + 100,
            "Volume": np.abs(np.random.randn(n)) * 1e6,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    ohlcva = predictor.preprocess_ohlcva(df)
    print(f"OHLCVA preprocessed shape: {ohlcva.shape}")

    # Multi-horizon test
    results = predictor.forecast_multi_horizon(ts)
    for label, (p, u) in results.items():
        print(f"  {label}: preds={p.shape}, unc={u.shape if u else None}")

    # DataFrame predict
    preds_df, _ = predictor.predict_from_dataframe(df, horizon=21)
    print(f"DataFrame predict shape: {preds_df.shape}")
