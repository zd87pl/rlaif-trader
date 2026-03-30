"""
Conformal prediction module for calibrated uncertainty quantification on trading signals.

Provides distribution-free prediction intervals with guaranteed coverage,
based on CPPS (Feb 2025) and TCP (Jul 2025) research.

Classes:
    ConformalPredictor          - Split/adaptive conformal prediction intervals
    TemporalConformalPredictor  - Time-series aware conformal (TCP, Jul 2025)
    SignalConformalWrapper      - Wraps any trading signal predictor with intervals
    PortfolioConformalPredictor - Portfolio-level conformal risk (CPPS paper)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from scipy import stats as scipy_stats

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol for pluggable predictors
# ---------------------------------------------------------------------------

@runtime_checkable
class Predictor(Protocol):
    """Any object that exposes a .predict() method."""

    def predict(self, X: Any) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# ConformalPredictor
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """Distribution-free prediction intervals with guaranteed coverage.

    Based on CPPS (Feb 2025) and TCP (Jul 2025) research.

    Parameters
    ----------
    alpha : float
        Miscoverage level.  ``alpha = 0.1`` → 90 % coverage guarantee.
    method : str
        ``'split'`` – split conformal prediction (fastest, simplest).
        ``'adaptive'`` – locally-adaptive intervals that scale with
        predicted uncertainty.
        ``'temporal'`` – alias that delegates to :class:`TemporalConformalPredictor`.
    """

    VALID_METHODS = ("split", "adaptive", "temporal")

    def __init__(self, alpha: float = 0.1, method: str = "split") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got '{method}'"
            )
        self.alpha = alpha
        self.method = method
        self.coverage_guarantee = 1.0 - alpha

        # Calibration state
        self._scores: Optional[np.ndarray] = None
        self._quantile: Optional[float] = None
        self._mad_scale: Optional[np.ndarray] = None  # for adaptive
        self._calibrated = False

        logger.info(
            "ConformalPredictor created: alpha=%.3f  method=%s  coverage=%.1f%%",
            alpha,
            method,
            self.coverage_guarantee * 100,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """Compute nonconformity scores and the calibration quantile.

        Parameters
        ----------
        predictions : array-like, shape (n,)
            Point predictions on the calibration set.
        actuals : array-like, shape (n,)
            True values on the calibration set.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()

        if predictions.shape[0] != actuals.shape[0]:
            raise ValueError("predictions and actuals must have equal length")
        if predictions.shape[0] < 2:
            raise ValueError("Need at least 2 calibration samples")

        n = predictions.shape[0]
        residuals = np.abs(actuals - predictions)

        if self.method == "split":
            self._scores = residuals
            # Finite-sample valid quantile: ceil((n+1)*(1-alpha)) / n
            q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self._quantile = float(np.quantile(residuals, q_level))

        elif self.method == "adaptive":
            # Normalise residuals by local MAD estimate to get adaptive scores
            # Use a rolling MAD proxy: residuals / (local_std + eps)
            eps = 1e-8
            local_std = self._rolling_std(predictions, window=max(10, n // 20))
            normalised = residuals / (local_std + eps)
            self._scores = normalised
            self._mad_scale = local_std  # store for later reference

            q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self._quantile = float(np.quantile(normalised, q_level))

        self._calibrated = True
        logger.info(
            "Calibrated on %d samples – quantile=%.6f  method=%s",
            n,
            self._quantile,
            self.method,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, point_predictions: np.ndarray) -> Dict[str, Any]:
        """Produce prediction intervals around point predictions.

        Returns
        -------
        dict with keys:
            predictions, lower, upper, interval_width, coverage_guarantee
        """
        self._check_calibrated()
        preds = np.asarray(point_predictions, dtype=np.float64).ravel()

        if self.method == "split":
            half_width = np.full_like(preds, self._quantile)
        elif self.method == "adaptive":
            # Scale the quantile by an estimate of local volatility
            local_std = self._rolling_std(preds, window=max(10, len(preds) // 20))
            half_width = self._quantile * (local_std + 1e-8)
        else:
            half_width = np.full_like(preds, self._quantile)

        lower = preds - half_width
        upper = preds + half_width

        return {
            "predictions": preds,
            "lower": lower,
            "upper": upper,
            "interval_width": upper - lower,
            "coverage_guarantee": self.coverage_guarantee,
        }

    def predict_single(self, point_prediction: float) -> Dict[str, Any]:
        """Single-point prediction with conformal interval."""
        result = self.predict(np.array([point_prediction]))
        return {
            "prediction": float(result["predictions"][0]),
            "lower": float(result["lower"][0]),
            "upper": float(result["upper"][0]),
            "interval_width": float(result["interval_width"][0]),
            "coverage_guarantee": self.coverage_guarantee,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_calibrated(self) -> None:
        if not self._calibrated:
            raise RuntimeError("Must call .calibrate() before .predict()")

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int = 10) -> np.ndarray:
        """Simple rolling standard deviation (causal, padded)."""
        n = len(arr)
        window = max(2, min(window, n))
        out = np.empty(n, dtype=np.float64)
        cumsum = np.cumsum(arr)
        cumsum2 = np.cumsum(arr ** 2)
        for i in range(n):
            start = max(0, i - window + 1)
            cnt = i - start + 1
            mean = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / cnt
            mean2 = (cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)) / cnt
            out[i] = max(np.sqrt(max(mean2 - mean ** 2, 0.0)), 1e-8)
        return out


# ---------------------------------------------------------------------------
# TemporalConformalPredictor
# ---------------------------------------------------------------------------

class TemporalConformalPredictor(ConformalPredictor):
    """Time-series aware conformal prediction (TCP, Jul 2025).

    Handles temporal dependencies via exponentially-weighted calibration
    scores and supports online (streaming) updates.

    Parameters
    ----------
    alpha : float
        Miscoverage level (default 0.1 → 90 % coverage).
    window_size : int
        Maximum number of calibration scores to retain.
    decay_factor : float
        Exponential decay applied to older calibration scores (0 < λ ≤ 1).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 252,
        decay_factor: float = 0.99,
    ) -> None:
        super().__init__(alpha=alpha, method="split")
        self.window_size = window_size
        self.decay_factor = decay_factor

        # Temporal calibration buffers
        self._temporal_scores: List[float] = []
        self._temporal_timestamps: List[Any] = []
        self._temporal_weights: List[float] = []

        logger.info(
            "TemporalConformalPredictor: window=%d  decay=%.4f",
            window_size,
            decay_factor,
        )

    # ------------------------------------------------------------------
    # Temporal calibration
    # ------------------------------------------------------------------

    def calibrate_temporal(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Time-aware calibration with recency weighting.

        Parameters
        ----------
        predictions, actuals : array-like, shape (n,)
        timestamps : array-like, shape (n,)
            Monotonically-increasing timestamp or index values.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()
        timestamps = np.asarray(timestamps).ravel()

        if not (len(predictions) == len(actuals) == len(timestamps)):
            raise ValueError("predictions, actuals, timestamps must have equal length")

        n = len(predictions)
        residuals = np.abs(actuals - predictions)

        # Keep only the most recent `window_size` samples
        if n > self.window_size:
            residuals = residuals[-self.window_size:]
            timestamps = timestamps[-self.window_size:]
            predictions = predictions[-self.window_size:]
            actuals = actuals[-self.window_size:]

        # Exponential recency weights: most recent → weight 1.0
        m = len(residuals)
        weights = np.array(
            [self.decay_factor ** (m - 1 - i) for i in range(m)], dtype=np.float64
        )
        weights /= weights.sum()

        self._temporal_scores = residuals.tolist()
        self._temporal_timestamps = timestamps.tolist()
        self._temporal_weights = weights.tolist()

        # Weighted quantile
        self._quantile = self._weighted_quantile(
            residuals, weights, 1.0 - self.alpha
        )
        self._scores = residuals
        self._calibrated = True

        logger.info(
            "Temporal calibration on %d samples – quantile=%.6f",
            m,
            self._quantile,
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        new_prediction: float,
        new_actual: float,
        timestamp: Any = None,
    ) -> None:
        """Online update of the calibration set with a single new observation."""
        score = abs(new_actual - new_prediction)
        self._temporal_scores.append(score)
        self._temporal_timestamps.append(timestamp)

        # Trim to window
        if len(self._temporal_scores) > self.window_size:
            self._temporal_scores = self._temporal_scores[-self.window_size:]
            self._temporal_timestamps = self._temporal_timestamps[-self.window_size:]

        # Recompute weights
        m = len(self._temporal_scores)
        weights = np.array(
            [self.decay_factor ** (m - 1 - i) for i in range(m)], dtype=np.float64
        )
        weights /= weights.sum()
        self._temporal_weights = weights.tolist()

        scores_arr = np.array(self._temporal_scores, dtype=np.float64)
        self._quantile = self._weighted_quantile(scores_arr, weights, 1.0 - self.alpha)
        self._scores = scores_arr
        self._calibrated = True

    # ------------------------------------------------------------------
    # Weighted quantile helper
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        q: float,
    ) -> float:
        """Compute a weighted quantile (linear interpolation)."""
        idx = np.argsort(values)
        sorted_vals = values[idx]
        sorted_w = weights[idx]
        cum_w = np.cumsum(sorted_w)
        # Normalise cumulative weights to [0, 1]
        cum_w /= cum_w[-1]
        # Find first index where cumulative weight ≥ q
        pos = np.searchsorted(cum_w, q)
        pos = min(pos, len(sorted_vals) - 1)
        return float(sorted_vals[pos])


# ---------------------------------------------------------------------------
# SignalConformalWrapper
# ---------------------------------------------------------------------------

class SignalConformalWrapper:
    """Wraps any trading signal predictor with conformal prediction intervals.

    Integrates with the risk engine for position sizing: narrow intervals
    (high confidence) scale position size up; wide intervals scale down or
    skip the trade entirely.

    Parameters
    ----------
    predictor : Predictor
        Any object with a ``.predict(X)`` method returning an ndarray.
    conformal : ConformalPredictor
        A calibrated conformal predictor instance.
    risk_scaling : bool
        When True, compute ``position_size_multiplier`` based on interval
        width relative to the median calibration width.
    """

    # Thresholds for position sizing
    NARROW_THRESHOLD = 0.5   # ≤ 0.5x median → full position
    WIDE_THRESHOLD = 2.0     # ≥ 2.0x median → skip trade
    MIN_MULTIPLIER = 0.25

    def __init__(
        self,
        predictor: Any,
        conformal: ConformalPredictor,
        risk_scaling: bool = True,
    ) -> None:
        self.predictor = predictor
        self.conformal = conformal
        self.risk_scaling = risk_scaling

        # Pre-compute median interval width from calibration scores
        self._median_width: Optional[float] = None
        if conformal._calibrated and conformal._scores is not None:
            self._median_width = float(np.median(conformal._scores)) * 2.0

        logger.info(
            "SignalConformalWrapper: risk_scaling=%s  median_width=%s",
            risk_scaling,
            f"{self._median_width:.6f}" if self._median_width else "N/A",
        )

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict_with_confidence(self, X: Any) -> Dict[str, Any]:
        """Run the wrapped predictor and overlay conformal intervals.

        Returns
        -------
        dict with keys:
            predictions, lower, upper, interval_width,
            coverage_guarantee, position_size_multiplier (if risk_scaling)
        """
        raw_preds = np.asarray(self.predictor.predict(X), dtype=np.float64).ravel()
        conf_result = self.conformal.predict(raw_preds)

        result: Dict[str, Any] = {
            "predictions": conf_result["predictions"],
            "lower": conf_result["lower"],
            "upper": conf_result["upper"],
            "interval_width": conf_result["interval_width"],
            "coverage_guarantee": conf_result["coverage_guarantee"],
        }

        if self.risk_scaling:
            result["position_size_multiplier"] = self._compute_multiplier(
                conf_result["interval_width"]
            )

        return result

    # ------------------------------------------------------------------
    # Calibration statistics
    # ------------------------------------------------------------------

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Return empirical calibration diagnostics.

        Returns
        -------
        dict with keys:
            empirical_coverage, avg_interval_width, median_interval_width,
            efficiency_ratio, n_calibration_samples
        """
        if not self.conformal._calibrated or self.conformal._scores is None:
            return {"error": "conformal predictor not yet calibrated"}

        scores = self.conformal._scores
        n = len(scores)

        # For split conformal the "interval width" is 2 * score
        widths = scores * 2.0

        # Empirical coverage: fraction of scores ≤ quantile
        empirical_coverage = float(np.mean(scores <= self.conformal._quantile))

        avg_width = float(np.mean(widths))
        median_width = float(np.median(widths))

        # Efficiency = (ideal constant-width interval) / (avg actual width)
        # Lower avg_width → higher efficiency
        ideal_width = 2.0 * self.conformal._quantile if self.conformal._quantile else 1.0
        efficiency = ideal_width / avg_width if avg_width > 0 else 0.0

        return {
            "empirical_coverage": empirical_coverage,
            "avg_interval_width": avg_width,
            "median_interval_width": median_width,
            "efficiency_ratio": efficiency,
            "n_calibration_samples": n,
            "target_coverage": self.conformal.coverage_guarantee,
        }

    # ------------------------------------------------------------------
    # Risk-scaling helper
    # ------------------------------------------------------------------

    def _compute_multiplier(self, interval_widths: np.ndarray) -> np.ndarray:
        """Map interval widths to position-size multipliers in [0, 1]."""
        if self._median_width is None or self._median_width <= 0:
            return np.ones_like(interval_widths)

        ratio = interval_widths / self._median_width

        multiplier = np.ones_like(ratio)

        # Linear interpolation between NARROW and WIDE thresholds
        mid_mask = (ratio > self.NARROW_THRESHOLD) & (ratio < self.WIDE_THRESHOLD)
        multiplier[mid_mask] = 1.0 - (
            (ratio[mid_mask] - self.NARROW_THRESHOLD)
            / (self.WIDE_THRESHOLD - self.NARROW_THRESHOLD)
            * (1.0 - self.MIN_MULTIPLIER)
        )

        # Very wide → skip trade
        multiplier[ratio >= self.WIDE_THRESHOLD] = 0.0

        return multiplier


# ---------------------------------------------------------------------------
# PortfolioConformalPredictor
# ---------------------------------------------------------------------------

class PortfolioConformalPredictor:
    """Conformal prediction for portfolio-level risk management.

    Implements ideas from CPPS (Feb 2025): distribution-free prediction
    intervals for portfolio returns and conformal VaR / CVaR estimation.

    Parameters
    ----------
    alpha : float
        Miscoverage level for portfolio return intervals (default 0.1).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.coverage_guarantee = 1.0 - alpha

        # Calibration state
        self._portfolio_scores: Optional[np.ndarray] = None
        self._portfolio_quantile: Optional[float] = None
        self._calibrated = False

        logger.info(
            "PortfolioConformalPredictor: alpha=%.3f  coverage=%.1f%%",
            alpha,
            self.coverage_guarantee * 100,
        )

    # ------------------------------------------------------------------
    # Calibration on historical portfolio returns
    # ------------------------------------------------------------------

    def calibrate(
        self,
        historical_predicted_returns: np.ndarray,
        historical_actual_returns: np.ndarray,
    ) -> None:
        """Calibrate on historical portfolio-level prediction errors.

        Parameters
        ----------
        historical_predicted_returns : array, shape (T,)
        historical_actual_returns : array, shape (T,)
        """
        preds = np.asarray(historical_predicted_returns, dtype=np.float64).ravel()
        actuals = np.asarray(historical_actual_returns, dtype=np.float64).ravel()
        if len(preds) != len(actuals):
            raise ValueError("Predicted and actual return arrays must match in length")

        scores = np.abs(actuals - preds)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self._portfolio_scores = scores
        self._portfolio_quantile = float(np.quantile(scores, q_level))
        self._calibrated = True

        logger.info(
            "Portfolio conformal calibrated on %d periods – quantile=%.6f",
            n,
            self._portfolio_quantile,
        )

    # ------------------------------------------------------------------
    # Portfolio return prediction intervals
    # ------------------------------------------------------------------

    def predict_portfolio_return(
        self,
        weights: np.ndarray,
        asset_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute portfolio-level prediction intervals.

        Implements HR-LR CPPS: limits drawdowns while maintaining growth
        potential by providing calibrated bounds.

        Parameters
        ----------
        weights : array, shape (n_assets,)
            Portfolio weights (should sum to ~1).
        asset_predictions : array, shape (n_assets,) or (T, n_assets)
            Per-asset point predictions.  If 2-D, each row is a time step.

        Returns
        -------
        dict with keys:
            portfolio_prediction, lower, upper, interval_width,
            coverage_guarantee, weights_used
        """
        if not self._calibrated:
            raise RuntimeError("Must call .calibrate() first")

        weights = np.asarray(weights, dtype=np.float64).ravel()
        asset_predictions = np.asarray(asset_predictions, dtype=np.float64)

        if asset_predictions.ndim == 1:
            asset_predictions = asset_predictions.reshape(1, -1)

        if asset_predictions.shape[1] != len(weights):
            raise ValueError(
                f"asset_predictions has {asset_predictions.shape[1]} assets "
                f"but weights has {len(weights)}"
            )

        # Portfolio-level point prediction
        port_pred = asset_predictions @ weights  # shape (T,)

        half_width = self._portfolio_quantile
        lower = port_pred - half_width
        upper = port_pred + half_width

        return {
            "portfolio_prediction": port_pred.squeeze(),
            "lower": lower.squeeze(),
            "upper": upper.squeeze(),
            "interval_width": float(2.0 * half_width),
            "coverage_guarantee": self.coverage_guarantee,
            "weights_used": weights,
        }

    # ------------------------------------------------------------------
    # Conformal VaR / CVaR
    # ------------------------------------------------------------------

    def var_estimate(
        self,
        portfolio_returns: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Distribution-free Value-at-Risk and Conditional VaR.

        Uses conformal quantile estimation: no parametric distributional
        assumptions required.

        Parameters
        ----------
        portfolio_returns : array, shape (T,)
            Historical portfolio returns (or P&L).
        alpha : float
            VaR confidence level (default 0.05 → 95 % VaR).

        Returns
        -------
        dict with keys:
            var, cvar, coverage, n_observations, alpha
        """
        rets = np.asarray(portfolio_returns, dtype=np.float64).ravel()
        n = len(rets)
        if n < 2:
            raise ValueError("Need at least 2 return observations for VaR")

        # Conformal finite-sample correction
        q_level = np.ceil((n + 1) * alpha) / n
        q_level = min(q_level, 1.0)

        # VaR: quantile of losses (negative returns)
        sorted_rets = np.sort(rets)
        var_value = -float(np.quantile(sorted_rets, q_level))

        # CVaR (Expected Shortfall): mean of returns below VaR threshold
        tail_mask = rets <= -var_value
        if tail_mask.sum() > 0:
            cvar_value = -float(np.mean(rets[tail_mask]))
        else:
            cvar_value = var_value

        # Empirical coverage: fraction of returns above -VaR
        empirical_coverage = float(np.mean(rets > -var_value))

        return {
            "var": var_value,
            "cvar": cvar_value,
            "coverage": empirical_coverage,
            "n_observations": n,
            "alpha": alpha,
        }

    # ------------------------------------------------------------------
    # Portfolio-level drawdown bounds (CPPS HR-LR)
    # ------------------------------------------------------------------

    def drawdown_bounds(
        self,
        cumulative_returns: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Conformal bounds on maximum drawdown.

        Parameters
        ----------
        cumulative_returns : array, shape (T,)
            Cumulative return series (e.g. growth of $1).
        alpha : float or None
            Override miscoverage level (default: self.alpha).

        Returns
        -------
        dict with keys:
            max_drawdown, drawdown_bound, peak, trough, coverage
        """
        alpha = alpha if alpha is not None else self.alpha
        cum = np.asarray(cumulative_returns, dtype=np.float64).ravel()
        n = len(cum)

        # Rolling maximum and drawdown series
        running_max = np.maximum.accumulate(cum)
        drawdowns = (running_max - cum) / np.where(running_max > 0, running_max, 1.0)

        max_dd = float(np.max(drawdowns))
        peak_idx = int(np.argmax(running_max[:np.argmax(drawdowns) + 1]))
        trough_idx = int(np.argmax(drawdowns))

        # Conformal bound: quantile of drawdown distribution
        q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
        dd_bound = float(np.quantile(drawdowns, q_level))

        return {
            "max_drawdown": max_dd,
            "drawdown_bound": dd_bound,
            "peak_index": peak_idx,
            "trough_index": trough_idx,
            "coverage": 1.0 - alpha,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_conformal_predictor(
    method: str = "split",
    alpha: float = 0.1,
    **kwargs: Any,
) -> ConformalPredictor:
    """Factory function for conformal predictors.

    Parameters
    ----------
    method : str
        ``'split'``, ``'adaptive'``, or ``'temporal'``.
    alpha : float
        Miscoverage level (default 0.1).
    **kwargs
        Extra keyword arguments forwarded to the constructor
        (e.g. ``window_size``, ``decay_factor`` for temporal).
    """
    if method == "temporal":
        return TemporalConformalPredictor(alpha=alpha, **kwargs)
    return ConformalPredictor(alpha=alpha, method=method)
