"""
Meta-learning regime detection and adaptation module.

Detects market regimes using an ensemble of Hidden Markov Models,
Gaussian Mixture clustering, and rule-based indicators. Adapts
trading strategy weights, agent weights, and risk parameters based
on the detected regime.

References:
    - FinPFN: Meta-learning for financial prediction (Nov 2025)
    - Dual-Head PPO: Regime-aware reinforcement learning (Sep 2025)
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from sklearn.mixture import GaussianMixture

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from hmmlearn.hmm import GaussianHMM

    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Market Regime Enum
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    """Discrete market regime classification."""

    LOW_VOL_BULL = "low_vol_bull"  # Calm uptrend – sell premium
    HIGH_VOL_BULL = "high_vol_bull"  # Volatile uptrend – directional spreads
    LOW_VOL_BEAR = "low_vol_bear"  # Calm downtrend – protective puts
    HIGH_VOL_BEAR = "high_vol_bear"  # Crisis mode – buy protection
    MEAN_REVERTING = "mean_reverting"  # Range-bound – iron condors
    TRENDING = "trending"  # Strong trend – directional
    TRANSITION = "transition"  # Regime change detected – reduce exposure


# Mapping from integer labels (HMM / GMM) to MarketRegime
_LABEL_TO_REGIME: Dict[int, MarketRegime] = {
    0: MarketRegime.LOW_VOL_BULL,
    1: MarketRegime.HIGH_VOL_BULL,
    2: MarketRegime.LOW_VOL_BEAR,
    3: MarketRegime.HIGH_VOL_BEAR,
    4: MarketRegime.MEAN_REVERTING,
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RegimeSnapshot:
    """Single point-in-time regime detection result."""

    regime: MarketRegime
    confidence: float
    features: Dict[str, float]
    transition_probability: float
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class TransitionInfo:
    """Information about a regime transition."""

    previous_regime: MarketRegime
    current_regime: MarketRegime
    days_in_current: int
    transition_probability: float
    recommended_action: str


# ---------------------------------------------------------------------------
# HMM Regime Model
# ---------------------------------------------------------------------------


class HMMRegimeModel:
    """Hidden Markov Model for regime detection.

    Uses *hmmlearn* when available; falls back to a manual Viterbi
    implementation backed by Gaussian emissions otherwise.
    """

    def __init__(self, n_states: int = 5, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: Any = None
        self._fitted = False

        # Manual fallback parameters
        self._means: Optional[np.ndarray] = None
        self._covars: Optional[np.ndarray] = None
        self._trans_mat: Optional[np.ndarray] = None
        self._start_prob: Optional[np.ndarray] = None

    # -- public API ----------------------------------------------------------

    def fit(self, features: np.ndarray, n_states: Optional[int] = None) -> "HMMRegimeModel":
        """Fit the HMM to *features* (T x D)."""
        if n_states is not None:
            self.n_states = n_states

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        if _HAS_HMMLEARN:
            self._fit_hmmlearn(features)
        else:
            self._fit_manual(features)

        self._fitted = True
        logger.info("HMM fitted with %d states on %d observations", self.n_states, len(features))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return most-likely state sequence for *features*."""
        if not self._fitted:
            raise RuntimeError("HMMRegimeModel has not been fitted yet")

        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        if _HAS_HMMLEARN and self._model is not None:
            return self._model.predict(features)
        return self._viterbi(features)

    def predict_single(self, features: np.ndarray) -> int:
        """Predict regime label for a single observation."""
        seq = self.predict(features[-1:])
        return int(seq[0])

    def transition_matrix(self) -> np.ndarray:
        """Return the state-transition matrix (n_states x n_states)."""
        if not self._fitted:
            raise RuntimeError("HMMRegimeModel has not been fitted yet")
        if _HAS_HMMLEARN and self._model is not None:
            return np.array(self._model.transmat_)
        assert self._trans_mat is not None
        return self._trans_mat.copy()

    # -- hmmlearn backend ----------------------------------------------------

    def _fit_hmmlearn(self, features: np.ndarray) -> None:
        self._model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(features)

    # -- manual Gaussian HMM with Viterbi ------------------------------------

    def _fit_manual(self, features: np.ndarray) -> None:
        """Rough EM-initialised fit using k-means + count transitions."""
        T, D = features.shape
        rng = np.random.RandomState(self.random_state)
        K = self.n_states

        # Initialise with k-means-style assignment
        indices = rng.choice(T, size=K, replace=False)
        centres = features[indices].copy()

        labels = np.zeros(T, dtype=int)
        for _ in range(30):
            dists = np.linalg.norm(features[:, None, :] - centres[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    centres[k] = features[mask].mean(axis=0)

        # Estimate Gaussian params per state
        self._means = np.zeros((K, D))
        self._covars = np.zeros((K, D, D))
        for k in range(K):
            mask = labels == k
            if mask.sum() < 2:
                self._means[k] = centres[k]
                self._covars[k] = np.eye(D) * 1e-2
            else:
                self._means[k] = features[mask].mean(axis=0)
                cov = np.cov(features[mask], rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                self._covars[k] = cov + np.eye(D) * 1e-6

        # Transition counts
        trans = np.ones((K, K)) * 1e-3
        for t in range(T - 1):
            trans[labels[t], labels[t + 1]] += 1
        self._trans_mat = trans / trans.sum(axis=1, keepdims=True)

        # Start probability
        start = np.ones(K) * 1e-3
        start[labels[0]] += 1
        self._start_prob = start / start.sum()

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log-probability of *obs* (1 x D) under each Gaussian."""
        assert self._means is not None and self._covars is not None
        K = self.n_states
        D = obs.shape[-1]
        log_probs = np.zeros(K)
        for k in range(K):
            diff = obs - self._means[k]
            cov = self._covars[k]
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = D * np.log(1e-6)
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logdet = D * np.log(1e-6)
                inv_cov = np.eye(D) * 1e6
            log_probs[k] = -0.5 * (D * np.log(2 * np.pi) + logdet + diff @ inv_cov @ diff)
        return log_probs

    def _viterbi(self, features: np.ndarray) -> np.ndarray:
        """Manual Viterbi decoding."""
        assert self._trans_mat is not None and self._start_prob is not None
        T = len(features)
        K = self.n_states

        log_trans = np.log(self._trans_mat + 1e-300)
        log_start = np.log(self._start_prob + 1e-300)

        # Trellis
        V = np.zeros((T, K))
        ptr = np.zeros((T, K), dtype=int)

        V[0] = log_start + self._log_emission(features[0])
        for t in range(1, T):
            emit = self._log_emission(features[t])
            for k in range(K):
                scores = V[t - 1] + log_trans[:, k]
                ptr[t, k] = int(np.argmax(scores))
                V[t, k] = scores[ptr[t, k]] + emit[k]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]
        return path


# ---------------------------------------------------------------------------
# Regime Detector (ensemble)
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Meta-learning regime detection and adaptation.

    Based on FinPFN (Nov 2025) and Dual-Head PPO (Sep 2025) research.
    Combines HMM, GMM clustering, and rule-based indicators to classify
    the current market regime with confidence scores.
    """

    def __init__(
        self,
        feature_window: int = 60,
        n_regimes: int = 5,
        method: str = "ensemble",
    ):
        """
        Args:
            feature_window: Rolling window length for feature computation.
            n_regimes: Number of latent regimes for statistical models.
            method: Detection method – 'hmm', 'clustering', 'ensemble'.
        """
        if method not in ("hmm", "clustering", "ensemble"):
            raise ValueError(f"Unknown method '{method}'; choose hmm|clustering|ensemble")

        self.feature_window = feature_window
        self.n_regimes = n_regimes
        self.method = method

        # Sub-models
        self._hmm = HMMRegimeModel(n_states=n_regimes)
        self._gmm: Optional[Any] = None  # sklearn GaussianMixture
        self._hmm_fitted = False
        self._gmm_fitted = False

        # History
        self._regime_history: deque[RegimeSnapshot] = deque(maxlen=2520)  # ~10 yrs daily
        self._feature_buffer: deque[np.ndarray] = deque(maxlen=max(feature_window * 2, 252))

        logger.info(
            "RegimeDetector initialised: method=%s, window=%d, n_regimes=%d",
            method,
            feature_window,
            n_regimes,
        )

    # -- feature extraction ---------------------------------------------------

    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """Build a 22-dimensional feature vector from raw market data.

        Expected keys in *market_data* (missing keys default to NaN):
            vix, vix_3m, realized_vol, implied_vol, gex,
            hy_spread, ig_spread, hy_spread_change,
            breadth_pct, spx_return, spx_price,
            put_call_ratio, skew_index
        """

        def _g(key: str, default: float = np.nan) -> float:
            return float(market_data.get(key, default))

        vix = _g("vix")
        vix_3m = _g("vix_3m", vix)  # 3-month VIX for term structure
        rvol = _g("realized_vol")
        ivol = _g("implied_vol", vix / 100.0 if not np.isnan(vix) else np.nan)
        gex = _g("gex", 0.0)
        hy_spread = _g("hy_spread")
        ig_spread = _g("ig_spread")
        hy_chg = _g("hy_spread_change", 0.0)
        breadth = _g("breadth_pct")
        spx_ret = _g("spx_return", 0.0)
        spx_price = _g("spx_price")
        pcr = _g("put_call_ratio", 1.0)
        skew = _g("skew_index", 100.0)

        # Derived indicators
        vix_term_slope = (vix_3m - vix) if not (np.isnan(vix_3m) or np.isnan(vix)) else 0.0
        rv_iv_ratio = (rvol / ivol) if (ivol and not np.isnan(ivol) and ivol > 1e-9) else 1.0
        credit_spread = (hy_spread - ig_spread) if not (np.isnan(hy_spread) or np.isnan(ig_spread)) else 0.0
        gex_sign = float(np.sign(gex))

        # Rolling stats from buffer
        buf = list(self._feature_buffer)
        if len(buf) >= 5:
            recent = np.array(buf[-5:])
            mean_5 = float(np.nanmean(recent[:, 0]))  # vix rolling mean
            std_5 = float(np.nanstd(recent[:, 0]))
        else:
            mean_5 = vix if not np.isnan(vix) else 0.0
            std_5 = 0.0

        if len(buf) >= 20:
            recent20 = np.array(buf[-20:])
            mean_20 = float(np.nanmean(recent20[:, 0]))
            std_20 = float(np.nanstd(recent20[:, 0]))
            ret_20 = float(np.nanmean(recent20[:, 9]))  # spx_return col
        else:
            mean_20 = vix if not np.isnan(vix) else 0.0
            std_20 = 0.0
            ret_20 = spx_ret

        feature_vec = np.array(
            [
                vix,                   # 0  VIX level
                vix_term_slope,        # 1  term structure slope
                rvol,                  # 2  realised vol
                ivol,                  # 3  implied vol
                rv_iv_ratio,           # 4  RV/IV ratio
                gex,                   # 5  gamma exposure
                gex_sign,              # 6  GEX sign
                hy_spread,             # 7  HY spread
                ig_spread,             # 8  IG spread
                credit_spread,         # 9  HY-IG
                hy_chg,                # 10 HY spread change
                breadth,               # 11 market breadth %
                spx_ret,               # 12 SPX daily return
                spx_price,             # 13 SPX price
                pcr,                   # 14 put/call ratio
                skew,                  # 15 CBOE skew
                mean_5,                # 16 VIX 5d mean
                std_5,                 # 17 VIX 5d std
                mean_20,               # 18 VIX 20d mean
                std_20,                # 19 VIX 20d std
                ret_20,                # 20 SPX 20d mean return
                vix_term_slope * rv_iv_ratio,  # 21 interaction term
            ],
            dtype=np.float64,
        )

        # Replace NaN with 0 for model consumption
        feature_vec = np.nan_to_num(feature_vec, nan=0.0)
        self._feature_buffer.append(feature_vec)
        return feature_vec

    # -- regime detection -----------------------------------------------------

    def detect_regime(self, market_data: Dict) -> Dict:
        """Detect the current market regime.

        Returns a dict with keys:
            regime (MarketRegime), confidence (float 0-1),
            features (Dict), transition_probability (float).
        """
        features = self._extract_features(market_data)

        if self.method == "hmm":
            regime, confidence = self._detect_hmm(features)
        elif self.method == "clustering":
            regime, confidence = self._detect_gmm(features)
        else:  # ensemble
            regime, confidence = self._detect_ensemble(features)

        # Detect transitions
        trans_prob = 0.0
        if self._regime_history:
            prev = self._regime_history[-1].regime
            if prev != regime:
                trans_prob = 1.0 - confidence
                if confidence < 0.5:
                    regime = MarketRegime.TRANSITION
                    confidence = 0.5

        snapshot = RegimeSnapshot(
            regime=regime,
            confidence=round(confidence, 4),
            features={
                "vix": float(features[0]),
                "vix_term_slope": float(features[1]),
                "rv_iv_ratio": float(features[4]),
                "gex_sign": float(features[6]),
                "credit_spread": float(features[9]),
                "breadth_pct": float(features[11]),
                "spx_return": float(features[12]),
                "put_call_ratio": float(features[14]),
            },
            transition_probability=round(trans_prob, 4),
            timestamp=pd.Timestamp.now(),
        )
        self._regime_history.append(snapshot)

        logger.debug("Regime detected: %s (conf=%.2f)", regime.value, confidence)
        return {
            "regime": regime,
            "confidence": snapshot.confidence,
            "features": snapshot.features,
            "transition_probability": snapshot.transition_probability,
        }

    # -- sub-detectors --------------------------------------------------------

    def _detect_rule_based(self, features: np.ndarray) -> Tuple[MarketRegime, float]:
        """Deterministic rule-based classification."""
        vix = features[0]
        term_slope = features[1]
        rv_iv = features[4]
        gex_sign = features[6]
        credit_spread = features[9]
        breadth = features[11]
        spx_ret = features[12]

        # Thresholds
        high_vol = vix > 25
        crisis_vol = vix > 35
        bullish = spx_ret > 0 and breadth > 50
        bearish = spx_ret < 0 and breadth < 50
        mean_rev = abs(gex_sign) > 0 and gex_sign > 0 and abs(spx_ret) < 0.005
        trending = abs(spx_ret) > 0.01

        if crisis_vol and bearish:
            return MarketRegime.HIGH_VOL_BEAR, 0.85
        if crisis_vol and bullish:
            return MarketRegime.HIGH_VOL_BULL, 0.70
        if high_vol and bearish:
            return MarketRegime.HIGH_VOL_BEAR, 0.70
        if high_vol and bullish:
            return MarketRegime.HIGH_VOL_BULL, 0.65
        if trending:
            return MarketRegime.TRENDING, 0.60
        if mean_rev and not high_vol:
            return MarketRegime.MEAN_REVERTING, 0.65
        if not high_vol and bullish:
            return MarketRegime.LOW_VOL_BULL, 0.70
        if not high_vol and bearish:
            return MarketRegime.LOW_VOL_BEAR, 0.65

        # Fallback
        return MarketRegime.MEAN_REVERTING, 0.40

    def _detect_hmm(self, features: np.ndarray) -> Tuple[MarketRegime, float]:
        """HMM-based regime detection."""
        buf = np.array(list(self._feature_buffer))
        if len(buf) < max(self.feature_window, 30):
            return self._detect_rule_based(features)

        if not self._hmm_fitted:
            try:
                self._hmm.fit(buf)
                self._hmm_fitted = True
            except Exception as exc:
                logger.warning("HMM fit failed: %s – using rule-based fallback", exc)
                return self._detect_rule_based(features)

        try:
            label = self._hmm.predict_single(buf)
            regime = _LABEL_TO_REGIME.get(label % len(_LABEL_TO_REGIME), MarketRegime.MEAN_REVERTING)
            # Confidence from transition-matrix diagonal
            trans = self._hmm.transition_matrix()
            confidence = float(trans[label, label])
            return regime, min(confidence, 0.95)
        except Exception as exc:
            logger.warning("HMM predict failed: %s", exc)
            return self._detect_rule_based(features)

    def _detect_gmm(self, features: np.ndarray) -> Tuple[MarketRegime, float]:
        """GMM clustering-based regime detection."""
        if not _HAS_SKLEARN:
            logger.warning("sklearn not available – falling back to rule-based")
            return self._detect_rule_based(features)

        buf = np.array(list(self._feature_buffer))
        if len(buf) < max(self.feature_window, 30):
            return self._detect_rule_based(features)

        if not self._gmm_fitted:
            try:
                self._gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    max_iter=200,
                    random_state=42,
                )
                self._gmm.fit(buf)
                self._gmm_fitted = True
            except Exception as exc:
                logger.warning("GMM fit failed: %s", exc)
                return self._detect_rule_based(features)

        try:
            probs = self._gmm.predict_proba(features.reshape(1, -1))[0]
            label = int(np.argmax(probs))
            confidence = float(probs[label])
            regime = _LABEL_TO_REGIME.get(label % len(_LABEL_TO_REGIME), MarketRegime.MEAN_REVERTING)
            return regime, min(confidence, 0.95)
        except Exception as exc:
            logger.warning("GMM predict failed: %s", exc)
            return self._detect_rule_based(features)

    def _detect_ensemble(self, features: np.ndarray) -> Tuple[MarketRegime, float]:
        """Ensemble of HMM + GMM + rule-based with weighted voting."""
        rule_regime, rule_conf = self._detect_rule_based(features)
        hmm_regime, hmm_conf = self._detect_hmm(features)
        gmm_regime, gmm_conf = self._detect_gmm(features)

        # Weighted vote
        candidates: Dict[MarketRegime, float] = {}
        w_rule, w_hmm, w_gmm = 0.3, 0.35, 0.35
        for regime, conf, w in [
            (rule_regime, rule_conf, w_rule),
            (hmm_regime, hmm_conf, w_hmm),
            (gmm_regime, gmm_conf, w_gmm),
        ]:
            candidates[regime] = candidates.get(regime, 0.0) + conf * w

        best_regime = max(candidates, key=candidates.get)  # type: ignore[arg-type]
        total_weight = sum(candidates.values())
        confidence = candidates[best_regime] / total_weight if total_weight > 0 else 0.5
        return best_regime, min(confidence, 0.95)

    # -- history & transitions -----------------------------------------------

    def get_regime_history(self, lookback_days: int = 252) -> pd.DataFrame:
        """Return historical regime classifications as a DataFrame."""
        history = list(self._regime_history)[-lookback_days:]
        if not history:
            return pd.DataFrame(columns=["timestamp", "regime", "confidence", "transition_probability"])

        rows = [
            {
                "timestamp": s.timestamp,
                "regime": s.regime.value,
                "confidence": s.confidence,
                "transition_probability": s.transition_probability,
                **s.features,
            }
            for s in history
        ]
        return pd.DataFrame(rows)

    def detect_transition(
        self,
        current: MarketRegime,
        previous: MarketRegime,
    ) -> Dict:
        """Analyse a regime transition and recommend action.

        Returns:
            Dict with days_in_current_regime, transition_probability,
            recommended_action.
        """
        # Count days in current regime
        days_in_current = 0
        for snap in reversed(list(self._regime_history)):
            if snap.regime == current:
                days_in_current += 1
            else:
                break

        # Transition probability from HMM if available
        trans_prob = 0.0
        if self._hmm_fitted:
            try:
                trans = self._hmm.transition_matrix()
                # Approximate: max off-diagonal element for current state
                cur_idx = list(_LABEL_TO_REGIME.values()).index(current) if current in _LABEL_TO_REGIME.values() else 0
                if cur_idx < trans.shape[0]:
                    row = trans[cur_idx].copy()
                    row[cur_idx] = 0.0
                    trans_prob = float(np.max(row))
            except Exception:
                trans_prob = 0.1

        # Action recommendation
        action_map: Dict[Tuple[str, str], str] = {
            ("bull", "bear"): "hedge_portfolio_reduce_exposure",
            ("bear", "bull"): "increase_exposure_add_longs",
            ("low_vol", "high_vol"): "reduce_short_gamma_add_protection",
            ("high_vol", "low_vol"): "sell_premium_increase_theta",
        }

        prev_cat = "bull" if "bull" in previous.value else ("bear" if "bear" in previous.value else "neutral")
        curr_cat = "bull" if "bull" in current.value else ("bear" if "bear" in current.value else "neutral")
        prev_vol = "high_vol" if "high_vol" in previous.value else "low_vol"
        curr_vol = "high_vol" if "high_vol" in current.value else "low_vol"

        action = action_map.get(
            (prev_cat, curr_cat),
            action_map.get((prev_vol, curr_vol), "rebalance_to_neutral"),
        )

        if current == MarketRegime.TRANSITION:
            action = "reduce_exposure_wait_for_clarity"

        info = TransitionInfo(
            previous_regime=previous,
            current_regime=current,
            days_in_current=days_in_current,
            transition_probability=round(trans_prob, 4),
            recommended_action=action,
        )

        return {
            "previous_regime": info.previous_regime.value,
            "current_regime": info.current_regime.value,
            "days_in_current_regime": info.days_in_current,
            "transition_probability": info.transition_probability,
            "recommended_action": info.recommended_action,
        }

    # -- model persistence ---------------------------------------------------

    def refit(self, force: bool = False) -> None:
        """Re-fit statistical models on accumulated buffer data."""
        buf = np.array(list(self._feature_buffer))
        if len(buf) < max(self.feature_window, 30):
            logger.info("Not enough data to refit (%d samples)", len(buf))
            return

        if force or not self._hmm_fitted:
            try:
                self._hmm.fit(buf)
                self._hmm_fitted = True
                logger.info("HMM refitted on %d samples", len(buf))
            except Exception as exc:
                logger.warning("HMM refit failed: %s", exc)

        if _HAS_SKLEARN and (force or not self._gmm_fitted):
            try:
                self._gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    max_iter=200,
                    random_state=42,
                )
                self._gmm.fit(buf)
                self._gmm_fitted = True
                logger.info("GMM refitted on %d samples", len(buf))
            except Exception as exc:
                logger.warning("GMM refit failed: %s", exc)


# ---------------------------------------------------------------------------
# Adaptive Strategy Selector
# ---------------------------------------------------------------------------


class AdaptiveStrategySelector:
    """Adapts trading strategy, agent weights, and risk parameters
    based on the detected market regime."""

    # Strategy-weight templates per regime
    _STRATEGY_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
        MarketRegime.LOW_VOL_BULL: {
            "iron_condor": 0.40,
            "credit_put_spread": 0.30,
            "covered_call": 0.30,
        },
        MarketRegime.HIGH_VOL_BULL: {
            "call_debit_spread": 0.35,
            "bull_put_spread": 0.30,
            "long_call": 0.20,
            "iron_condor": 0.15,
        },
        MarketRegime.LOW_VOL_BEAR: {
            "protective_put": 0.35,
            "bear_call_spread": 0.30,
            "collar": 0.20,
            "iron_condor": 0.15,
        },
        MarketRegime.HIGH_VOL_BEAR: {
            "long_put": 0.40,
            "put_debit_spread": 0.30,
            "collar": 0.30,
        },
        MarketRegime.MEAN_REVERTING: {
            "iron_condor": 0.45,
            "iron_butterfly": 0.25,
            "credit_put_spread": 0.15,
            "credit_call_spread": 0.15,
        },
        MarketRegime.TRENDING: {
            "long_call": 0.25,
            "long_put": 0.25,
            "call_debit_spread": 0.25,
            "put_debit_spread": 0.25,
        },
        MarketRegime.TRANSITION: {
            "collar": 0.30,
            "protective_put": 0.30,
            "cash": 0.40,
        },
    }

    # Agent-weight templates per regime
    _AGENT_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
        MarketRegime.LOW_VOL_BULL: {
            "fundamental": 0.25,
            "technical": 0.30,
            "sentiment": 0.15,
            "risk": 0.15,
            "options": 0.15,
        },
        MarketRegime.HIGH_VOL_BULL: {
            "fundamental": 0.20,
            "technical": 0.25,
            "sentiment": 0.20,
            "risk": 0.20,
            "options": 0.15,
        },
        MarketRegime.LOW_VOL_BEAR: {
            "fundamental": 0.30,
            "technical": 0.20,
            "sentiment": 0.15,
            "risk": 0.20,
            "options": 0.15,
        },
        MarketRegime.HIGH_VOL_BEAR: {
            "fundamental": 0.15,
            "technical": 0.15,
            "sentiment": 0.25,
            "risk": 0.30,
            "options": 0.15,
        },
        MarketRegime.MEAN_REVERTING: {
            "fundamental": 0.20,
            "technical": 0.35,
            "sentiment": 0.10,
            "risk": 0.15,
            "options": 0.20,
        },
        MarketRegime.TRENDING: {
            "fundamental": 0.20,
            "technical": 0.35,
            "sentiment": 0.15,
            "risk": 0.15,
            "options": 0.15,
        },
        MarketRegime.TRANSITION: {
            "fundamental": 0.15,
            "technical": 0.15,
            "sentiment": 0.20,
            "risk": 0.35,
            "options": 0.15,
        },
    }

    # Risk-parameter templates per regime
    _RISK_PARAMS: Dict[MarketRegime, Dict[str, float]] = {
        MarketRegime.LOW_VOL_BULL: {
            "max_position_risk": 0.025,
            "max_daily_loss": 0.035,
            "max_portfolio_delta": 0.30,
            "max_portfolio_vega": 0.15,
            "position_size_mult": 1.2,
            "stop_loss_mult": 1.0,
        },
        MarketRegime.HIGH_VOL_BULL: {
            "max_position_risk": 0.020,
            "max_daily_loss": 0.030,
            "max_portfolio_delta": 0.25,
            "max_portfolio_vega": 0.20,
            "position_size_mult": 0.9,
            "stop_loss_mult": 1.2,
        },
        MarketRegime.LOW_VOL_BEAR: {
            "max_position_risk": 0.018,
            "max_daily_loss": 0.025,
            "max_portfolio_delta": 0.20,
            "max_portfolio_vega": 0.15,
            "position_size_mult": 0.8,
            "stop_loss_mult": 1.1,
        },
        MarketRegime.HIGH_VOL_BEAR: {
            "max_position_risk": 0.010,
            "max_daily_loss": 0.020,
            "max_portfolio_delta": 0.15,
            "max_portfolio_vega": 0.25,
            "position_size_mult": 0.5,
            "stop_loss_mult": 1.5,
        },
        MarketRegime.MEAN_REVERTING: {
            "max_position_risk": 0.022,
            "max_daily_loss": 0.030,
            "max_portfolio_delta": 0.15,
            "max_portfolio_vega": 0.10,
            "position_size_mult": 1.0,
            "stop_loss_mult": 1.0,
        },
        MarketRegime.TRENDING: {
            "max_position_risk": 0.020,
            "max_daily_loss": 0.030,
            "max_portfolio_delta": 0.35,
            "max_portfolio_vega": 0.15,
            "position_size_mult": 1.0,
            "stop_loss_mult": 0.9,
        },
        MarketRegime.TRANSITION: {
            "max_position_risk": 0.010,
            "max_daily_loss": 0.015,
            "max_portfolio_delta": 0.10,
            "max_portfolio_vega": 0.10,
            "position_size_mult": 0.4,
            "stop_loss_mult": 1.5,
        },
    }

    def __init__(self, regime_detector: RegimeDetector):
        self.regime_detector = regime_detector
        logger.info("AdaptiveStrategySelector initialised")

    def get_strategy_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Return option-strategy allocation weights for a given regime."""
        return dict(self._STRATEGY_WEIGHTS.get(regime, self._STRATEGY_WEIGHTS[MarketRegime.TRANSITION]))

    def get_agent_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Return multi-agent weighting for a given regime.

        HIGH_VOL regimes boost risk_analyst and sentiment weights.
        LOW_VOL regimes boost technical and fundamental weights.
        """
        return dict(self._AGENT_WEIGHTS.get(regime, self._AGENT_WEIGHTS[MarketRegime.TRANSITION]))

    def get_risk_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """Return adjusted risk-management parameters for a given regime.

        Crisis regimes tighten limits; calm regimes allow more exposure.
        """
        return dict(self._RISK_PARAMS.get(regime, self._RISK_PARAMS[MarketRegime.TRANSITION]))

    def adapt_pipeline(self, regime: MarketRegime, pipeline_config: Dict) -> Dict:
        """Fully adapt a pipeline configuration to the current regime.

        Merges strategy weights, agent weights, and risk parameters into
        the provided *pipeline_config* dict (non-destructively).
        """
        adapted = dict(pipeline_config)
        adapted["regime"] = regime.value
        adapted["strategy_weights"] = self.get_strategy_weights(regime)
        adapted["agent_weights"] = self.get_agent_weights(regime)
        adapted["risk_params"] = self.get_risk_adjustments(regime)

        # Scale position sizes
        risk = adapted["risk_params"]
        if "base_position_size" in adapted:
            adapted["adjusted_position_size"] = (
                adapted["base_position_size"] * risk.get("position_size_mult", 1.0)
            )

        # Adjust stop-loss if present
        if "base_stop_loss" in adapted:
            adapted["adjusted_stop_loss"] = (
                adapted["base_stop_loss"] * risk.get("stop_loss_mult", 1.0)
            )

        logger.info(
            "Pipeline adapted for regime %s: pos_mult=%.1f, max_risk=%.3f",
            regime.value,
            risk.get("position_size_mult", 1.0),
            risk.get("max_position_risk", 0.02),
        )
        return adapted

    def full_detect_and_adapt(self, market_data: Dict, pipeline_config: Dict) -> Dict:
        """Convenience: detect regime then adapt pipeline in one call."""
        result = self.regime_detector.detect_regime(market_data)
        regime: MarketRegime = result["regime"]
        adapted = self.adapt_pipeline(regime, pipeline_config)
        adapted["detection"] = result
        return adapted
