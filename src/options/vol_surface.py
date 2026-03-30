"""Volatility surface modeling, interpolation, and variance risk premium analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata, CubicSpline

from ..utils.logging import get_logger

logger = get_logger(__name__)


class VolatilitySurface:
    """
    Build, interpolate, and analyze implied-volatility surfaces.

    The surface is stored as a 2-D array indexed by (moneyness, time_to_expiry).
    Moneyness is defined as strike / spot.
    """

    def __init__(
        self,
        chains_provider=None,
        greeks_provider=None,
        history_provider=None,
    ):
        """
        Args:
            chains_provider: object with ``get_chain(symbol, expiration)`` and
                ``get_expirations(symbol)`` returning options chain DataFrames.
            greeks_provider: object with ``implied_volatility(...)`` if IVs are
                not already present in the chain data.
            history_provider: object with ``get_bars(symbol, ...)`` returning
                historical price DataFrame (needs 'close' column).
        """
        self.chains = chains_provider
        self.greeks = greeks_provider
        self.history = history_provider

        # Cached surface data
        self._surface: Optional[Dict[str, Any]] = None
        self._symbol: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_surface(self, symbol: str) -> Dict[str, Any]:
        """Fetch all expirations and build IV surface organised by (strike, expiry).

        Returns a dict with keys:
            moneyness   : 1-D np.ndarray of moneyness values (sorted)
            tte         : 1-D np.ndarray of time-to-expiry in years (sorted)
            iv_grid     : 2-D np.ndarray shape (len(moneyness), len(tte))
            spot        : current spot price
            raw_points  : list of dicts with raw (strike, expiry, iv, moneyness, tte)
            built_at    : ISO timestamp
        """
        logger.info("Building vol surface for %s", symbol)

        expirations = self.chains.get_expirations(symbol)
        if not expirations:
            raise ValueError(f"No expirations found for {symbol}")

        spot = self._get_spot(symbol)
        raw_points: List[Dict[str, Any]] = []
        today = datetime.utcnow().date()

        for exp in expirations:
            exp_date = pd.Timestamp(exp).date() if not isinstance(exp, datetime) else exp.date()
            tte_days = (exp_date - today).days
            if tte_days <= 0:
                continue
            tte_years = tte_days / 365.0

            chain = self.chains.get_chain(symbol, exp)
            if chain is None or chain.empty:
                continue

            for _, row in chain.iterrows():
                iv = self._extract_iv(row, spot)
                if iv is None or np.isnan(iv) or iv <= 0:
                    continue
                strike = float(row.get("strike", 0))
                if strike <= 0:
                    continue
                moneyness = strike / spot
                raw_points.append(
                    {
                        "strike": strike,
                        "expiry": str(exp),
                        "iv": float(iv),
                        "moneyness": moneyness,
                        "tte": tte_years,
                    }
                )

        if len(raw_points) < 4:
            raise ValueError(
                f"Insufficient data to build surface for {symbol} "
                f"(got {len(raw_points)} valid IV points)"
            )

        # Build regular grid via interpolation
        moneyness_vals = np.array([p["moneyness"] for p in raw_points])
        tte_vals = np.array([p["tte"] for p in raw_points])
        iv_vals = np.array([p["iv"] for p in raw_points])

        # Create grid axes
        m_min, m_max = moneyness_vals.min(), moneyness_vals.max()
        t_min, t_max = tte_vals.min(), tte_vals.max()
        n_m = min(50, len(np.unique(moneyness_vals)))
        n_t = min(30, len(np.unique(tte_vals)))

        m_grid = np.linspace(m_min, m_max, n_m)
        t_grid = np.linspace(t_min, t_max, n_t)
        mm, tt = np.meshgrid(m_grid, t_grid, indexing="ij")

        # Interpolate scattered data onto regular grid
        iv_grid = griddata(
            (moneyness_vals, tte_vals),
            iv_vals,
            (mm, tt),
            method="cubic",
            fill_value=np.nan,
        )
        # Fill remaining NaNs with linear interpolation
        nan_mask = np.isnan(iv_grid)
        if nan_mask.any():
            iv_linear = griddata(
                (moneyness_vals, tte_vals),
                iv_vals,
                (mm, tt),
                method="linear",
                fill_value=np.nanmean(iv_vals),
            )
            iv_grid[nan_mask] = iv_linear[nan_mask]

        surface = {
            "moneyness": m_grid,
            "tte": t_grid,
            "iv_grid": iv_grid,
            "spot": spot,
            "raw_points": raw_points,
            "built_at": datetime.utcnow().isoformat(),
        }

        self._surface = surface
        self._symbol = symbol
        logger.info(
            "Vol surface built: %d raw points -> %dx%d grid",
            len(raw_points),
            n_m,
            n_t,
        )
        return surface

    def interpolate_iv(self, strike: float, tte: float) -> float:
        """Interpolate IV for arbitrary strike / time-to-expiry.

        Uses RectBivariateSpline on the cached regular grid.

        Args:
            strike: absolute strike price
            tte: time to expiry in years

        Returns:
            Interpolated implied volatility (annualised, decimal).
        """
        if self._surface is None:
            raise RuntimeError("Surface not built – call build_surface() first")

        spot = self._surface["spot"]
        moneyness = strike / spot

        m_grid = self._surface["moneyness"]
        t_grid = self._surface["tte"]
        iv_grid = self._surface["iv_grid"]

        # Clamp to grid bounds
        moneyness = np.clip(moneyness, m_grid[0], m_grid[-1])
        tte = np.clip(tte, t_grid[0], t_grid[-1])

        try:
            spline = RectBivariateSpline(m_grid, t_grid, iv_grid, kx=3, ky=3)
            iv = float(spline(moneyness, tte)[0, 0])
        except Exception:
            # Fallback: nearest-neighbour from raw points
            logger.warning("Spline interpolation failed, using nearest point")
            pts = self._surface["raw_points"]
            dists = [
                abs(p["moneyness"] - moneyness) + abs(p["tte"] - tte)
                for p in pts
            ]
            iv = pts[int(np.argmin(dists))]["iv"]

        return max(iv, 0.001)

    def fit_smile(self, chain: pd.DataFrame, spot: float) -> Dict[str, Any]:
        """Fit a cubic spline to the IV smile for a single expiration.

        Args:
            chain: DataFrame with at least 'strike' and 'impliedVolatility' (or 'iv').
            spot: current underlying price.

        Returns:
            Dict with keys: moneyness, iv, spline_coeffs, fitted_iv, residuals,
            skew_25d (approx 25-delta skew), smile_curvature.
        """
        strikes, ivs = self._chain_to_strike_iv(chain, spot)
        if len(strikes) < 4:
            raise ValueError("Need >= 4 valid IV points to fit smile")

        moneyness = strikes / spot
        sort_idx = np.argsort(moneyness)
        moneyness = moneyness[sort_idx]
        ivs = ivs[sort_idx]

        cs = CubicSpline(moneyness, ivs)
        fitted = cs(moneyness)
        residuals = ivs - fitted

        # Approximate 25-delta skew (90% moneyness vs 110%)
        m_low = max(0.90, moneyness[0])
        m_high = min(1.10, moneyness[-1])
        skew_25d = float(cs(m_low) - cs(m_high))

        # Curvature at ATM
        atm_m = 1.0
        if moneyness[0] <= atm_m <= moneyness[-1]:
            curvature = float(cs(atm_m, 2))
        else:
            curvature = 0.0

        return {
            "moneyness": moneyness.tolist(),
            "iv": ivs.tolist(),
            "fitted_iv": fitted.tolist(),
            "residuals": residuals.tolist(),
            "skew_25d": skew_25d,
            "smile_curvature": curvature,
            "spline_coeffs": cs.c.tolist(),
        }

    def detect_skew_anomalies(
        self, surface: Optional[Dict] = None, threshold: float = 2.0
    ) -> List[Dict]:
        """Find points where IV deviates significantly from the smooth surface.

        These are mispricing candidates.

        Args:
            surface: vol surface dict (uses cached if None).
            threshold: z-score threshold for flagging anomalies.

        Returns:
            List of dicts with strike, expiry, iv, fitted_iv, z_score.
        """
        surface = surface or self._surface
        if surface is None:
            raise RuntimeError("No surface available")

        raw = surface["raw_points"]
        m_grid = surface["moneyness"]
        t_grid = surface["tte"]
        iv_grid = surface["iv_grid"]

        try:
            spline = RectBivariateSpline(m_grid, t_grid, iv_grid, kx=3, ky=3)
        except Exception:
            logger.warning("Cannot fit spline for anomaly detection")
            return []

        residuals = []
        for pt in raw:
            m = np.clip(pt["moneyness"], m_grid[0], m_grid[-1])
            t = np.clip(pt["tte"], t_grid[0], t_grid[-1])
            fitted = float(spline(m, t)[0, 0])
            residuals.append(pt["iv"] - fitted)

        residuals = np.array(residuals)
        if len(residuals) < 2:
            return []

        std = np.std(residuals)
        if std < 1e-8:
            return []

        anomalies = []
        for i, pt in enumerate(raw):
            z = residuals[i] / std
            if abs(z) >= threshold:
                anomalies.append(
                    {
                        "strike": pt["strike"],
                        "expiry": pt["expiry"],
                        "moneyness": pt["moneyness"],
                        "tte": pt["tte"],
                        "iv": pt["iv"],
                        "fitted_iv": pt["iv"] - residuals[i],
                        "z_score": float(z),
                        "direction": "rich" if z > 0 else "cheap",
                    }
                )

        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        logger.info("Detected %d skew anomalies (threshold=%.1f)", len(anomalies), threshold)
        return anomalies

    def get_term_structure(
        self, symbol: str, strike_type: str = "atm"
    ) -> pd.DataFrame:
        """IV term structure across expirations.

        Args:
            symbol: ticker symbol.
            strike_type: 'atm', 'otm_put_25d', or 'otm_call_25d'.

        Returns:
            DataFrame with columns: expiry, tte, iv.
        """
        if self._surface is None or self._symbol != symbol:
            self.build_surface(symbol)

        surface = self._surface
        m_grid = surface["moneyness"]
        t_grid = surface["tte"]
        iv_grid = surface["iv_grid"]

        # Pick moneyness based on strike_type
        if strike_type == "atm":
            target_m = 1.0
        elif strike_type == "otm_put_25d":
            target_m = 0.95
        elif strike_type == "otm_call_25d":
            target_m = 1.05
        else:
            target_m = 1.0

        target_m = np.clip(target_m, m_grid[0], m_grid[-1])

        try:
            spline = RectBivariateSpline(m_grid, t_grid, iv_grid, kx=3, ky=3)
            ivs = [float(spline(target_m, t)[0, 0]) for t in t_grid]
        except Exception:
            m_idx = int(np.argmin(np.abs(m_grid - target_m)))
            ivs = iv_grid[m_idx, :].tolist()

        df = pd.DataFrame({"tte": t_grid, "iv": ivs})
        df["expiry_days"] = (df["tte"] * 365).astype(int)
        return df

    def variance_risk_premium(
        self, symbol: str, window: int = 30
    ) -> Dict[str, Any]:
        """Compare implied vol to realised vol; compute VRP metrics.

        VRP = IV_30d - RV_30d.  Signal fires when VRP > 1.5 * median(VRP).

        Args:
            symbol: ticker symbol.
            window: rolling window in calendar days.

        Returns:
            Dict with iv_current, rv_current, vrp, vrp_median,
            vrp_signal (bool), vrp_z_score, history (DataFrame).
        """
        logger.info("Computing VRP for %s (window=%d)", symbol, window)

        # Get implied vol (ATM, nearest expiry)
        if self._surface is None or self._symbol != symbol:
            self.build_surface(symbol)

        surface = self._surface
        m_grid = surface["moneyness"]
        t_grid = surface["tte"]
        iv_grid = surface["iv_grid"]

        # ATM IV for shortest expiry
        atm_idx = int(np.argmin(np.abs(m_grid - 1.0)))
        iv_current = float(iv_grid[atm_idx, 0])

        # Realised vol from history
        bars = self.history.get_bars(symbol, days=max(window * 3, 90))
        if bars is None or len(bars) < window:
            raise ValueError(f"Insufficient price history for {symbol}")

        close = bars["close"].values.astype(float)
        log_returns = np.diff(np.log(close))

        # Annualised realised vol (use trading-day window)
        trading_window = int(window * 252 / 365)
        trading_window = min(trading_window, len(log_returns))
        rv_current = float(np.std(log_returns[-trading_window:]) * np.sqrt(252))

        # Rolling VRP history
        rv_series = pd.Series(log_returns).rolling(trading_window).std() * np.sqrt(252)
        rv_series = rv_series.dropna()

        vrp = iv_current - rv_current
        vrp_median = float(rv_series.apply(lambda rv: iv_current - rv).median())
        vrp_std = float(rv_series.apply(lambda rv: iv_current - rv).std())
        vrp_z = vrp / vrp_std if vrp_std > 1e-8 else 0.0

        signal = vrp > 1.5 * abs(vrp_median) if vrp_median != 0 else False

        result = {
            "symbol": symbol,
            "iv_current": iv_current,
            "rv_current": rv_current,
            "vrp": vrp,
            "vrp_median": vrp_median,
            "vrp_z_score": vrp_z,
            "vrp_signal": bool(signal),
            "signal_description": (
                "IV significantly elevated vs RV — consider selling vol"
                if signal
                else "VRP within normal range"
            ),
            "window_days": window,
        }

        logger.info(
            "VRP for %s: IV=%.2f%% RV=%.2f%% VRP=%.2f%% signal=%s",
            symbol,
            iv_current * 100,
            rv_current * 100,
            vrp * 100,
            signal,
        )
        return result

    # ------------------------------------------------------------------
    # Overnight VRP decomposition (Papagelis & Dotsis, Apr 2025)
    # ------------------------------------------------------------------

    def overnight_vrp(self, symbol: str, window: int = 30) -> Dict[str, Any]:
        """Decompose the variance risk premium into overnight and intraday components.

        Research (Papagelis & Dotsis, Apr 2025) shows the overnight VRP is
        *negative* (IV over-estimates overnight realised vol), making overnight
        short-vol strategies profitable, while the intraday VRP is positive.

        Uses yfinance for OHLC data when ``self.history`` is unavailable.

        Args:
            symbol: ticker symbol.
            window: rolling window in calendar days.

        Returns:
            Dict with overnight_vrp, intraday_vrp, total_vrp, overnight_rv,
            intraday_rv, iv_30d, signal, signal_strength.
        """
        logger.info("Computing overnight VRP decomposition for %s (window=%d)", symbol, window)

        # --- Get IV (ATM, nearest expiry) ---
        if self._surface is None or self._symbol != symbol:
            self.build_surface(symbol)

        surface = self._surface
        m_grid = surface["moneyness"]
        iv_grid = surface["iv_grid"]
        atm_idx = int(np.argmin(np.abs(m_grid - 1.0)))
        iv_30d = float(iv_grid[atm_idx, 0])

        # --- Get OHLC data ---
        ohlc = None
        fetch_days = max(window * 3, 120)

        # Try history provider first
        if self.history is not None:
            try:
                bars = self.history.get_bars(symbol, days=fetch_days)
                if bars is not None and not bars.empty:
                    needed_cols = {"open", "close"}
                    if needed_cols.issubset({c.lower() for c in bars.columns}):
                        ohlc = bars.rename(columns={c: c.lower() for c in bars.columns})
            except Exception:
                pass

        # Fallback: yfinance
        if ohlc is None or len(ohlc) < window:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                ohlc = ticker.history(period=f"{fetch_days}d")
                ohlc.columns = [c.lower() for c in ohlc.columns]
            except ImportError:
                raise RuntimeError(
                    "yfinance is required for overnight_vrp when history provider "
                    "does not supply OHLC data. Install with: pip install yfinance"
                )

        if ohlc is None or len(ohlc) < window:
            raise ValueError(f"Insufficient OHLC data for {symbol} (got {len(ohlc) if ohlc is not None else 0} bars)")

        open_prices = ohlc["open"].values.astype(float)
        close_prices = ohlc["close"].values.astype(float)

        # Overnight return: open_t / close_{t-1} - 1
        overnight_returns = open_prices[1:] / close_prices[:-1] - 1.0
        # Intraday return: close_t / open_t - 1
        intraday_returns = close_prices[1:] / open_prices[1:] - 1.0

        trading_window = min(int(window * 252 / 365), len(overnight_returns))

        overnight_rv = float(np.std(overnight_returns[-trading_window:]) * np.sqrt(252))
        intraday_rv = float(np.std(intraday_returns[-trading_window:]) * np.sqrt(252))

        # VRP decomposition
        overnight_vrp_val = iv_30d - overnight_rv   # typically negative = profitable to sell
        intraday_vrp_val = iv_30d - intraday_rv      # typically positive = not profitable
        total_vrp = iv_30d - float(
            np.std(np.log(close_prices[1:] / close_prices[:-1])[-trading_window:]) * np.sqrt(252)
        )

        # Historical overnight VRP for signal calibration
        rolling_on_rv = pd.Series(overnight_returns).rolling(trading_window).std() * np.sqrt(252)
        rolling_on_vrp = iv_30d - rolling_on_rv.dropna()
        median_on_vrp = float(rolling_on_vrp.median()) if len(rolling_on_vrp) > 0 else overnight_vrp_val

        # Signal generation
        signal = "neutral"
        signal_strength = 0.0

        if median_on_vrp != 0:
            ratio = overnight_vrp_val / abs(median_on_vrp) if abs(median_on_vrp) > 1e-8 else 0.0
            if overnight_vrp_val < -abs(median_on_vrp) * 1.5:
                signal = "sell_overnight"
                signal_strength = min(abs(ratio) / 3.0, 1.0)
            elif overnight_vrp_val < 0:
                signal = "mild_sell_overnight"
                signal_strength = min(abs(ratio) / 2.0, 0.6)
            else:
                signal = "no_edge"
                signal_strength = 0.0

        result = {
            "symbol": symbol,
            "overnight_vrp": round(overnight_vrp_val, 6),
            "intraday_vrp": round(intraday_vrp_val, 6),
            "total_vrp": round(total_vrp, 6),
            "overnight_rv": round(overnight_rv, 6),
            "intraday_rv": round(intraday_rv, 6),
            "iv_30d": round(iv_30d, 6),
            "signal": signal,
            "signal_strength": round(signal_strength, 4),
            "window_days": window,
            "decomposition_note": (
                "Negative overnight_vrp = IV overstates overnight realised vol "
                "(profitable to sell). Based on Papagelis & Dotsis (Apr 2025)."
            ),
        }

        logger.info(
            "Overnight VRP for %s: ON_VRP=%.4f IN_VRP=%.4f signal=%s (%.2f)",
            symbol,
            overnight_vrp_val,
            intraday_vrp_val,
            signal,
            signal_strength,
        )
        return result

    def vrp_regime(self, symbol: str, window: int = 60) -> Dict[str, Any]:
        """Classify the current variance risk premium regime.

        Regimes:
            - ``premium_rich``:  VRP > 1.5 × historical median → sell vol
            - ``premium_poor``:  VRP < 0.5 × historical median → avoid selling
            - ``inverted``:      VRP < 0 → do NOT sell vol, consider buying protection
            - ``normal``:        otherwise

        Args:
            symbol: ticker symbol.
            window: lookback days for regime classification.

        Returns:
            Dict with regime, vrp, vrp_median, vrp_percentile, action.
        """
        logger.info("Computing VRP regime for %s (window=%d)", symbol, window)

        # Get current VRP data
        vrp_data = self.variance_risk_premium(symbol, window=window)
        vrp = vrp_data["vrp"]
        iv_current = vrp_data["iv_current"]
        rv_current = vrp_data["rv_current"]

        # Compute historical VRP distribution
        fetch_days = max(window * 5, 300)
        bars = self.history.get_bars(symbol, days=fetch_days)
        if bars is None or len(bars) < window:
            raise ValueError(f"Insufficient history for VRP regime ({symbol})")

        close = bars["close"].values.astype(float)
        log_returns = np.diff(np.log(close))
        trading_window = int(window * 252 / 365)
        trading_window = min(trading_window, len(log_returns))

        rv_series = pd.Series(log_returns).rolling(trading_window).std() * np.sqrt(252)
        rv_series = rv_series.dropna()

        # Approximate historical VRP as IV_current - RV_rolling
        # (conservative: uses current IV as proxy since we don't have historical IV)
        vrp_history = iv_current - rv_series
        vrp_median = float(vrp_history.median())
        vrp_std = float(vrp_history.std())

        # Percentile of current VRP in distribution
        vrp_percentile = float((vrp_history < vrp).mean() * 100)

        # Regime classification
        if vrp < 0:
            regime = "inverted"
            action = "DO NOT sell vol. Consider buying protection / long gamma."
        elif vrp_median > 0 and vrp > 1.5 * vrp_median:
            regime = "premium_rich"
            action = "Sell vol: IV substantially above RV. Favour strangles, iron condors."
        elif vrp_median > 0 and vrp < 0.5 * vrp_median:
            regime = "premium_poor"
            action = "Avoid selling vol: insufficient premium vs risk."
        else:
            regime = "normal"
            action = "Standard premium levels. Selective vol selling acceptable."

        result = {
            "symbol": symbol,
            "regime": regime,
            "vrp": round(vrp, 6),
            "iv_current": round(iv_current, 6),
            "rv_current": round(rv_current, 6),
            "vrp_median": round(vrp_median, 6),
            "vrp_percentile": round(vrp_percentile, 2),
            "vrp_z_score": round((vrp - vrp_median) / vrp_std if vrp_std > 1e-8 else 0.0, 4),
            "action": action,
            "window_days": window,
        }

        logger.info(
            "VRP regime for %s: %s (VRP=%.4f, median=%.4f, pctl=%.1f%%)",
            symbol,
            regime,
            vrp,
            vrp_median,
            vrp_percentile,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_spot(self, symbol: str) -> float:
        """Get current spot price from chains or history provider."""
        if hasattr(self.chains, "get_spot"):
            return float(self.chains.get_spot(symbol))
        if self.history is not None:
            bars = self.history.get_bars(symbol, days=2)
            if bars is not None and not bars.empty:
                return float(bars["close"].iloc[-1])
        raise ValueError(f"Cannot determine spot price for {symbol}")

    def _extract_iv(self, row: pd.Series, spot: float) -> Optional[float]:
        """Extract IV from a chain row, trying common column names."""
        for col in ("impliedVolatility", "implied_volatility", "iv", "IV"):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        # If greeks provider available, compute IV
        if self.greeks is not None and hasattr(self.greeks, "implied_volatility"):
            try:
                return self.greeks.implied_volatility(
                    option_price=float(row.get("lastPrice", row.get("mid", 0))),
                    spot=spot,
                    strike=float(row["strike"]),
                    tte=float(row.get("tte", 30 / 365)),
                    option_type=row.get("type", row.get("option_type", "call")),
                )
            except Exception:
                pass
        return None

    @staticmethod
    def _chain_to_strike_iv(
        chain: pd.DataFrame, spot: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract parallel arrays of strikes and IVs from a chain DataFrame."""
        iv_col = None
        for c in ("impliedVolatility", "implied_volatility", "iv", "IV"):
            if c in chain.columns:
                iv_col = c
                break
        if iv_col is None:
            raise ValueError("Chain DataFrame has no IV column")

        mask = chain[iv_col].notna() & (chain[iv_col] > 0) & chain["strike"].notna()
        filtered = chain.loc[mask]
        strikes = filtered["strike"].values.astype(float)
        ivs = filtered[iv_col].values.astype(float)
        return strikes, ivs
