"""Options Greeks calculation using Black-Scholes model.

Uses py_vollib when available, with a manual scipy-based fallback.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Try importing py_vollib; set flag for fallback
# ---------------------------------------------------------------------------
_HAS_VOLLIB = False
try:
    from py_vollib.black_scholes import black_scholes as _bs_price
    from py_vollib.black_scholes.greeks.analytical import (
        delta as _vol_delta,
        gamma as _vol_gamma,
        theta as _vol_theta,
        vega as _vol_vega,
        rho as _vol_rho,
    )
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _vol_iv

    _HAS_VOLLIB = True
    logger.debug("py_vollib available — using analytical Greeks")
except ImportError:
    logger.info("py_vollib not installed — using manual Black-Scholes fallback")

# ---------------------------------------------------------------------------
# Manual Black-Scholes fallback (scipy)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import norm as _norm

    _cdf = _norm.cdf
    _pdf = _norm.pdf
except ImportError:
    # Ultra-minimal fallback using math.erf
    def _cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def _bs_call_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)


def _bs_put_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


# Manual Greeks -----------------------------------------------------------

def _manual_delta(S, K, T, r, sigma, flag):
    d1 = _d1(S, K, T, r, sigma)
    if flag == "c":
        return _cdf(d1)
    return _cdf(d1) - 1.0


def _manual_gamma(S, K, T, r, sigma, _flag):
    d1 = _d1(S, K, T, r, sigma)
    return _pdf(d1) / (S * sigma * math.sqrt(T))


def _manual_theta(S, K, T, r, sigma, flag):
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * _pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if flag == "c":
        return (term1 - r * K * math.exp(-r * T) * _cdf(d2)) / 365.0
    return (term1 + r * K * math.exp(-r * T) * _cdf(-d2)) / 365.0


def _manual_vega(S, K, T, r, sigma, _flag):
    d1 = _d1(S, K, T, r, sigma)
    return S * _pdf(d1) * math.sqrt(T) / 100.0  # per 1% move


def _manual_rho(S, K, T, r, sigma, flag):
    d2 = _d2(S, K, T, r, sigma)
    if flag == "c":
        return K * T * math.exp(-r * T) * _cdf(d2) / 100.0
    return -K * T * math.exp(-r * T) * _cdf(-d2) / 100.0


# Manual IV via bisection -------------------------------------------------

def _manual_iv(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    flag: str,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Implied volatility via bisection search."""
    price_fn = _bs_call_price if flag == "c" else _bs_put_price
    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p = price_fn(S, K, T, r, mid)
        if abs(p - option_price) < tol:
            return mid
        if p > option_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ===========================================================================
# Public class
# ===========================================================================

class GreeksCalculator:
    """Compute Black-Scholes Greeks for options.

    Parameters
    ----------
    risk_free_rate : float, optional
        Annual risk-free rate. If None, attempts to fetch the 10-year
        Treasury yield via yfinance (^TNX / 100), falling back to 0.05.
    """

    def __init__(self, risk_free_rate: Optional[float] = None):
        if risk_free_rate is not None:
            self.rate = risk_free_rate
        else:
            self.rate = self._fetch_risk_free_rate()
        logger.info("GreeksCalculator initialized (r=%.4f, vollib=%s)", self.rate, _HAS_VOLLIB)

    # ------------------------------------------------------------------
    # Risk-free rate
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_risk_free_rate() -> float:
        try:
            import yfinance as yf

            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d")
            if not hist.empty:
                rate = float(hist["Close"].iloc[-1]) / 100.0
                logger.debug("Fetched risk-free rate: %.4f", rate)
                return rate
        except Exception as exc:
            logger.warning("Could not fetch ^TNX: %s — using 0.05 default", exc)
        return 0.05

    # ------------------------------------------------------------------
    # Single-contract Greeks
    # ------------------------------------------------------------------

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: Optional[float] = None,
        sigma: float = 0.2,
        option_type: str = "c",
    ) -> Dict[str, float]:
        """Compute Greeks for a single vanilla option.

        Parameters
        ----------
        S : float   Spot price.
        K : float   Strike price.
        T : float   Time to expiry in *years*.
        r : float   Risk-free rate (defaults to instance rate).
        sigma : float  Implied volatility.
        option_type : str  'c' for call, 'p' for put.

        Returns
        -------
        dict  Keys: delta, gamma, theta, vega, rho.
        """
        if r is None:
            r = self.rate
        flag = option_type[0].lower()
        if T <= 0:
            T = 1e-6  # avoid div-by-zero at expiration

        if _HAS_VOLLIB:
            try:
                return {
                    "delta": _vol_delta(flag, S, K, T, r, sigma),
                    "gamma": _vol_gamma(flag, S, K, T, r, sigma),
                    "theta": _vol_theta(flag, S, K, T, r, sigma) / 365.0,
                    "vega": _vol_vega(flag, S, K, T, r, sigma) / 100.0,
                    "rho": _vol_rho(flag, S, K, T, r, sigma) / 100.0,
                }
            except Exception as exc:
                logger.debug("py_vollib error, falling back: %s", exc)

        return {
            "delta": _manual_delta(S, K, T, r, sigma, flag),
            "gamma": _manual_gamma(S, K, T, r, sigma, flag),
            "theta": _manual_theta(S, K, T, r, sigma, flag),
            "vega": _manual_vega(S, K, T, r, sigma, flag),
            "rho": _manual_rho(S, K, T, r, sigma, flag),
        }

    # ------------------------------------------------------------------
    # Batch Greeks on a chain DataFrame
    # ------------------------------------------------------------------

    def batch_greeks(
        self,
        chain: pd.DataFrame,
        spot: float,
        rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add Greek columns to an options chain DataFrame.

        Expects columns: strike, impliedVolatility, expiration, optionType.
        Adds: delta, gamma, theta, vega, rho.

        Parameters
        ----------
        chain : pd.DataFrame
        spot : float   Current underlying price.
        rate : float   Risk-free rate override.

        Returns
        -------
        pd.DataFrame  Copy with added Greek columns.
        """
        if rate is None:
            rate = self.rate

        df = chain.copy()
        greeks = {"delta": [], "gamma": [], "theta": [], "vega": [], "rho": []}

        for _, row in df.iterrows():
            T = self._years_to_expiry(row.get("expiration"))
            sigma = float(row.get("impliedVolatility", 0.2) or 0.2)
            K = float(row["strike"])
            flag = "c" if row.get("optionType", "call") == "call" else "p"

            g = self.calculate_greeks(spot, K, T, rate, sigma, flag)
            for k in greeks:
                greeks[k].append(g[k])

        for k, vals in greeks.items():
            df[k] = vals
        return df

    # ------------------------------------------------------------------
    # Implied Volatility
    # ------------------------------------------------------------------

    def calculate_iv(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: Optional[float] = None,
        option_type: str = "c",
    ) -> float:
        """Compute implied volatility from an observed option price.

        Parameters
        ----------
        option_price : float
        S, K, T, r : float  Standard BS inputs.
        option_type : str  'c' or 'p'.

        Returns
        -------
        float  Implied volatility.
        """
        if r is None:
            r = self.rate
        flag = option_type[0].lower()
        if T <= 0:
            T = 1e-6

        if _HAS_VOLLIB:
            try:
                return _vol_iv(option_price, S, K, T, r, flag)
            except Exception as exc:
                logger.debug("py_vollib IV error, falling back: %s", exc)

        return _manual_iv(option_price, S, K, T, r, flag)

    # ------------------------------------------------------------------
    # Greek surface
    # ------------------------------------------------------------------

    def greek_surface(
        self,
        chain: pd.DataFrame,
        spot: float,
        rate: Optional[float] = None,
    ) -> Dict:
        """Build a Greek surface organized by strike and expiration.

        Returns
        -------
        dict
            Structure: {expiration: {strike: {delta, gamma, ...}}}
        """
        df = self.batch_greeks(chain, spot, rate)
        surface: Dict = {}
        for _, row in df.iterrows():
            exp = str(row.get("expiration", "unknown"))
            strike = float(row["strike"])
            otype = row.get("optionType", "call")
            key = f"{otype}_{strike}"
            surface.setdefault(exp, {})[key] = {
                "strike": strike,
                "optionType": otype,
                "delta": row["delta"],
                "gamma": row["gamma"],
                "theta": row["theta"],
                "vega": row["vega"],
                "rho": row["rho"],
            }
        return surface

    # ------------------------------------------------------------------
    # Portfolio / position-level Greeks
    # ------------------------------------------------------------------

    def position_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """Aggregate Greeks across a list of option positions.

        Parameters
        ----------
        positions : list of dict
            Each dict must contain: S, K, T, sigma, option_type, quantity.
            Optional: r (defaults to instance rate).

        Returns
        -------
        dict  Aggregated delta, gamma, theta, vega, rho.
        """
        totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for pos in positions:
            qty = pos.get("quantity", 1)
            multiplier = pos.get("multiplier", 100)  # standard option = 100 shares
            g = self.calculate_greeks(
                S=pos["S"],
                K=pos["K"],
                T=pos["T"],
                r=pos.get("r"),
                sigma=pos["sigma"],
                option_type=pos.get("option_type", "c"),
            )
            for k in totals:
                totals[k] += g[k] * qty * multiplier
        return totals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _years_to_expiry(expiration) -> float:
        """Convert an expiration date string to years from now."""
        if expiration is None:
            return 30 / 365.0  # ~1 month default
        try:
            from datetime import datetime, date

            if isinstance(expiration, (datetime, date)):
                exp_date = expiration if isinstance(expiration, date) else expiration.date()
            else:
                exp_date = datetime.strptime(str(expiration)[:10], "%Y-%m-%d").date()
            delta = (exp_date - date.today()).days
            return max(delta, 1) / 365.0
        except Exception:
            return 30 / 365.0
