"""Options chain data fetching and management.

Supports yfinance (free), Alpaca, and Polygon as data backends.
Chains are cached with a short TTL since options data changes frequently.
"""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Cache TTL in seconds (5 minutes default for options data)
DEFAULT_CACHE_TTL = 300


class OptionsChainManager:
    """Fetches and manages options chain data from multiple backends.

    Parameters
    ----------
    backend : str
        Data source: 'yfinance' (default/free), 'alpaca', or 'polygon'.
    cache_ttl : int
        Cache time-to-live in seconds. Default 300 (5 min).
    alpaca_api_key : str, optional
        Alpaca API key (required if backend='alpaca').
    alpaca_secret_key : str, optional
        Alpaca secret key (required if backend='alpaca').
    polygon_api_key : str, optional
        Polygon API key (required if backend='polygon').
    """

    def __init__(
        self,
        backend: str = "yfinance",
        cache_ttl: int = DEFAULT_CACHE_TTL,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        polygon_api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, object]] = {}

        # Backend-specific setup
        self._alpaca_api_key = alpaca_api_key
        self._alpaca_secret_key = alpaca_secret_key
        self._polygon_api_key = polygon_api_key

        logger.info("OptionsChainManager initialized with backend=%s", backend)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str):
        """Return cached value if present and not expired, else None."""
        if key in self._cache:
            ts, value = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return value
            del self._cache[key]
        return None

    def _cache_set(self, key: str, value):
        self._cache[key] = (time.time(), value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_expirations(self, symbol: str) -> List[str]:
        """Return available expiration dates for *symbol*.

        Returns
        -------
        List[str]
            Sorted list of expiration date strings (YYYY-MM-DD).
        """
        cache_key = f"exp:{symbol}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        expirations: List[str] = []

        if self.backend == "yfinance":
            expirations = self._yf_expirations(symbol)
        elif self.backend == "alpaca":
            expirations = self._alpaca_expirations(symbol)
        elif self.backend == "polygon":
            expirations = self._polygon_expirations(symbol)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._cache_set(cache_key, expirations)
        return expirations

    def fetch_chain(
        self, symbol: str, expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch the full options chain for *symbol*.

        Parameters
        ----------
        symbol : str
            Underlying ticker.
        expiration : str, optional
            Specific expiration date (YYYY-MM-DD). If None, uses the
            nearest expiration.

        Returns
        -------
        pd.DataFrame
            Columns: strike, lastPrice, bid, ask, volume, openInterest,
            impliedVolatility, expiration, optionType.
        """
        cache_key = f"chain:{symbol}:{expiration}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self.backend == "yfinance":
            df = self._yf_fetch_chain(symbol, expiration)
        elif self.backend == "alpaca":
            df = self._alpaca_fetch_chain(symbol, expiration)
        elif self.backend == "polygon":
            df = self._polygon_fetch_chain(symbol, expiration)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._cache_set(cache_key, df)
        return df

    def filter_chain(
        self,
        chain: pd.DataFrame,
        option_type: str = "call",
        min_volume: int = 10,
        min_oi: int = 100,
        moneyness_range: Tuple[float, float] = (-0.3, 0.3),
    ) -> pd.DataFrame:
        """Filter an options chain DataFrame.

        Parameters
        ----------
        chain : pd.DataFrame
            Chain data as returned by :meth:`fetch_chain`.
        option_type : str
            'call' or 'put'.
        min_volume : int
            Minimum volume filter.
        min_oi : int
            Minimum open-interest filter.
        moneyness_range : tuple of float
            (low, high) moneyness bounds. Requires a 'moneyness' column
            (see :meth:`calculate_moneyness`).

        Returns
        -------
        pd.DataFrame
        """
        df = chain.copy()
        df = df[df["optionType"] == option_type]
        df = df[df["volume"] >= min_volume]
        df = df[df["openInterest"] >= min_oi]

        if "moneyness" in df.columns:
            lo, hi = moneyness_range
            df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

        logger.debug(
            "filter_chain: %d rows after filtering (type=%s, vol>=%d, oi>=%d)",
            len(df),
            option_type,
            min_volume,
            min_oi,
        )
        return df.reset_index(drop=True)

    def get_atm_options(self, symbol: str, expiration: str) -> Dict:
        """Return the ATM call and put for *symbol* at *expiration*.

        Returns
        -------
        dict
            {'call': dict, 'put': dict, 'strike': float}
        """
        chain = self.fetch_chain(symbol, expiration)
        spot = self._get_spot(symbol)
        chain = self.calculate_moneyness(chain, spot)

        result: Dict = {}
        for otype in ("call", "put"):
            sub = chain[chain["optionType"] == otype].copy()
            if sub.empty:
                result[otype] = None
                continue
            idx = sub["moneyness"].abs().idxmin()
            result[otype] = sub.loc[idx].to_dict()

        strike = result.get("call", result.get("put", {}))
        result["strike"] = strike.get("strike") if isinstance(strike, dict) else None
        return result

    def get_option_by_strike(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        option_type: str = "call",
    ) -> Dict:
        """Fetch a single option contract by strike and type.

        Returns
        -------
        dict
            Contract details or empty dict if not found.
        """
        chain = self.fetch_chain(symbol, expiration)
        sub = chain[
            (chain["strike"] == strike) & (chain["optionType"] == option_type)
        ]
        if sub.empty:
            logger.warning(
                "No %s option found for %s @ strike=%.2f exp=%s",
                option_type,
                symbol,
                strike,
                expiration,
            )
            return {}
        return sub.iloc[0].to_dict()

    def calculate_moneyness(
        self, chain: pd.DataFrame, spot_price: float
    ) -> pd.DataFrame:
        """Add a ``moneyness`` column to *chain*.

        Moneyness is defined as ``(strike - spot) / spot`` so that ATM ~ 0,
        ITM calls < 0, OTM calls > 0.

        Returns
        -------
        pd.DataFrame
            Copy of *chain* with the extra column.
        """
        df = chain.copy()
        df["moneyness"] = (df["strike"] - spot_price) / spot_price
        return df

    # ------------------------------------------------------------------
    # Spot price helper
    # ------------------------------------------------------------------

    def _get_spot(self, symbol: str) -> float:
        """Return the current spot price for *symbol* via yfinance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return float(info["lastPrice"])
        except Exception:
            logger.warning("Could not fetch spot for %s, using last close", symbol)
            try:
                import yfinance as yf

                hist = yf.Ticker(symbol).history(period="1d")
                return float(hist["Close"].iloc[-1])
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to fetch spot price for {symbol}"
                ) from exc

    # ------------------------------------------------------------------
    # yfinance backend
    # ------------------------------------------------------------------

    def _yf_expirations(self, symbol: str) -> List[str]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        return sorted(list(ticker.options))

    def _yf_fetch_chain(
        self, symbol: str, expiration: Optional[str]
    ) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        if expiration is None:
            exps = list(ticker.options)
            if not exps:
                logger.warning("No options expirations found for %s", symbol)
                return self._empty_chain()
            expiration = exps[0]

        opt = ticker.option_chain(expiration)

        calls = opt.calls.copy()
        calls["optionType"] = "call"
        calls["expiration"] = expiration

        puts = opt.puts.copy()
        puts["optionType"] = "put"
        puts["expiration"] = expiration

        chain = pd.concat([calls, puts], ignore_index=True)
        return self._normalize_columns(chain)

    # ------------------------------------------------------------------
    # Alpaca backend (stub — requires alpaca-py)
    # ------------------------------------------------------------------

    def _alpaca_expirations(self, symbol: str) -> List[str]:
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionChainRequest

            client = OptionHistoricalDataClient(
                self._alpaca_api_key, self._alpaca_secret_key
            )
            req = OptionChainRequest(underlying_symbol=symbol)
            snapshots = client.get_option_chain(req)
            expirations = sorted(
                {s.expiration_date.isoformat() for s in snapshots.values()}
            )
            return expirations
        except ImportError:
            logger.error("alpaca-py not installed; falling back to yfinance")
            return self._yf_expirations(symbol)
        except Exception as exc:
            logger.error("Alpaca expirations error: %s", exc)
            return self._yf_expirations(symbol)

    def _alpaca_fetch_chain(
        self, symbol: str, expiration: Optional[str]
    ) -> pd.DataFrame:
        logger.warning("Alpaca chain fetch not fully implemented; using yfinance")
        return self._yf_fetch_chain(symbol, expiration)

    # ------------------------------------------------------------------
    # Polygon backend (stub — requires polygon-api-client)
    # ------------------------------------------------------------------

    def _polygon_expirations(self, symbol: str) -> List[str]:
        try:
            from polygon import RESTClient

            client = RESTClient(api_key=self._polygon_api_key)
            contracts = list(
                client.list_options_contracts(underlying_ticker=symbol, limit=1000)
            )
            expirations = sorted({c.expiration_date for c in contracts})
            return expirations
        except ImportError:
            logger.error("polygon-api-client not installed; falling back to yfinance")
            return self._yf_expirations(symbol)
        except Exception as exc:
            logger.error("Polygon expirations error: %s", exc)
            return self._yf_expirations(symbol)

    def _polygon_fetch_chain(
        self, symbol: str, expiration: Optional[str]
    ) -> pd.DataFrame:
        logger.warning("Polygon chain fetch not fully implemented; using yfinance")
        return self._yf_fetch_chain(symbol, expiration)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the canonical column set is present."""
        canonical = [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "volume",
            "openInterest",
            "impliedVolatility",
            "expiration",
            "optionType",
        ]
        for col in canonical:
            if col not in df.columns:
                df[col] = None
        # Fill NaN volume/OI with 0 for filtering
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["openInterest"] = (
            pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int)
        )
        return df[canonical].reset_index(drop=True)

    @staticmethod
    def _empty_chain() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "volume",
                "openInterest",
                "impliedVolatility",
                "expiration",
                "optionType",
            ]
        )
