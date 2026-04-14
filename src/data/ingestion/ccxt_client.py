"""CCXT-based market data client — free, keyless, 100+ exchanges.

Drop-in replacement for AlpacaDataClient. Uses CCXT's unified API to pull
OHLCV data from Binance, Bybit, OKX, Kraken, Coinbase, etc. No API key
needed for public market data.

Install: pip install ccxt
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...utils.logging import get_logger

logger = get_logger(__name__)

# Map rlaif-trader timeframe strings to CCXT timeframe strings
_TIMEFRAME_MAP = {
    "1Min": "1m",
    "1min": "1m",
    "5Min": "5m",
    "5min": "5m",
    "15Min": "15m",
    "15min": "15m",
    "1Hour": "1h",
    "1hour": "1h",
    "1h": "1h",
    "4Hour": "4h",
    "4hour": "4h",
    "4h": "4h",
    "1Day": "1d",
    "1day": "1d",
    "1d": "1d",
    "1Week": "1w",
    "1week": "1w",
    "1w": "1w",
}

# Approximate candles per day for each timeframe (for pagination math)
_CANDLES_PER_DAY = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1h": 24,
    "4h": 6,
    "1d": 1,
    "1w": 1 / 7,
}


class CCXTDataClient:
    """Unified crypto market data client via CCXT.

    Drop-in compatible with AlpacaDataClient — same method signatures,
    same return format (MultiIndex DataFrame with OHLCV columns).

    Parameters
    ----------
    exchange : str
        Exchange name: binance, bybit, okx, kraken, coinbase, etc.
    cache_dir : str or Path, optional
        Directory for parquet cache. Default: data/cache/ccxt
    api_key, secret_key : str, optional
        Only needed for private endpoints (not needed for OHLCV data).
    """

    def __init__(
        self,
        exchange: str = "kraken",
        cache_dir: Optional[Union[str, Path]] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt not installed. Run: pip install ccxt")

        self.exchange_id = exchange.lower()
        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(
                f"Unknown exchange: {exchange}. "
                f"Available: {', '.join(ccxt.exchanges[:20])}..."
            )

        config: Dict[str, Any] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if secret_key:
            config["secret"] = secret_key

        self.exchange = exchange_class(config)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/ccxt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "CCXTDataClient initialised (exchange=%s, markets=%s)",
            self.exchange_id,
            "loading...",
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Accept both 'BTCUSDT' and 'BTC/USDT' formats."""
        if "/" in symbol:
            return symbol.upper()
        # Common USDT pairs
        for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH", "USD"):
            if symbol.upper().endswith(quote) and len(symbol) > len(quote):
                base = symbol.upper()[: -len(quote)]
                return f"{base}/{quote}"
        return symbol.upper()

    def _to_ccxt_timeframe(self, timeframe: str) -> str:
        tf = _TIMEFRAME_MAP.get(timeframe, timeframe)
        if tf not in self.exchange.timeframes:
            available = list(self.exchange.timeframes.keys())
            logger.warning(
                "Timeframe %s not available on %s, using 1d. Available: %s",
                tf, self.exchange_id, available,
            )
            tf = "1d"
        return tf

    def _cache_key(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        """Generate a deterministic cache path.

        For daily timeframes, we cache by month chunks to maximize reuse.
        For intraday, we cache by the exact date range.
        """
        safe_sym = symbol.replace("/", "_")
        if timeframe in ("1d", "1w"):
            # Cache the whole start-end range (most common pattern)
            key = f"{self.exchange_id}_{safe_sym}_{timeframe}_{start}_{end}"
        else:
            key = f"{self.exchange_id}_{safe_sym}_{timeframe}_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{safe_sym}_{timeframe}_{h}.parquet"

    def download_bars(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        timeframe: str = "1Day",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download OHLCV bars for one or more symbols.

        Parameters
        ----------
        symbols : str or list of str
            Trading pair(s): "BTC/USDT", "BTCUSDT", or ["BTC/USDT", "ETH/USDT"]
        start, end : str or datetime
            Date range. Strings as "YYYY-MM-DD".
        timeframe : str
            Candle size: "1Min", "5Min", "1Hour", "1Day", etc.
        use_cache : bool
            If True, check parquet cache before hitting the API.

        Returns
        -------
        pd.DataFrame
            MultiIndex (symbol, timestamp) with columns [open, high, low, close, volume].
            Same format as AlpacaDataClient.download_bars().
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        start_str = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
        end_str = end if isinstance(end, str) else end.strftime("%Y-%m-%d")
        ccxt_tf = self._to_ccxt_timeframe(timeframe)

        all_frames = []
        for raw_symbol in symbols:
            symbol = self._normalize_symbol(raw_symbol)
            cache_path = self._cache_key(symbol, ccxt_tf, start_str, end_str)

            # Check cache
            if use_cache and cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    logger.debug("Cache hit: %s (%d rows)", cache_path.name, len(df))
                    all_frames.append(df)
                    continue
                except Exception:
                    pass

            # Fetch from exchange
            df = self._fetch_ohlcv(symbol, ccxt_tf, start_str, end_str)
            if df.empty:
                logger.warning("No data returned for %s", symbol)
                continue

            # Add symbol column for MultiIndex
            df["symbol"] = symbol

            # Cache
            if use_cache:
                try:
                    df.to_parquet(cache_path)
                    logger.debug("Cached: %s (%d rows)", cache_path.name, len(df))
                except Exception as e:
                    logger.debug("Cache write failed: %s", e)

            all_frames.append(df)

        if not all_frames:
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            empty.index = pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"])
            return empty

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.set_index(["symbol", "timestamp"]).sort_index()
        return combined

    def download_latest(
        self,
        symbols: Union[str, List[str]],
        days: int = 365,
        timeframe: str = "1Day",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download the most recent N days of data.

        Convenience wrapper around download_bars().
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        return self.download_bars(
            symbols=symbols,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            timeframe=timeframe,
            use_cache=use_cache,
        )

    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic info about a trading pair."""
        sym = self._normalize_symbol(symbol)
        try:
            self.exchange.load_markets()
            market = self.exchange.market(sym)
            return {
                "symbol": market["symbol"],
                "exchange": self.exchange_id,
                "asset_class": "crypto",
                "base": market.get("base", ""),
                "quote": market.get("quote", ""),
                "active": market.get("active", True),
                "precision_price": market.get("precision", {}).get("price"),
                "precision_amount": market.get("precision", {}).get("amount"),
                "min_amount": market.get("limits", {}).get("amount", {}).get("min"),
            }
        except Exception as e:
            logger.warning("get_asset_info failed for %s: %s", symbol, e)
            return {"symbol": sym, "exchange": self.exchange_id, "asset_class": "crypto"}

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the current ticker (price, bid, ask, volume)."""
        sym = self._normalize_symbol(symbol)
        try:
            ticker = self.exchange.fetch_ticker(sym)
            return {
                "symbol": sym,
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "volume": ticker.get("baseVolume"),
                "change_pct": ticker.get("percentage"),
                "timestamp": ticker.get("datetime"),
            }
        except Exception as e:
            logger.warning("get_ticker failed for %s: %s", symbol, e)
            return None

    def list_symbols(self, quote: str = "USDT") -> List[str]:
        """List available trading pairs for a quote currency."""
        try:
            self.exchange.load_markets()
            return sorted([
                s for s in self.exchange.symbols
                if s.endswith(f"/{quote}")
            ])
        except Exception as e:
            logger.warning("list_symbols failed: %s", e)
            return []

    # ── Internal ────────────────────────────────────────────────────────

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_str: str,
        end_str: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV with automatic pagination for large date ranges."""
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        since_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_candles = []
        limit = 1000  # CCXT default max per request
        fetch_count = 0
        max_fetches = 500  # safety cap

        logger.info(
            "Fetching %s %s from %s to %s on %s",
            symbol, timeframe, start_str, end_str, self.exchange_id,
        )

        while since_ms < end_ms and fetch_count < max_fetches:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except Exception as e:
                logger.warning("fetch_ohlcv error for %s: %s", symbol, e)
                break

            if not candles:
                break

            fetch_count += 1
            all_candles.extend(candles)

            # Advance since to after the last candle
            last_ts = candles[-1][0]
            if last_ts <= since_ms:
                break  # no progress, avoid infinite loop
            since_ms = last_ts + 1

            # Small sleep to be polite even with rate limiter
            if fetch_count % 10 == 0:
                time.sleep(0.5)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )

        # Deduplicate (pagination overlap)
        df = df.drop_duplicates(subset=["timestamp_ms"], keep="last")

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp_ms"])

        # Filter to requested range
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Fetched %d candles for %s (%d API calls)",
            len(df), symbol, fetch_count,
        )

        return df
