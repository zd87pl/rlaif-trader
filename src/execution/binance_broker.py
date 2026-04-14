"""Binance broker implementation for 24/7 crypto trading.

Implements BrokerInterface using the python-binance library.
Supports spot and futures trading. 24/7 markets mean more experiment
cycles per day for the autotrader loop.

Install: pip install python-binance
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .broker import BrokerInterface
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BinanceBroker(BrokerInterface):
    """Broker implementation for Binance spot and futures.

    Parameters
    ----------
    api_key : str
        Binance API key. Falls back to BINANCE_API_KEY env var.
    api_secret : str
        Binance API secret. Falls back to BINANCE_API_SECRET env var.
    testnet : bool
        Use Binance testnet (paper trading equivalent). Default True.
    futures : bool
        Use USDT-M futures instead of spot. Default False.
    """

    SPOT_BASE = "https://api.binance.com"
    SPOT_TESTNET = "https://testnet.binance.vision"
    FUTURES_BASE = "https://fapi.binance.com"
    FUTURES_TESTNET = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        futures: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.testnet = testnet
        self.futures = futures
        self._client = None
        self._connected = False

        logger.info(
            "BinanceBroker initialised (testnet=%s, futures=%s)",
            self.testnet,
            self.futures,
        )

    # ── connection ──────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            from binance.client import Client

            self._client = Client(
                self.api_key,
                self.api_secret,
                testnet=self.testnet,
            )
            # Verify connectivity
            self._client.ping()
            account = self._client.get_account()
            self._connected = True
            logger.info("BinanceBroker connected (testnet=%s)", self.testnet)
            return True
        except ImportError:
            logger.error(
                "python-binance not installed. Run: pip install python-binance"
            )
            return False
        except Exception as exc:
            logger.error("BinanceBroker connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._client = None
        self._connected = False
        logger.info("BinanceBroker disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── account / positions ─────────────────────────────────────────────

    def get_account(self) -> Dict[str, Any]:
        if self.futures:
            account = self._client.futures_account()
            return {
                "balance": float(account.get("totalWalletBalance", 0)),
                "buying_power": float(account.get("availableBalance", 0)),
                "equity": float(account.get("totalMarginBalance", 0)),
                "unrealized_pnl": float(account.get("totalUnrealizedProfit", 0)),
                "positions": len([
                    p for p in account.get("positions", [])
                    if float(p.get("positionAmt", 0)) != 0
                ]),
            }

        account = self._client.get_account()
        balances = account.get("balances", [])
        # Sum USDT equivalent
        total_balance = 0.0
        for bal in balances:
            free = float(bal.get("free", 0))
            locked = float(bal.get("locked", 0))
            if bal["asset"] == "USDT":
                total_balance += free + locked

        return {
            "balance": total_balance,
            "buying_power": total_balance,
            "equity": total_balance,
            "positions": len([
                b for b in balances
                if float(b.get("free", 0)) + float(b.get("locked", 0)) > 0
                and b["asset"] != "USDT"
            ]),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        if self.futures:
            account = self._client.futures_account()
            positions = []
            for p in account.get("positions", []):
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                positions.append({
                    "symbol": p["symbol"],
                    "qty": amt,
                    "side": "long" if amt > 0 else "short",
                    "avg_entry_price": float(p.get("entryPrice", 0)),
                    "market_value": abs(amt) * float(p.get("markPrice", 0)),
                    "unrealized_pl": float(p.get("unrealizedProfit", 0)),
                    "leverage": int(p.get("leverage", 1)),
                })
            return positions

        # Spot: positions are non-zero balances
        account = self._client.get_account()
        positions = []
        for bal in account.get("balances", []):
            free = float(bal.get("free", 0))
            locked = float(bal.get("locked", 0))
            total = free + locked
            if total > 0 and bal["asset"] not in ("USDT", "BUSD", "USDC"):
                symbol = bal["asset"] + "USDT"
                try:
                    ticker = self._client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker["price"])
                except Exception:
                    price = 0.0
                positions.append({
                    "symbol": symbol,
                    "qty": total,
                    "side": "long",
                    "avg_entry_price": 0.0,  # Spot doesn't track entry
                    "market_value": total * price,
                    "unrealized_pl": 0.0,
                })
        return positions

    # ── orders ──────────────────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        from binance.enums import (
            SIDE_BUY, SIDE_SELL,
            ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT,
            ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_STOP_MARKET,
            TIME_IN_FORCE_GTC,
        )

        bn_side = SIDE_BUY if side.lower() == "buy" else SIDE_SELL

        kwargs: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": bn_side,
            "quantity": self._format_qty(symbol, qty),
        }

        if order_type.lower() == "market":
            kwargs["type"] = ORDER_TYPE_MARKET
        elif order_type.lower() == "limit":
            kwargs["type"] = ORDER_TYPE_LIMIT
            kwargs["price"] = str(limit_price)
            kwargs["timeInForce"] = TIME_IN_FORCE_GTC
        elif order_type.lower() in ("stop", "stop_market"):
            if self.futures:
                kwargs["type"] = ORDER_TYPE_STOP_MARKET
                kwargs["stopPrice"] = str(stop_price)
            else:
                kwargs["type"] = ORDER_TYPE_STOP_LOSS_LIMIT
                kwargs["price"] = str(limit_price or stop_price)
                kwargs["stopPrice"] = str(stop_price)
                kwargs["timeInForce"] = TIME_IN_FORCE_GTC

        try:
            if self.futures:
                order = self._client.futures_create_order(**kwargs)
            else:
                order = self._client.create_order(**kwargs)

            logger.info(
                "Binance order submitted: %s %s %s @ %s",
                side, qty, symbol, order_type,
            )
            return {
                "order_id": str(order.get("orderId", "")),
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "order_type": order_type,
                "status": order.get("status", "NEW"),
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "fills": order.get("fills", []),
            }
        except Exception as exc:
            logger.error("Binance order failed: %s", exc)
            return {
                "order_id": "",
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "status": "rejected",
                "error": str(exc),
            }

    def submit_option_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        option_symbol: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Binance options (VOPTIONS) have limited support
        # For now, route to regular order on the underlying
        logger.warning(
            "Binance options not fully supported, routing as spot order on %s",
            symbol,
        )
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
        )

    def submit_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Binance doesn't support multi-leg orders natively
        # Execute each leg individually
        group_id = uuid.uuid4().hex[:8]
        results = []
        for i, leg in enumerate(legs):
            result = self.submit_order(
                symbol=leg.get("symbol", leg.get("option_symbol", "")),
                qty=leg["qty"],
                side=leg["side"],
                order_type=order_type,
                limit_price=limit_price,
            )
            result["group_id"] = group_id
            result["leg_index"] = i
            results.append(result)
        return {"group_id": group_id, "legs": results, "status": "submitted"}

    # ── order management ────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        try:
            # Need symbol to cancel on Binance -- try open orders
            if self.futures:
                open_orders = self._client.futures_get_open_orders()
            else:
                open_orders = self._client.get_open_orders()

            for o in open_orders:
                if str(o.get("orderId")) == order_id:
                    symbol = o["symbol"]
                    if self.futures:
                        self._client.futures_cancel_order(
                            symbol=symbol, orderId=int(order_id)
                        )
                    else:
                        self._client.cancel_order(
                            symbol=symbol, orderId=int(order_id)
                        )
                    logger.info("Binance order cancelled: %s", order_id)
                    return True

            logger.warning("Order %s not found in open orders", order_id)
            return False
        except Exception as exc:
            logger.error("Failed to cancel Binance order %s: %s", order_id, exc)
            return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        # Binance requires symbol to query order -- return basic info
        return {"order_id": order_id, "status": "unknown"}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            if self.futures:
                orders = self._client.futures_get_open_orders()
            else:
                orders = self._client.get_open_orders()

            return [
                {
                    "order_id": str(o.get("orderId")),
                    "symbol": o.get("symbol", ""),
                    "qty": float(o.get("origQty", 0)),
                    "side": o.get("side", "").lower(),
                    "status": o.get("status", ""),
                }
                for o in orders
            ]
        except Exception as exc:
            logger.error("Failed to get Binance open orders: %s", exc)
            return []

    # ── market data ─────────────────────────────────────────────────────

    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        logger.warning("Binance option chains not yet supported")
        return pd.DataFrame()

    def is_market_open(self) -> bool:
        # Crypto markets are 24/7
        return True

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Deprecated: use CCXTDataClient for market data instead.

        Kept for backward compatibility. For new code, use:
            from src.data.ingestion.ccxt_client import CCXTDataClient
            client = CCXTDataClient(exchange="binance")
            client.download_bars("BTC/USDT", start, end, "1Day")
        """
        logger.warning("BinanceBroker.get_klines() is deprecated — use CCXTDataClient")
        try:
            if self.futures:
                klines = self._client.futures_klines(
                    symbol=symbol, interval=interval, limit=limit
                )
            else:
                klines = self._client.get_klines(
                    symbol=symbol, interval=interval, limit=limit
                )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.set_index("timestamp")
            return df[["open", "high", "low", "close", "volume"]]

        except Exception as exc:
            logger.error("Failed to fetch Binance klines: %s", exc)
            return pd.DataFrame()

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception:
            return None

    # ── helpers ──────────────────────────────────────────────────────────

    def _format_qty(self, symbol: str, qty: float) -> str:
        """Format quantity to Binance's precision requirements."""
        try:
            info = self._client.get_symbol_info(symbol.upper())
            if info:
                for f in info.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                        precision = len(f["stepSize"].rstrip("0").split(".")[-1])
                        adjusted = round(qty - (qty % step), precision)
                        return str(adjusted)
        except Exception:
            pass
        return str(round(qty, 6))

    def flatten_all(self) -> Dict[str, Any]:
        """Close all positions (spot: sell all, futures: close positions)."""
        positions = self.get_positions()
        results = []
        for pos in positions:
            close_side = "sell" if pos["qty"] > 0 else "buy"
            result = self.submit_order(
                symbol=pos["symbol"],
                qty=abs(pos["qty"]),
                side=close_side,
                order_type="market",
            )
            results.append(result)
        return {"closed": len(results), "results": results}
