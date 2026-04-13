"""Broker interfaces and implementations for trade execution."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BrokerInterface(ABC):
    """Abstract broker interface that all broker implementations must follow."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        ...

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Return account info: balance, buying_power, equity, positions."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Return current open positions."""
        ...

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Submit a stock/equity order. Returns order receipt dict."""
        ...

    @abstractmethod
    def submit_option_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        option_symbol: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit a single-leg option order."""
        ...

    @abstractmethod
    def submit_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit a multi-leg option order (spreads, straddles, etc.)."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation accepted."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get the status / details of a specific order."""
        ...

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Return all currently open (pending) orders."""
        ...

    @abstractmethod
    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Fetch an option chain for *symbol* at a given *expiration* date."""
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check whether the market is currently open for trading."""
        ...


# ---------------------------------------------------------------------------
# Alpaca implementation
# ---------------------------------------------------------------------------

class AlpacaBroker(BrokerInterface):
    """Broker implementation using the alpaca-py SDK.

    Supports paper and live trading.  Paper mode is the default unless
    ``paper=False`` is explicitly passed.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._trading_client = None
        self._options_client = None
        self._data_client = None
        logger.info(
            "AlpacaBroker initialised (paper=%s)",
            self.paper,
        )

    # -- connection -----------------------------------------------------------

    def connect(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                self.api_key,
                self.secret_key,
                paper=self.paper,
            )
            # Verify connectivity
            self._trading_client.get_account()
            logger.info("AlpacaBroker connected successfully")
            return True
        except Exception as exc:
            logger.error("AlpacaBroker connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._trading_client = None
        self._options_client = None
        self._data_client = None
        logger.info("AlpacaBroker disconnected")

    # -- account / positions --------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        acct = self._trading_client.get_account()
        return {
            "balance": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "positions": len(self.get_positions()),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        positions = self._trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    # -- equity orders --------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = getattr(TimeInForce, time_in_force.upper(), TimeInForce.DAY)

        if order_type.lower() == "market":
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side_enum, time_in_force=tif,
            )
        elif order_type.lower() == "limit":
            req = LimitOrderRequest(
                symbol=symbol, qty=qty, side=side_enum,
                time_in_force=tif, limit_price=limit_price,
            )
        elif order_type.lower() == "stop":
            req = StopOrderRequest(
                symbol=symbol, qty=qty, side=side_enum,
                time_in_force=tif, stop_price=stop_price,
            )
        elif order_type.lower() == "stop_limit":
            req = StopLimitOrderRequest(
                symbol=symbol, qty=qty, side=side_enum,
                time_in_force=tif, limit_price=limit_price,
                stop_price=stop_price,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        order = self._trading_client.submit_order(req)
        logger.info("Alpaca order submitted: %s %s %s @ %s", side, qty, symbol, order_type)
        return {
            "order_id": str(order.id),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }

    # -- option orders --------------------------------------------------------

    def submit_option_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        option_symbol: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_enum = OrderSide.BUY if side.lower() in ("buy", "buy_to_open", "buy_to_close") else OrderSide.SELL

        if order_type.lower() == "limit" and limit_price is not None:
            req = LimitOrderRequest(
                symbol=option_symbol, qty=qty, side=side_enum,
                time_in_force=TimeInForce.DAY, limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=option_symbol, qty=qty, side=side_enum,
                time_in_force=TimeInForce.DAY,
            )

        order = self._trading_client.submit_order(req)
        logger.info("Alpaca option order submitted: %s %s %s", side, qty, option_symbol)
        return {
            "order_id": str(order.id),
            "symbol": symbol,
            "option_symbol": option_symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }

    def submit_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Alpaca multi-leg: submit as individual orders grouped by client_order_id
        group_id = str(uuid.uuid4())[:8]
        results: List[Dict[str, Any]] = []
        for i, leg in enumerate(legs):
            result = self.submit_option_order(
                symbol=leg.get("symbol", ""),
                qty=leg["qty"],
                side=leg["side"],
                order_type=order_type,
                option_symbol=leg["option_symbol"],
                limit_price=limit_price,
            )
            result["group_id"] = group_id
            result["leg_index"] = i
            results.append(result)
        logger.info("Alpaca multi-leg order submitted: %d legs, group=%s", len(legs), group_id)
        return {"group_id": group_id, "legs": results, "status": "submitted"}

    # -- order management -----------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("Alpaca order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel Alpaca order %s: %s", order_id, exc)
            return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        order = self._trading_client.get_order_by_id(order_id)
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": str(order.side),
            "order_type": str(order.order_type),
            "status": str(order.status),
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def get_open_orders(self) -> List[Dict[str, Any]]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self._trading_client.get_orders(req)
        return [
            {
                "order_id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty),
                "side": str(o.side),
                "status": str(o.status),
            }
            for o in orders
        ]

    # -- market data helpers --------------------------------------------------

    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.requests import OptionChainRequest

            if self._data_client is None:
                self._data_client = OptionHistoricalDataClient(
                    self.api_key, self.secret_key,
                )
            req = OptionChainRequest(underlying_symbol=symbol, expiration_date=expiration)
            chain = self._data_client.get_option_chain(req)
            rows = []
            for sym, snapshot in chain.items():
                rows.append({
                    "option_symbol": sym,
                    "bid": snapshot.latest_quote.bid_price if snapshot.latest_quote else None,
                    "ask": snapshot.latest_quote.ask_price if snapshot.latest_quote else None,
                    "last": snapshot.latest_trade.price if snapshot.latest_trade else None,
                    "volume": snapshot.latest_trade.size if snapshot.latest_trade else 0,
                })
            return pd.DataFrame(rows)
        except Exception as exc:
            logger.error("Failed to fetch Alpaca option chain: %s", exc)
            return pd.DataFrame()

    def is_market_open(self) -> bool:
        clock = self._trading_client.get_clock()
        return clock.is_open


# ---------------------------------------------------------------------------
# Tradier implementation
# ---------------------------------------------------------------------------

class TradierBroker(BrokerInterface):
    """Broker implementation using the Tradier REST API.

    Supports sandbox (default) and production modes with full options
    multi-leg support.
    """

    SANDBOX_URL = "https://sandbox.tradier.com/v1"
    PRODUCTION_URL = "https://api.tradier.com/v1"

    def __init__(
        self,
        access_token: str,
        sandbox: bool = True,
        account_id: Optional[str] = None,
    ) -> None:
        self.access_token = access_token
        self.sandbox = sandbox
        self.account_id = account_id
        self.base_url = self.SANDBOX_URL if sandbox else self.PRODUCTION_URL
        self._session = None
        logger.info("TradierBroker initialised (sandbox=%s)", self.sandbox)

    # -- helpers --------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        import requests

        resp = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params or {},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: Optional[Dict] = None) -> Any:
        import requests

        resp = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            data=data or {},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # -- connection -----------------------------------------------------------

    def connect(self) -> bool:
        try:
            profile = self._get("/user/profile")
            if self.account_id is None:
                accounts = profile.get("profile", {}).get("account", [])
                if isinstance(accounts, dict):
                    accounts = [accounts]
                if accounts:
                    self.account_id = accounts[0].get("account_number")
            logger.info("TradierBroker connected (account=%s)", self.account_id)
            return True
        except Exception as exc:
            logger.error("TradierBroker connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        self._session = None
        logger.info("TradierBroker disconnected")

    # -- account / positions --------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        data = self._get(f"/accounts/{self.account_id}/balances")
        bal = data.get("balances", {})
        return {
            "balance": float(bal.get("total_cash", 0)),
            "buying_power": float(bal.get("option_buying_power", bal.get("stock_buying_power", 0))),
            "equity": float(bal.get("total_equity", 0)),
            "positions": len(self.get_positions()),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        data = self._get(f"/accounts/{self.account_id}/positions")
        positions = data.get("positions", {})
        if positions == "null" or not positions:
            return []
        pos_list = positions.get("position", [])
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        return [
            {
                "symbol": p.get("symbol"),
                "qty": float(p.get("quantity", 0)),
                "side": "long" if float(p.get("quantity", 0)) > 0 else "short",
                "avg_entry_price": float(p.get("cost_basis", 0)) / max(abs(float(p.get("quantity", 1))), 1),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0) if p.get("unrealized_pl") else 0),
            }
            for p in pos_list
        ]

    # -- equity orders --------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "class": "equity",
            "symbol": symbol,
            "quantity": int(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "duration": time_in_force.lower(),
        }
        if limit_price is not None:
            payload["price"] = limit_price
        if stop_price is not None:
            payload["stop"] = stop_price

        data = self._post(f"/accounts/{self.account_id}/orders", data=payload)
        order_info = data.get("order", {})
        order_id = str(order_info.get("id", ""))
        logger.info("Tradier order submitted: %s %s %s (id=%s)", side, qty, symbol, order_id)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": order_info.get("status", "pending"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

    # -- option orders --------------------------------------------------------

    def submit_option_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        option_symbol: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "class": "option",
            "symbol": symbol,
            "option_symbol": option_symbol,
            "quantity": int(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "duration": "day",
        }
        if limit_price is not None:
            payload["price"] = limit_price

        data = self._post(f"/accounts/{self.account_id}/orders", data=payload)
        order_info = data.get("order", {})
        order_id = str(order_info.get("id", ""))
        logger.info("Tradier option order submitted: %s %s %s (id=%s)", side, qty, option_symbol, order_id)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "option_symbol": option_symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": order_info.get("status", "pending"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

    def submit_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "class": "multileg",
            "symbol": legs[0].get("symbol", ""),
            "type": order_type.lower(),
            "duration": "day",
        }
        if limit_price is not None:
            payload["price"] = limit_price
        for i, leg in enumerate(legs):
            payload[f"option_symbol[{i}]"] = leg["option_symbol"]
            payload[f"quantity[{i}]"] = int(leg["qty"])
            payload[f"side[{i}]"] = leg["side"].lower()

        data = self._post(f"/accounts/{self.account_id}/orders", data=payload)
        order_info = data.get("order", {})
        order_id = str(order_info.get("id", ""))
        logger.info("Tradier multi-leg order submitted: %d legs (id=%s)", len(legs), order_id)
        return {
            "order_id": order_id,
            "legs": legs,
            "order_type": order_type,
            "status": order_info.get("status", "pending"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

    # -- order management -----------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        import requests

        try:
            resp = requests.delete(
                f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}",
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("Tradier order cancelled: %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel Tradier order %s: %s", order_id, exc)
            return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        data = self._get(f"/accounts/{self.account_id}/orders/{order_id}")
        o = data.get("order", {})
        return {
            "order_id": str(o.get("id", order_id)),
            "symbol": o.get("symbol", ""),
            "qty": float(o.get("quantity", 0)),
            "side": o.get("side", ""),
            "order_type": o.get("type", ""),
            "status": o.get("status", "unknown"),
            "filled_qty": float(o.get("exec_quantity", 0)),
            "filled_avg_price": float(o.get("avg_fill_price", 0)) if o.get("avg_fill_price") else None,
        }

    def get_open_orders(self) -> List[Dict[str, Any]]:
        data = self._get(f"/accounts/{self.account_id}/orders")
        orders = data.get("orders", {})
        if not orders or orders == "null":
            return []
        order_list = orders.get("order", [])
        if isinstance(order_list, dict):
            order_list = [order_list]
        return [
            {
                "order_id": str(o.get("id")),
                "symbol": o.get("symbol", ""),
                "qty": float(o.get("quantity", 0)),
                "side": o.get("side", ""),
                "status": o.get("status", ""),
            }
            for o in order_list
            if o.get("status") in ("pending", "open", "partially_filled")
        ]

    # -- market data helpers --------------------------------------------------

    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        try:
            data = self._get(
                "/markets/options/chains",
                params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
            )
            options = data.get("options", {}).get("option", [])
            if isinstance(options, dict):
                options = [options]
            return pd.DataFrame(options) if options else pd.DataFrame()
        except Exception as exc:
            logger.error("Failed to fetch Tradier option chain: %s", exc)
            return pd.DataFrame()

    def is_market_open(self) -> bool:
        try:
            data = self._get("/markets/clock")
            return data.get("clock", {}).get("state", "").lower() == "open"
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Paper (simulated) broker
# ---------------------------------------------------------------------------

class PaperBroker(BrokerInterface):
    """Simulated broker for testing – no real API calls.

    Tracks positions, fills orders at last price + configurable slippage.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_bps = slippage_bps  # basis points
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self._last_prices: Dict[str, float] = {}
        self._connected = False
        self._market_open = True
        logger.info(
            "PaperBroker initialised (cash=%.2f, slippage_bps=%.1f)",
            initial_cash,
            slippage_bps,
        )

    # -- helpers --------------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        """Set the simulated last price for a symbol."""
        self._last_prices[symbol] = price

    def set_market_open(self, is_open: bool) -> None:
        self._market_open = is_open

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * (self.slippage_bps / 10_000)
        return price + slip if side.lower() == "buy" else price - slip

    def _generate_order_id(self) -> str:
        return f"paper-{uuid.uuid4().hex[:12]}"

    def _fill_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        symbol = order.get("option_symbol", order["symbol"])
        price = self._last_prices.get(symbol, 100.0)
        fill_price = self._apply_slippage(price, order["side"])
        qty = order["qty"]

        # Update cash
        cost = fill_price * qty
        if order["side"].lower() in ("buy", "buy_to_open"):
            self.cash -= cost
        else:
            self.cash += cost

        # Update positions
        current = self.positions.get(symbol, {"qty": 0, "avg_entry_price": 0})
        if order["side"].lower() in ("buy", "buy_to_open"):
            total_qty = current["qty"] + qty
            if total_qty > 0:
                current["avg_entry_price"] = (
                    (current["avg_entry_price"] * current["qty"] + fill_price * qty) / total_qty
                )
            current["qty"] = total_qty
        else:
            current["qty"] -= qty
        current["symbol"] = symbol

        if abs(current["qty"]) < 1e-9:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = current

        order["status"] = "filled"
        order["filled_qty"] = qty
        order["filled_avg_price"] = fill_price
        order["filled_at"] = datetime.now(timezone.utc).isoformat()
        return order

    # -- connection -----------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info("PaperBroker connected")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected")

    # -- account / positions --------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        equity = self.cash + sum(
            self._last_prices.get(s, 100.0) * p["qty"]
            for s, p in self.positions.items()
        )
        return {
            "balance": self.cash,
            "buying_power": self.cash,
            "equity": equity,
            "positions": len(self.positions),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        result = []
        for symbol, pos in self.positions.items():
            price = self._last_prices.get(symbol, 100.0)
            result.append({
                "symbol": symbol,
                "qty": pos["qty"],
                "side": "long" if pos["qty"] > 0 else "short",
                "avg_entry_price": pos["avg_entry_price"],
                "market_value": price * pos["qty"],
                "unrealized_pl": (price - pos["avg_entry_price"]) * pos["qty"],
            })
        return result

    # -- equity orders --------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order: Dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "time_in_force": time_in_force,
            "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        # Immediate fill for market orders
        if order_type.lower() == "market":
            order = self._fill_order(order)
        else:
            self.open_orders[order_id] = order

        self.orders[order_id] = order
        logger.info("PaperBroker order: %s %s %s @ %s -> %s", side, qty, symbol, order_type, order["status"])
        return order

    # -- option orders --------------------------------------------------------

    def submit_option_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        option_symbol: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        order_id = self._generate_order_id()
        order: Dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "option_symbol": option_symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        if order_type.lower() == "market":
            order = self._fill_order(order)
        else:
            self.open_orders[order_id] = order

        self.orders[order_id] = order
        logger.info("PaperBroker option order: %s %s %s -> %s", side, qty, option_symbol, order["status"])
        return order

    def submit_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        group_id = str(uuid.uuid4())[:8]
        results: List[Dict[str, Any]] = []
        for i, leg in enumerate(legs):
            result = self.submit_option_order(
                symbol=leg.get("symbol", ""),
                qty=leg["qty"],
                side=leg["side"],
                order_type=order_type,
                option_symbol=leg.get("option_symbol", leg.get("symbol", "")),
                limit_price=limit_price,
            )
            result["group_id"] = group_id
            result["leg_index"] = i
            results.append(result)
        logger.info("PaperBroker multi-leg: %d legs, group=%s", len(legs), group_id)
        return {"group_id": group_id, "legs": results, "status": "submitted"}

    # -- order management -----------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.open_orders:
            self.open_orders[order_id]["status"] = "cancelled"
            del self.open_orders[order_id]
            if order_id in self.orders:
                self.orders[order_id]["status"] = "cancelled"
            logger.info("PaperBroker order cancelled: %s", order_id)
            return True
        logger.warning("PaperBroker cancel failed – order not found: %s", order_id)
        return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self.orders.get(order_id, {"order_id": order_id, "status": "not_found"})

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return list(self.open_orders.values())

    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        logger.warning("PaperBroker does not provide real option chain data")
        return pd.DataFrame()

    def is_market_open(self) -> bool:
        return self._market_open

    def is_connected(self) -> bool:
        return self._connected

    def flatten_all(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for position in list(self.get_positions()):
            close_side = "sell" if position["qty"] > 0 else "buy"
            results.append(
                self.submit_order(
                    symbol=position["symbol"],
                    qty=abs(position["qty"]),
                    side=close_side,
                    order_type="market",
                )
            )
        return {"closed": len(results), "results": results}
