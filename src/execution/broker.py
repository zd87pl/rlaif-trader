"""Alpaca broker integration for live and paper trading"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AlpacaBroker:
    """
    Broker interface to Alpaca for order execution.

    Supports:
    - Paper trading (default) and live trading
    - Market and limit orders
    - Position queries
    - Account info and buying power
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = paper

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        account = self.client.get_account()
        logger.info(
            f"Alpaca broker initialized ({'PAPER' if paper else 'LIVE'}). "
            f"Equity: ${float(account.equity):,.2f}, "
            f"Buying power: ${float(account.buying_power):,.2f}"
        )

    def get_account(self) -> Dict[str, Any]:
        """Get account summary."""
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        try:
            p = self.client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            }
        except Exception:
            return None

    def market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Stock symbol
            qty: Number of shares (fractional supported)
            side: "buy" or "sell"
            time_in_force: "day", "gtc", "ioc", "fok"
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY),
        )

        order = self.client.submit_order(request)
        logger.info(
            f"Market order placed: {side.upper()} {qty} {symbol} "
            f"(order_id={order.id}, status={order.status})"
        )

        return self._order_to_dict(order)

    def limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Place a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
        }

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=limit_price,
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY),
        )

        order = self.client.submit_order(request)
        logger.info(
            f"Limit order placed: {side.upper()} {qty} {symbol} @ ${limit_price:.2f} "
            f"(order_id={order.id})"
        )

        return self._order_to_dict(order)

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close an entire position for a symbol."""
        order = self.client.close_position(symbol)
        logger.info(f"Closing position: {symbol} (order_id={order.id})")
        return self._order_to_dict(order)

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions (emergency liquidation)."""
        logger.warning("CLOSING ALL POSITIONS")
        responses = self.client.close_all_positions(cancel_orders=True)
        return [{"symbol": r.symbol, "status": str(r.status)} for r in responses]

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status by ID."""
        order = self.client.get_order_by_id(order_id)
        return self._order_to_dict(order)

    def get_recent_orders(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent orders."""
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )
        orders = self.client.get_orders(request)
        return [self._order_to_dict(o) for o in orders]

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        self.client.cancel_order_by_id(order_id)
        logger.info(f"Cancelled order: {order_id}")

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.client.get_clock()
        return clock.is_open

    def _order_to_dict(self, order) -> Dict[str, Any]:
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
            "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
            "filled_avg_price": (
                str(order.filled_avg_price) if order.filled_avg_price else None
            ),
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
        }
