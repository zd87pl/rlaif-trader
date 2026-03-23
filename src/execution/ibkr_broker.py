"""Interactive Brokers integration via ib_insync"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    HAS_IB = True
except ImportError:
    HAS_IB = False


class IBKRBroker:
    """
    Broker interface to Interactive Brokers via TWS/Gateway.

    Requirements:
    - pip install ib_insync
    - TWS or IB Gateway running (paper: port 7497, live: port 7496)

    Supports:
    - Paper and live trading
    - Market and limit orders
    - Position queries
    - Account info
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ):
        if not HAS_IB:
            raise ImportError(
                "ib_insync is required for IBKR. Install with: pip install ib_insync"
            )

        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

        self._connect()

    def _connect(self) -> None:
        """Connect to TWS/Gateway with retry."""
        for attempt in range(3):
            try:
                self.ib.connect(
                    self.host, self.port, clientId=self.client_id, timeout=10
                )
                account_values = self.ib.accountSummary()
                equity = next(
                    (
                        float(v.value)
                        for v in account_values
                        if v.tag == "NetLiquidation"
                    ),
                    0.0,
                )
                logger.info(
                    f"Connected to IBKR (port {self.port}). "
                    f"Equity: ${equity:,.2f}"
                )
                return
            except Exception as e:
                logger.warning(f"IBKR connection attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        raise ConnectionError(
            f"Cannot connect to IBKR at {self.host}:{self.port}. "
            "Ensure TWS or IB Gateway is running."
        )

    def get_account(self) -> Dict[str, Any]:
        """Get account summary."""
        values = self.ib.accountSummary()
        result = {}
        key_map = {
            "NetLiquidation": "equity",
            "BuyingPower": "buying_power",
            "TotalCashValue": "cash",
            "GrossPositionValue": "portfolio_value",
        }
        for v in values:
            if v.tag in key_map:
                result[key_map[v.tag]] = float(v.value)

        result.setdefault("equity", 0.0)
        result.setdefault("buying_power", 0.0)
        result.setdefault("cash", 0.0)
        result.setdefault("portfolio_value", 0.0)
        result["long_market_value"] = 0.0
        result["short_market_value"] = 0.0
        result["pattern_day_trader"] = False
        result["trading_blocked"] = False
        result["account_blocked"] = False

        return result

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        positions = self.ib.positions()
        result = []
        for p in positions:
            contract = p.contract
            avg_cost = p.avgCost
            qty = p.position

            # Get current price
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(1)
            current_price = ticker.marketPrice()
            if current_price != current_price:  # NaN check
                current_price = avg_cost

            market_value = qty * current_price
            cost_basis = qty * avg_cost
            unrealized_pl = market_value - cost_basis

            result.append(
                {
                    "symbol": contract.symbol,
                    "qty": float(qty),
                    "side": "long" if qty > 0 else "short",
                    "market_value": float(market_value),
                    "cost_basis": float(cost_basis),
                    "unrealized_pl": float(unrealized_pl),
                    "unrealized_plpc": (
                        float(unrealized_pl / cost_basis) if cost_basis else 0.0
                    ),
                    "current_price": float(current_price),
                    "avg_entry_price": float(avg_cost),
                }
            )
            self.ib.cancelMktData(contract)

        return result

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for p in positions:
            if p["symbol"] == symbol:
                return p
        return None

    def market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Place a market order."""
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        action = "BUY" if side.lower() == "buy" else "SELL"
        order = MarketOrder(action, abs(qty))

        if time_in_force.lower() == "gtc":
            order.tif = "GTC"

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)

        logger.info(
            f"IBKR market order: {action} {qty} {symbol} "
            f"(order_id={trade.order.orderId}, status={trade.orderStatus.status})"
        )

        return self._trade_to_dict(trade, symbol)

    def limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """Place a limit order."""
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        action = "BUY" if side.lower() == "buy" else "SELL"
        order = LimitOrder(action, abs(qty), limit_price)

        if time_in_force.lower() == "gtc":
            order.tif = "GTC"

        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)

        logger.info(
            f"IBKR limit order: {action} {qty} {symbol} @ ${limit_price:.2f} "
            f"(order_id={trade.order.orderId})"
        )

        return self._trade_to_dict(trade, symbol)

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close entire position for a symbol."""
        pos = self.get_position(symbol)
        if not pos:
            raise ValueError(f"No position in {symbol}")

        side = "sell" if pos["qty"] > 0 else "buy"
        return self.market_order(symbol, abs(pos["qty"]), side)

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all positions."""
        logger.warning("IBKR: CLOSING ALL POSITIONS")
        results = []
        for pos in self.get_positions():
            try:
                result = self.close_position(pos["symbol"])
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to close {pos['symbol']}: {e}")
                results.append({"symbol": pos["symbol"], "status": f"error: {e}"})
        return results

    def is_market_open(self) -> bool:
        """Check if US stock market is open."""
        # ib_insync doesn't have a direct clock check; use a heuristic
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Market open: Mon-Fri 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)
        if now.weekday() >= 5:
            return False
        hour_min = now.hour * 100 + now.minute
        return 1430 <= hour_min < 2100

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        self.ib.disconnect()
        logger.info("Disconnected from IBKR")

    def _trade_to_dict(self, trade, symbol: str) -> Dict[str, Any]:
        status = trade.orderStatus
        return {
            "order_id": str(trade.order.orderId),
            "symbol": symbol,
            "qty": str(trade.order.totalQuantity),
            "side": trade.order.action.lower(),
            "type": trade.order.orderType,
            "status": status.status,
            "filled_qty": str(status.filled),
            "filled_avg_price": str(status.avgFillPrice) if status.avgFillPrice else None,
            "submitted_at": datetime.now().isoformat(),
            "filled_at": None,
        }
