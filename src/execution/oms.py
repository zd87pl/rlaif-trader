"""Order Management System – orchestrates order flow through the broker."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .broker import BrokerInterface
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OrderManagementSystem:
    """Manages order lifecycle, risk validation, and portfolio state.

    Parameters
    ----------
    broker : BrokerInterface
        The broker to route orders through.
    risk_engine : optional
        An object with a ``validate(order: Dict) -> (bool, str)`` method.
        When provided every order is checked before submission.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        risk_engine: Any = None,
    ) -> None:
        self.broker = broker
        self.risk_engine = risk_engine
        self.pending_orders: List[Dict[str, Any]] = []
        self.order_history: List[Dict[str, Any]] = []
        logger.info(
            "OMS initialised (broker=%s, risk_engine=%s)",
            type(broker).__name__,
            type(risk_engine).__name__ if risk_engine else "None",
        )

    # ------------------------------------------------------------------
    # Risk validation
    # ------------------------------------------------------------------

    def _validate_order(self, order_info: Dict[str, Any]) -> bool:
        """Run the order through the risk engine (if present).

        Returns True if the order is allowed.
        """
        if self.risk_engine is None:
            return True
        try:
            if hasattr(self.risk_engine, "validate_order") and order_info.get("expiration") is not None:
                account = self.broker.get_account() if self.broker is not None else {}
                portfolio = {
                    "equity": account.get("equity", account.get("balance", 0.0)),
                    "daily_pnl": 0.0,
                    "weekly_pnl": 0.0,
                    "total_exposure": 0.0,
                    "trade_count_5d": 0,
                }
                allowed, reason = self.risk_engine.validate_order(order_info, portfolio)
            elif hasattr(self.risk_engine, "check_risk"):
                allowed = self.risk_engine.check_risk(
                    symbol=order_info.get("symbol", ""),
                    action=order_info.get("side", order_info.get("action", "buy")),
                    size=float(order_info.get("qty", order_info.get("quantity", 0)) or 0),
                )
                reason = "approved" if allowed else "risk_check_failed"
            else:
                allowed, reason = self.risk_engine.validate(order_info)
            if not allowed:
                logger.warning(
                    "OMS risk check REJECTED order: %s – %s", order_info, reason,
                )
            return allowed
        except Exception as exc:
            logger.error("OMS risk engine error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Audit logging helper
    # ------------------------------------------------------------------

    def _record_order(self, order: Dict[str, Any], action: str) -> None:
        """Append full order details to history for audit trail."""
        record = {
            **order,
            "action": action,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.order_history.append(record)
        logger.info("OMS audit [%s]: %s", action, record)

    # ------------------------------------------------------------------
    # Strategy & signal execution
    # ------------------------------------------------------------------

    def execute_strategy(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a full options strategy (e.g. from OptionsStrategyBuilder).

        *strategy* is expected to contain at minimum:
        - ``legs``: List[Dict] with keys option_symbol, qty, side, symbol
        - ``order_type``: str  (market / limit / etc.)
        - ``limit_price``: optional float (net debit / credit for the combo)

        Returns a list of order result dicts.
        """
        legs: List[Dict[str, Any]] = strategy.get("legs", [])
        order_type: str = strategy.get("order_type", "market")
        limit_price: Optional[float] = strategy.get("limit_price")
        strategy_name: str = strategy.get("name", "unnamed")

        if not legs:
            logger.warning("OMS execute_strategy called with no legs")
            return []

        logger.info(
            "OMS executing strategy '%s' with %d leg(s)", strategy_name, len(legs),
        )

        # Validate via risk engine
        order_info = {
            "type": "strategy",
            "strategy_name": strategy_name,
            "legs": legs,
            "order_type": order_type,
            "limit_price": limit_price,
        }
        if not self._validate_order(order_info):
            return [{"status": "rejected", "reason": "risk_check_failed"}]

        results: List[Dict[str, Any]] = []
        if len(legs) == 1:
            leg = legs[0]
            result = self.broker.submit_option_order(
                symbol=leg.get("symbol", ""),
                qty=leg["qty"],
                side=leg["side"],
                order_type=order_type,
                option_symbol=leg.get("option_symbol", ""),
                limit_price=limit_price,
            )
            results.append(result)
            self._record_order(result, f"strategy:{strategy_name}:single_leg")
        else:
            result = self.broker.submit_multi_leg_order(
                legs=legs,
                order_type=order_type,
                limit_price=limit_price,
            )
            results.append(result)
            self._record_order(result, f"strategy:{strategy_name}:multi_leg")

        # Track pending
        for r in results:
            if r.get("status") not in ("filled", "rejected"):
                self.pending_orders.append(r)

        return results

    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a trading signal dict into an order and submit it.

        Expected signal keys:
        - ``symbol``: str
        - ``side``: 'buy' | 'sell'
        - ``qty``: float
        - ``order_type``: str (default 'market')
        - ``limit_price``, ``stop_price``: optional floats
        - ``time_in_force``: str (default 'day')
        - ``option_symbol``: optional – if present, routes as option order
        """
        symbol: str = signal["symbol"]
        side: str = signal["side"]
        qty: float = signal["qty"]
        order_type: str = signal.get("order_type", "market")

        logger.info("OMS executing signal: %s %s %s @ %s", side, qty, symbol, order_type)

        if not self._validate_order(signal):
            result = {"status": "rejected", "reason": "risk_check_failed", "signal": signal}
            self._record_order(result, "signal:rejected")
            return result

        if signal.get("option_symbol"):
            result = self.broker.submit_option_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                option_symbol=signal["option_symbol"],
                limit_price=signal.get("limit_price"),
            )
        else:
            result = self.broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                limit_price=signal.get("limit_price"),
                stop_price=signal.get("stop_price"),
                time_in_force=signal.get("time_in_force", "day"),
            )

        self._record_order(result, "signal:submitted")
        if result.get("status") not in ("filled", "rejected"):
            self.pending_orders.append(result)

        return result

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_portfolio_state(self) -> Dict[str, Any]:
        """Return current positions, P&L, and Greeks exposure."""
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        total_unrealized_pl = sum(p.get("unrealized_pl", 0) for p in positions)

        # Aggregate Greeks if positions carry them
        greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for pos in positions:
            for g in greeks:
                greeks[g] += pos.get(g, 0.0) * pos.get("qty", 0)

        return {
            "account": account,
            "positions": positions,
            "total_unrealized_pl": total_unrealized_pl,
            "greeks_exposure": greeks,
            "pending_orders": len(self.pending_orders),
            "total_orders_executed": len(self.order_history),
        }

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def close_position(self, symbol_or_order_id: str) -> Dict[str, Any]:
        """Close a position identified by symbol or order id."""
        positions = self.broker.get_positions()
        target = None
        for pos in positions:
            if pos["symbol"] == symbol_or_order_id:
                target = pos
                break

        if target is None:
            logger.warning("OMS close_position: '%s' not found", symbol_or_order_id)
            return {"status": "not_found", "symbol": symbol_or_order_id}

        close_side = "sell" if target["qty"] > 0 else "buy"
        result = self.broker.submit_order(
            symbol=target["symbol"],
            qty=abs(target["qty"]),
            side=close_side,
            order_type="market",
        )
        self._record_order(result, "close_position")
        return result

    def close_all_positions(self) -> Dict[str, Any]:
        """Close every open position."""
        positions = self.broker.get_positions()
        results: List[Dict[str, Any]] = []
        for pos in positions:
            r = self.close_position(pos["symbol"])
            results.append(r)
        logger.info("OMS close_all_positions: closed %d position(s)", len(results))
        return {"closed": len(results), "results": results}

    def adjust_position(self, position_id: str, new_qty: float) -> Dict[str, Any]:
        """Adjust an existing position to *new_qty*.

        If new_qty is 0 the position is closed.  If new_qty differs from the
        current quantity, a buy/sell order bridges the gap.
        """
        positions = self.broker.get_positions()
        target = None
        for pos in positions:
            if pos["symbol"] == position_id:
                target = pos
                break

        if target is None:
            logger.warning("OMS adjust_position: '%s' not found", position_id)
            return {"status": "not_found", "position_id": position_id}

        current_qty = target["qty"]
        diff = new_qty - current_qty

        if abs(diff) < 1e-9:
            return {"status": "no_change", "position_id": position_id, "qty": current_qty}

        if new_qty == 0:
            return self.close_position(position_id)

        side = "buy" if diff > 0 else "sell"
        order_info = {
            "symbol": position_id,
            "qty": abs(diff),
            "side": side,
            "order_type": "market",
            "type": "adjustment",
        }
        if not self._validate_order(order_info):
            return {"status": "rejected", "reason": "risk_check_failed"}

        result = self.broker.submit_order(
            symbol=position_id,
            qty=abs(diff),
            side=side,
            order_type="market",
        )
        self._record_order(result, f"adjust_position:{position_id}")
        return result

    def submit_order(self, **kwargs) -> Dict[str, Any]:
        """Compatibility wrapper for scheduler/integration code."""
        return self.execute_signal(kwargs)

    def get_positions(self) -> List[Dict[str, Any]]:
        return self.broker.get_positions()

    def get_daily_pnl(self) -> float:
        positions = self.broker.get_positions()
        return float(sum(position.get("unrealized_pl", 0.0) for position in positions))

    def get_daily_trades(self) -> List[Dict[str, Any]]:
        today = datetime.now(timezone.utc).date().isoformat()
        trades: List[Dict[str, Any]] = []
        for record in self.order_history:
            recorded_at = str(record.get("recorded_at", ""))
            if recorded_at.startswith(today):
                trades.append(record)
        return trades
