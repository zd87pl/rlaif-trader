"""Risk management and position sizing"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Enforces risk limits before any order goes to the broker.

    Rules:
    - Max position size as % of portfolio
    - Max total exposure
    - Min confidence threshold to trade
    - Max daily loss (circuit breaker)
    - Max number of open positions
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_total_exposure_pct: float = 0.60,
        min_confidence: float = 0.55,
        max_daily_loss_pct: float = 0.03,
        max_open_positions: int = 10,
        min_score_magnitude: float = 0.3,
    ):
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.min_confidence = min_confidence
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.min_score_magnitude = min_score_magnitude

        self.daily_pnl = 0.0
        self.trades_today = 0

        logger.info(
            f"RiskManager: max_pos={max_position_pct:.0%}, "
            f"max_exposure={max_total_exposure_pct:.0%}, "
            f"min_conf={min_confidence}, max_daily_loss={max_daily_loss_pct:.1%}"
        )

    def check_signal(
        self,
        symbol: str,
        action: str,
        score: float,
        confidence: float,
        current_price: float,
        account: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate a signal and compute position size.

        Returns:
            {
                "approved": bool,
                "reason": str,
                "qty": float,
                "dollar_amount": float,
            }
        """
        equity = account["equity"]

        # Circuit breaker: daily loss limit
        if self.daily_pnl < -(equity * self.max_daily_loss_pct):
            return self._reject(f"Daily loss limit hit (${self.daily_pnl:,.2f})")

        # Confidence gate
        if confidence < self.min_confidence:
            return self._reject(f"Confidence too low ({confidence:.0%} < {self.min_confidence:.0%})")

        # Score magnitude gate
        if abs(score) < self.min_score_magnitude:
            return self._reject(f"Score too weak ({score:.2f}, need >{self.min_score_magnitude})")

        # Signal direction check
        if action == "buy" and score < 0:
            return self._reject(f"Buy signal but negative score ({score:.2f})")
        if action == "sell" and score > 0:
            return self._reject(f"Sell signal but positive score ({score:.2f})")

        # Max open positions
        if action == "buy" and len(open_positions) >= self.max_open_positions:
            return self._reject(f"Max open positions reached ({self.max_open_positions})")

        # Check if already holding this symbol
        existing = [p for p in open_positions if p["symbol"] == symbol]
        if existing and action == "buy":
            return self._reject(f"Already holding {symbol}")

        # If selling, check we actually hold it
        if action == "sell" and not existing:
            return self._reject(f"No position in {symbol} to sell")

        # Total exposure check
        total_exposure = sum(abs(p["market_value"]) for p in open_positions)
        if total_exposure > equity * self.max_total_exposure_pct and action == "buy":
            return self._reject(
                f"Total exposure too high "
                f"(${total_exposure:,.0f} > {self.max_total_exposure_pct:.0%} of ${equity:,.0f})"
            )

        # Position sizing: scale by confidence and score magnitude
        base_size = equity * self.max_position_pct
        scaled_size = base_size * min(confidence, 1.0) * min(abs(score), 1.0)
        scaled_size = max(scaled_size, 0)

        # Don't exceed buying power
        buying_power = account["buying_power"]
        scaled_size = min(scaled_size, buying_power * 0.95)

        qty = int(scaled_size / current_price) if current_price > 0 else 0
        if qty < 1 and action == "buy":
            return self._reject(f"Position too small (${scaled_size:.2f} < 1 share @ ${current_price:.2f})")

        # For sells, use existing quantity
        if action == "sell" and existing:
            qty = int(float(existing[0]["qty"]))

        dollar_amount = qty * current_price

        logger.info(
            f"APPROVED: {action.upper()} {qty} {symbol} @ ${current_price:.2f} "
            f"= ${dollar_amount:,.2f} (score={score:.2f}, conf={confidence:.0%})"
        )

        return {
            "approved": True,
            "reason": "passed",
            "qty": qty,
            "dollar_amount": dollar_amount,
        }

    def record_trade_pnl(self, pnl: float) -> None:
        """Track daily P&L for circuit breaker."""
        self.daily_pnl += pnl
        self.trades_today += 1

    def reset_daily(self) -> None:
        """Reset daily counters (call at market open)."""
        self.daily_pnl = 0.0
        self.trades_today = 0

    def _reject(self, reason: str) -> Dict[str, Any]:
        logger.info(f"REJECTED: {reason}")
        return {
            "approved": False,
            "reason": reason,
            "qty": 0,
            "dollar_amount": 0.0,
        }
