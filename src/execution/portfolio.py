"""Portfolio state and trade log"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Portfolio:
    """
    Tracks all trades and portfolio state. Persists to disk.
    Separate from broker state — this is OUR record.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/portfolio")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.trade_log_path = self.storage_path / "trades.jsonl"
        self.trades: List[Dict[str, Any]] = []
        self._load_trades()

    def log_trade(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        order_id: str,
        signal_score: float,
        signal_confidence: float,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Log a completed trade."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": price,
            "dollar_amount": qty * price,
            "order_id": order_id,
            "signal_score": signal_score,
            "signal_confidence": signal_confidence,
            "reasoning": reasoning[:500],
        }

        self.trades.append(trade)
        self._append_trade(trade)

        logger.info(
            f"Trade logged: {action.upper()} {qty} {symbol} @ ${price:.2f} "
            f"(${qty * price:,.2f})"
        )
        return trade

    def get_summary(self) -> Dict[str, Any]:
        """Get trade history summary."""
        if not self.trades:
            return {"total_trades": 0}

        buys = [t for t in self.trades if t["action"] == "buy"]
        sells = [t for t in self.trades if t["action"] == "sell"]

        return {
            "total_trades": len(self.trades),
            "buys": len(buys),
            "sells": len(sells),
            "total_bought": sum(t["dollar_amount"] for t in buys),
            "total_sold": sum(t["dollar_amount"] for t in sells),
            "symbols_traded": list(set(t["symbol"] for t in self.trades)),
            "last_trade": self.trades[-1] if self.trades else None,
        }

    def _append_trade(self, trade: Dict[str, Any]) -> None:
        with open(self.trade_log_path, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _load_trades(self) -> None:
        if not self.trade_log_path.exists():
            return
        with open(self.trade_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.trades.append(json.loads(line))
        logger.info(f"Loaded {len(self.trades)} historical trades")
