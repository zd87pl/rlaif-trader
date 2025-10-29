"""
Outcome Tracker for RLAIF

Monitors live trading results and computes outcomes for RLAIF training.

Responsibilities:
1. Track open positions from trading decisions
2. Monitor price movements and compute P&L
3. Calculate risk-adjusted metrics (Sharpe, max drawdown)
4. Trigger outcome updates when positions close
5. Handle partial fills, splits, dividends

Integration:
- Works with PreferenceGenerator to update decision outcomes
- Provides real-time feedback on trading performance
- Supports both paper trading and live execution

Key Metrics Tracked:
- Realized P&L
- Unrealized P&L (for open positions)
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Time in position
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .preference_generator import TradingDecision, PreferenceGenerator
from ..data import AlpacaDataClient
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """
    Represents an open trading position

    Tracks all information needed to compute outcomes
    """

    position_id: str
    decision_id: str  # Links back to TradingDecision
    symbol: str
    action: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    entry_timestamp: datetime

    # Current state
    current_price: Optional[float] = None
    last_update: Optional[datetime] = None

    # Price history (for risk metrics)
    price_history: List[float] = field(default_factory=list)
    timestamp_history: List[datetime] = field(default_factory=list)

    # Exit info (when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "target", "stop_loss", "time_limit", "manual"

    # Computed metrics
    unrealized_pnl: Optional[float] = None
    unrealized_return: Optional[float] = None
    realized_pnl: Optional[float] = None
    realized_return: Optional[float] = None

    is_closed: bool = False

    def update_price(self, price: float, timestamp: datetime) -> None:
        """Update current price and history"""
        self.current_price = price
        self.last_update = timestamp

        self.price_history.append(price)
        self.timestamp_history.append(timestamp)

        # Update unrealized P&L
        if self.action == "buy":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_return = (price - self.entry_price) / self.entry_price
        elif self.action == "sell":
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_return = (self.entry_price - price) / self.entry_price

    def close_position(self, exit_price: float, exit_timestamp: datetime, reason: str) -> None:
        """Close position and compute final metrics"""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        self.is_closed = True

        # Compute realized P&L
        if self.action == "buy":
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
            self.realized_return = (exit_price - self.entry_price) / self.entry_price
        elif self.action == "sell":
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
            self.realized_return = (self.entry_price - exit_price) / self.entry_price

        logger.info(
            f"Position {self.position_id} closed: {self.symbol} {self.action} "
            f"P&L=${self.realized_pnl:.2f} ({self.realized_return:.2%}) - {reason}"
        )

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown during position hold"""
        if len(self.price_history) < 2:
            return 0.0

        prices = np.array(self.price_history)

        if self.action == "buy":
            # For long positions, drawdown is peak to trough
            cumulative_returns = (prices / self.entry_price) - 1
        elif self.action == "sell":
            # For short positions, drawdown is trough to peak (inverted)
            cumulative_returns = (self.entry_price / prices) - 1
        else:
            return 0.0

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max

        return float(drawdowns.min())

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio

        Args:
            risk_free_rate: Annual risk-free rate (default 0%)

        Returns:
            Annualized Sharpe ratio
        """
        if len(self.price_history) < 2:
            return 0.0

        # Compute returns
        prices = pd.Series(self.price_history)
        returns = prices.pct_change().dropna()

        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        # Annualize (assuming daily prices, 252 trading days)
        avg_return = returns.mean()
        std_return = returns.std()

        sharpe = (avg_return - risk_free_rate / 252) / std_return * np.sqrt(252)

        return float(sharpe)

    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only penalizes downside volatility)
        """
        if len(self.price_history) < 2:
            return 0.0

        prices = pd.Series(self.price_history)
        returns = prices.pct_change().dropna()

        if len(returns) < 2:
            return 0.0

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0 if returns.mean() <= 0 else 100.0  # All positive returns

        avg_return = returns.mean()
        downside_std = downside_returns.std()

        sortino = (avg_return - risk_free_rate / 252) / downside_std * np.sqrt(252)

        return float(sortino)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "position_id": self.position_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "current_price": self.current_price,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "exit_reason": self.exit_reason,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_return": self.unrealized_return,
            "realized_pnl": self.realized_pnl,
            "realized_return": self.realized_return,
            "is_closed": self.is_closed,
        }


class OutcomeTracker:
    """
    Tracks trading outcomes for RLAIF feedback

    Responsibilities:
    1. Monitor open positions
    2. Update prices in real-time (or near real-time)
    3. Compute outcomes when positions close
    4. Update PreferenceGenerator with outcomes

    Integration with PreferenceGenerator:
    - When a position closes, automatically update the corresponding TradingDecision
    - Trigger preference generation when enough outcomes are available
    """

    def __init__(
        self,
        preference_generator: PreferenceGenerator,
        data_client: Optional[AlpacaDataClient] = None,
        storage_path: Optional[Path] = None,
        update_interval_seconds: int = 60,  # How often to update prices
        auto_close_after_days: int = 5,  # Auto-close positions after N days
    ):
        """
        Initialize outcome tracker

        Args:
            preference_generator: PreferenceGenerator to update with outcomes
            data_client: Market data client for price updates
            storage_path: Where to store position data
            update_interval_seconds: How often to update prices
            auto_close_after_days: Automatically close positions after this many days
        """
        self.preference_generator = preference_generator
        self.data_client = data_client or AlpacaDataClient()

        self.storage_path = storage_path or Path("./data/rlaif/positions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.update_interval_seconds = update_interval_seconds
        self.auto_close_after_days = auto_close_after_days

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Load existing positions
        self._load_positions()

        logger.info(
            f"OutcomeTracker initialized: {len(self.positions)} active positions"
        )

    def track_decision(
        self,
        decision: TradingDecision,
        quantity: float,
    ) -> Position:
        """
        Start tracking a trading decision as a position

        Args:
            decision: Trading decision from agent
            quantity: Number of shares/contracts

        Returns:
            Created Position
        """
        position_id = f"pos_{decision.decision_id}"

        position = Position(
            position_id=position_id,
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            action=decision.action,
            quantity=quantity,
            entry_price=decision.entry_price or decision.market_data.get("close", 0.0),
            entry_timestamp=decision.timestamp,
        )

        # Initialize with current price
        position.update_price(position.entry_price, decision.timestamp)

        self.positions[position_id] = position
        self._save_position(position)

        logger.info(
            f"Tracking position {position_id}: {decision.action} {quantity} {decision.symbol} "
            f"@ ${position.entry_price:.2f}"
        )

        return position

    def update_positions(self) -> None:
        """
        Update all active positions with current prices

        This should be called periodically (e.g., every minute)
        """
        if not self.positions:
            return

        # Get unique symbols
        symbols = list(set(p.symbol for p in self.positions.values() if not p.is_closed))

        if not symbols:
            return

        logger.info(f"Updating {len(symbols)} symbols for {len(self.positions)} positions")

        try:
            # Fetch latest prices
            # Note: In production, you'd use real-time data or recent bars
            df = self.data_client.download_latest(
                symbols=symbols,
                days=1,
                timeframe="1Min",
            )

            current_time = datetime.now()

            # Update each position
            for position in self.positions.values():
                if position.is_closed:
                    continue

                # Get latest price for this symbol
                if position.symbol in df.index.get_level_values(0):
                    symbol_data = df.loc[position.symbol]
                    if len(symbol_data) > 0:
                        latest_price = symbol_data["close"].iloc[-1]
                        position.update_price(latest_price, current_time)

                # Check auto-close conditions
                self._check_auto_close(position, current_time)

            # Save updates
            for position in self.positions.values():
                self._save_position(position)

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    def close_position(
        self,
        position_id: str,
        exit_price: Optional[float] = None,
        reason: str = "manual",
    ) -> None:
        """
        Manually close a position

        Args:
            position_id: Position to close
            exit_price: Exit price (if None, uses current price)
            reason: Reason for closing
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return

        position = self.positions[position_id]

        if position.is_closed:
            logger.warning(f"Position {position_id} already closed")
            return

        # Use current price if not specified
        if exit_price is None:
            exit_price = position.current_price or position.entry_price

        # Close position
        position.close_position(exit_price, datetime.now(), reason)
        self._save_position(position)

        # Update PreferenceGenerator with outcome
        self._update_decision_outcome(position)

    def _check_auto_close(self, position: Position, current_time: datetime) -> None:
        """Check if position should be auto-closed"""
        if position.is_closed:
            return

        # Time-based auto-close
        time_in_position = current_time - position.entry_timestamp
        if time_in_position > timedelta(days=self.auto_close_after_days):
            exit_price = position.current_price or position.entry_price
            position.close_position(exit_price, current_time, "time_limit")
            self._update_decision_outcome(position)
            logger.info(
                f"Auto-closed position {position.position_id} after "
                f"{self.auto_close_after_days} days"
            )

    def _update_decision_outcome(self, position: Position) -> None:
        """
        Update the corresponding TradingDecision with outcome

        This feeds back into the PreferenceGenerator
        """
        if not position.is_closed:
            logger.warning("Cannot update outcome for open position")
            return

        # Create price history series
        price_series = pd.Series(
            position.price_history,
            index=position.timestamp_history,
        )

        # Update decision outcome
        self.preference_generator.update_outcome(
            decision_id=position.decision_id,
            exit_price=position.exit_price,
            price_history=price_series,
        )

        logger.info(
            f"Updated decision {position.decision_id} with outcome: "
            f"Return={position.realized_return:.2%}, "
            f"Sharpe={position.get_sharpe_ratio():.2f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked positions"""
        total_positions = len(self.positions)
        open_positions = [p for p in self.positions.values() if not p.is_closed]
        closed_positions = [p for p in self.positions.values() if p.is_closed]

        stats = {
            "total_positions": total_positions,
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
        }

        if closed_positions:
            returns = [p.realized_return for p in closed_positions if p.realized_return is not None]
            pnls = [p.realized_pnl for p in closed_positions if p.realized_pnl is not None]

            stats.update(
                {
                    "avg_return": float(np.mean(returns)) if returns else 0.0,
                    "total_pnl": float(np.sum(pnls)) if pnls else 0.0,
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0,
                    "avg_winning_return": (
                        float(np.mean([r for r in returns if r > 0]))
                        if any(r > 0 for r in returns)
                        else 0.0
                    ),
                    "avg_losing_return": (
                        float(np.mean([r for r in returns if r < 0]))
                        if any(r < 0 for r in returns)
                        else 0.0
                    ),
                }
            )

        if open_positions:
            unrealized_pnls = [
                p.unrealized_pnl for p in open_positions if p.unrealized_pnl is not None
            ]
            stats["unrealized_pnl"] = float(np.sum(unrealized_pnls)) if unrealized_pnls else 0.0

        return stats

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if not p.is_closed]

    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions"""
        return [p for p in self.positions.values() if p.is_closed]

    # ===================================================================================
    # Persistence
    # ===================================================================================

    def _save_position(self, position: Position) -> None:
        """Save position to disk"""
        path = self.storage_path / f"{position.position_id}.json"
        with open(path, "w") as f:
            json.dump(position.to_dict(), f, indent=2)

    def _load_positions(self) -> None:
        """Load positions from disk"""
        if not self.storage_path.exists():
            return

        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    # Reconstruct Position (simplified - you'd need full deserialization)
                    position_id = data["position_id"]
                    # Skip for now - full deserialization would be needed
                    logger.debug(f"Found saved position: {position_id}")
            except Exception as e:
                logger.warning(f"Could not load position from {file}: {e}")
