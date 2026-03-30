"""
Options Outcome Tracker for RLAIF

Extends the general outcome tracking to options-specific metrics:
  - Greeks accuracy tracking (predicted vs realized)
  - IV prediction accuracy
  - Theta capture efficiency
  - Vol surface edge realized
  - Strategy-level performance breakdowns

Generates options-specific preference pairs for RLAIF training,
comparing trades with similar setups (same strategy type, similar DTE).
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .preference_generator import PreferenceGenerator, TradingDecision, PreferencePair
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptionsOutcomeTracker:
    """
    Tracks options trade outcomes and generates options-specific preference pairs.

    Stores all trade data as JSON under data/options_outcomes/.

    Workflow:
      1. track_option_trade()   - record entry with Greeks/IV/strategy
      2. update_option_outcome() - periodic mark-to-market updates
      3. close_option_trade()   - compute final outcome metrics
      4. generate_option_preferences() - create RLAIF preference pairs
    """

    def __init__(
        self,
        preference_generator: PreferenceGenerator,
        storage_path: Optional[Path] = None,
    ):
        """
        Args:
            preference_generator: PreferenceGenerator for creating preference pairs.
            storage_path: Directory for JSON persistence.
        """
        self.preference_generator = preference_generator
        self.storage_path = storage_path or Path("./data/options_outcomes")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory trade store: tracking_id -> trade dict
        self._trades: Dict[str, Dict[str, Any]] = {}

        # Load existing trades from disk
        self._load_trades()

        logger.info(
            f"OptionsOutcomeTracker initialized: {len(self._trades)} existing trades"
        )

    # ==================================================================================
    # Core API
    # ==================================================================================

    def track_option_trade(self, trade: Dict[str, Any]) -> str:
        """
        Start tracking an options trade.

        Args:
            trade: Dict with keys:
                - symbol: str (underlying)
                - strategy_type: str (e.g. "long_call", "bull_call_spread",
                  "iron_condor", "covered_call", "straddle", etc.)
                - legs: List[Dict] - each leg has:
                    strike, expiry, option_type ("call"/"put"), side ("buy"/"sell"), qty
                - entry_price: float (net debit/credit)
                - entry_greeks: Dict (delta, gamma, theta, vega, rho)
                - entry_iv: float (implied vol at entry)
                - dte: int (days to expiration)
                - underlying_price: float
                - entry_timestamp: str (ISO) or datetime (optional, defaults to now)
                - max_loss: float (optional)
                - max_profit: float (optional)
                - rationale: str (optional)

        Returns:
            tracking_id: Unique ID for this trade.
        """
        tracking_id = f"opt_{uuid.uuid4().hex[:12]}"

        record: Dict[str, Any] = {
            "tracking_id": tracking_id,
            "symbol": trade.get("symbol", ""),
            "strategy_type": trade.get("strategy_type", "unknown"),
            "legs": trade.get("legs", []),
            "entry_price": trade.get("entry_price", 0.0),
            "entry_greeks": trade.get("entry_greeks", {}),
            "entry_iv": trade.get("entry_iv", 0.0),
            "dte": trade.get("dte", 0),
            "underlying_price": trade.get("underlying_price", 0.0),
            "entry_timestamp": _to_iso(trade.get("entry_timestamp", datetime.now())),
            "max_loss": trade.get("max_loss"),
            "max_profit": trade.get("max_profit"),
            "rationale": trade.get("rationale", ""),
            # Will be filled by updates
            "current_greeks": {},
            "current_iv": None,
            "current_price": None,
            "unrealized_pnl": None,
            "time_decay_captured": 0.0,
            "updates": [],
            # Closed state
            "is_closed": False,
            "exit_data": None,
            "outcome_metrics": None,
        }

        self._trades[tracking_id] = record
        self._save_trade(tracking_id)

        logger.info(
            f"Tracking option trade {tracking_id}: "
            f"{record['strategy_type']} on {record['symbol']} "
            f"(DTE={record['dte']}, IV={record['entry_iv']:.1%})"
        )

        return tracking_id

    def update_option_outcome(self, tracking_id: str, current_data: Dict[str, Any]) -> None:
        """
        Update a tracked option trade with current market data.

        Args:
            tracking_id: Trade tracking ID.
            current_data: Dict with keys (all optional):
                - current_greeks: Dict
                - current_iv: float
                - current_price: float (mark-to-market)
                - underlying_price: float
                - timestamp: str or datetime
        """
        if tracking_id not in self._trades:
            logger.warning(f"Trade {tracking_id} not found")
            return

        record = self._trades[tracking_id]
        if record["is_closed"]:
            logger.warning(f"Trade {tracking_id} is already closed")
            return

        timestamp = _to_iso(current_data.get("timestamp", datetime.now()))

        # Update Greeks
        if "current_greeks" in current_data:
            record["current_greeks"] = current_data["current_greeks"]

        # Update IV
        if "current_iv" in current_data:
            record["current_iv"] = current_data["current_iv"]

        # Update price / P&L
        if "current_price" in current_data:
            record["current_price"] = current_data["current_price"]
            record["unrealized_pnl"] = current_data["current_price"] - record["entry_price"]

        # Time decay captured = entry theta * days held (approximation)
        if record["entry_greeks"].get("theta") and record.get("entry_timestamp"):
            entry_dt = datetime.fromisoformat(record["entry_timestamp"])
            now_dt = datetime.fromisoformat(timestamp)
            days_held = max((now_dt - entry_dt).total_seconds() / 86400, 0)
            # Positive theta capture means we collected time value
            entry_theta = record["entry_greeks"]["theta"]
            # If we're short options (negative theta = collecting premium), captured is positive
            record["time_decay_captured"] = -entry_theta * days_held

        # Store update snapshot
        record["updates"].append({
            "timestamp": timestamp,
            "current_price": current_data.get("current_price"),
            "current_iv": current_data.get("current_iv"),
            "underlying_price": current_data.get("underlying_price"),
            "unrealized_pnl": record["unrealized_pnl"],
        })

        self._save_trade(tracking_id)

        logger.debug(
            f"Updated {tracking_id}: price={record['current_price']}, "
            f"pnl={record['unrealized_pnl']}"
        )

    def close_option_trade(self, tracking_id: str, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close an options trade and compute final outcome metrics.

        Args:
            tracking_id: Trade tracking ID.
            exit_data: Dict with keys:
                - exit_price: float (net proceeds)
                - exit_greeks: Dict (optional, Greeks at exit)
                - exit_iv: float (optional, IV at exit)
                - exit_timestamp: str or datetime (optional)
                - exit_reason: str (optional: "target", "stop", "expiry", "manual")
                - underlying_price_at_exit: float (optional)

        Returns:
            Dict of computed outcome metrics.
        """
        if tracking_id not in self._trades:
            logger.warning(f"Trade {tracking_id} not found")
            return {}

        record = self._trades[tracking_id]
        if record["is_closed"]:
            logger.warning(f"Trade {tracking_id} already closed")
            return record.get("outcome_metrics", {})

        exit_price = exit_data.get("exit_price", 0.0)
        exit_timestamp = _to_iso(exit_data.get("exit_timestamp", datetime.now()))

        # ---------- Compute outcome metrics ----------

        realized_pnl = exit_price - record["entry_price"]

        # IV prediction accuracy: how close was entry IV to realized vol
        iv_prediction_accuracy = 0.0
        exit_iv = exit_data.get("exit_iv", record.get("current_iv"))
        if record["entry_iv"] and exit_iv:
            # Accuracy = 1 - |predicted - actual| / predicted
            iv_diff = abs(record["entry_iv"] - exit_iv)
            if record["entry_iv"] > 0:
                iv_prediction_accuracy = max(0.0, 1.0 - iv_diff / record["entry_iv"])

        # Greeks accuracy: compare entry greeks predicted move to actual
        greeks_accuracy = self._compute_greeks_accuracy(record, exit_data)

        # Theta captured percentage
        theta_captured_pct = 0.0
        entry_theta = record["entry_greeks"].get("theta", 0)
        if entry_theta != 0:
            entry_dt = datetime.fromisoformat(record["entry_timestamp"])
            exit_dt = datetime.fromisoformat(exit_timestamp)
            days_held = max((exit_dt - entry_dt).total_seconds() / 86400, 0)
            expected_theta_pnl = -entry_theta * days_held
            if abs(expected_theta_pnl) > 0.001:
                theta_captured_pct = realized_pnl / expected_theta_pnl

        # Vol surface edge realized: did we capture the vol edge?
        vol_surface_edge_realized = 0.0
        if record["entry_iv"] and exit_iv:
            # If we sold vol (short vega) and IV dropped, that's realized edge
            entry_vega = record["entry_greeks"].get("vega", 0)
            iv_change = exit_iv - record["entry_iv"]
            if entry_vega != 0:
                # Expected P&L from vol move
                expected_vol_pnl = entry_vega * iv_change * 100  # vega per 1% IV
                if abs(expected_vol_pnl) > 0.001:
                    vol_surface_edge_realized = realized_pnl / abs(expected_vol_pnl)

        # Risk-adjusted return
        max_loss = record.get("max_loss")
        risk_adjusted_return = 0.0
        if max_loss and abs(max_loss) > 0:
            risk_adjusted_return = realized_pnl / abs(max_loss)

        metrics: Dict[str, Any] = {
            "realized_pnl": realized_pnl,
            "realized_return": realized_pnl / abs(record["entry_price"]) if record["entry_price"] else 0.0,
            "iv_prediction_accuracy": iv_prediction_accuracy,
            "greeks_accuracy": greeks_accuracy,
            "theta_captured_pct": theta_captured_pct,
            "vol_surface_edge_realized": vol_surface_edge_realized,
            "risk_adjusted_return": risk_adjusted_return,
            "days_held": max(
                (datetime.fromisoformat(exit_timestamp)
                 - datetime.fromisoformat(record["entry_timestamp"])).total_seconds() / 86400,
                0,
            ),
            "exit_reason": exit_data.get("exit_reason", "unknown"),
        }

        # Finalize record
        record["is_closed"] = True
        record["exit_data"] = {
            "exit_price": exit_price,
            "exit_greeks": exit_data.get("exit_greeks", {}),
            "exit_iv": exit_iv,
            "exit_timestamp": exit_timestamp,
            "exit_reason": exit_data.get("exit_reason", "unknown"),
            "underlying_price_at_exit": exit_data.get("underlying_price_at_exit"),
        }
        record["outcome_metrics"] = metrics

        self._save_trade(tracking_id)

        logger.info(
            f"Closed {tracking_id}: P&L={realized_pnl:.2f}, "
            f"IV_acc={iv_prediction_accuracy:.1%}, "
            f"theta_cap={theta_captured_pct:.1%}, "
            f"vol_edge={vol_surface_edge_realized:.2f}"
        )

        return metrics

    # ==================================================================================
    # Preference generation
    # ==================================================================================

    def generate_option_preferences(self, min_trades: int = 20) -> List[Dict[str, Any]]:
        """
        Create preference pairs specifically for options trades.

        Compares trades with similar setups (same strategy type, similar DTE range)
        and ranks by risk-adjusted return, theta capture efficiency, vol prediction accuracy.

        Args:
            min_trades: Minimum closed trades required to generate preferences.

        Returns:
            List of preference pair dicts.
        """
        closed = [t for t in self._trades.values() if t["is_closed"] and t.get("outcome_metrics")]

        if len(closed) < min_trades:
            logger.info(
                f"Not enough closed option trades for preferences "
                f"({len(closed)} < {min_trades})"
            )
            return []

        logger.info(f"Generating option preferences from {len(closed)} closed trades")

        # Group by strategy type
        by_strategy: Dict[str, List[Dict]] = {}
        for trade in closed:
            st = trade["strategy_type"]
            by_strategy.setdefault(st, []).append(trade)

        pairs: List[Dict[str, Any]] = []

        for strategy_type, trades in by_strategy.items():
            if len(trades) < 2:
                continue

            # Further bucket by DTE range (0-7, 7-30, 30-60, 60+)
            dte_buckets: Dict[str, List[Dict]] = {}
            for t in trades:
                dte = t.get("dte", 0)
                if dte <= 7:
                    bucket = "0-7"
                elif dte <= 30:
                    bucket = "7-30"
                elif dte <= 60:
                    bucket = "30-60"
                else:
                    bucket = "60+"
                dte_buckets.setdefault(bucket, []).append(t)

            for bucket_name, bucket_trades in dte_buckets.items():
                if len(bucket_trades) < 2:
                    continue

                # Score each trade: composite of risk-adj return, theta capture, vol accuracy
                scored = []
                for t in bucket_trades:
                    m = t["outcome_metrics"]
                    composite = (
                        0.4 * _safe_float(m.get("risk_adjusted_return", 0))
                        + 0.3 * _safe_float(m.get("theta_captured_pct", 0))
                        + 0.3 * _safe_float(m.get("iv_prediction_accuracy", 0))
                    )
                    scored.append((composite, t))

                scored.sort(key=lambda x: x[0], reverse=True)

                # Pair top vs bottom
                n_pairs = min(len(scored) // 2, 5)
                for i in range(n_pairs):
                    chosen_score, chosen_trade = scored[i]
                    rejected_score, rejected_trade = scored[-(i + 1)]

                    margin = chosen_score - rejected_score
                    if margin < 0.05:
                        continue

                    pair = {
                        "pair_id": f"opt_pref_{uuid.uuid4().hex[:8]}",
                        "timestamp": datetime.now().isoformat(),
                        "strategy_type": strategy_type,
                        "dte_bucket": bucket_name,
                        "chosen": {
                            "tracking_id": chosen_trade["tracking_id"],
                            "symbol": chosen_trade["symbol"],
                            "outcome_metrics": chosen_trade["outcome_metrics"],
                            "composite_score": chosen_score,
                        },
                        "rejected": {
                            "tracking_id": rejected_trade["tracking_id"],
                            "symbol": rejected_trade["symbol"],
                            "outcome_metrics": rejected_trade["outcome_metrics"],
                            "composite_score": rejected_score,
                        },
                        "margin": margin,
                        "comparison_metric": "options_composite",
                    }
                    pairs.append(pair)

        # Persist preference pairs
        pref_path = self.storage_path / "preference_pairs.json"
        existing: List[Dict] = []
        if pref_path.exists():
            try:
                existing = json.loads(pref_path.read_text())
            except Exception:
                existing = []
        existing.extend(pairs)
        pref_path.write_text(json.dumps(existing, indent=2, default=str))

        logger.info(f"Generated {len(pairs)} option preference pairs")
        return pairs

    # ==================================================================================
    # Performance summary
    # ==================================================================================

    def get_options_performance_summary(self) -> Dict[str, Any]:
        """
        Compute aggregate performance statistics for options trades.

        Returns:
            Dict with keys:
                win_rate_by_strategy, avg_pnl_by_strategy, best_performing_strategies,
                vol_prediction_accuracy, theta_capture_rate, total_trades, total_pnl
        """
        closed = [t for t in self._trades.values() if t["is_closed"] and t.get("outcome_metrics")]

        if not closed:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate_by_strategy": {},
                "avg_pnl_by_strategy": {},
                "best_performing_strategies": [],
                "vol_prediction_accuracy": 0.0,
                "theta_capture_rate": 0.0,
            }

        # Group by strategy
        by_strategy: Dict[str, List[Dict]] = {}
        for t in closed:
            st = t["strategy_type"]
            by_strategy.setdefault(st, []).append(t)

        win_rate_by_strategy: Dict[str, float] = {}
        avg_pnl_by_strategy: Dict[str, float] = {}

        for st, trades in by_strategy.items():
            pnls = [t["outcome_metrics"]["realized_pnl"] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            win_rate_by_strategy[st] = wins / len(pnls) if pnls else 0.0
            avg_pnl_by_strategy[st] = float(np.mean(pnls)) if pnls else 0.0

        # Best performing strategies sorted by avg P&L
        best_performing = sorted(
            avg_pnl_by_strategy.items(), key=lambda x: x[1], reverse=True
        )
        best_performing_strategies = [
            {"strategy": s, "avg_pnl": p} for s, p in best_performing
        ]

        # Aggregate metrics
        all_iv_acc = [
            t["outcome_metrics"]["iv_prediction_accuracy"]
            for t in closed
            if t["outcome_metrics"].get("iv_prediction_accuracy") is not None
        ]
        all_theta_cap = [
            t["outcome_metrics"]["theta_captured_pct"]
            for t in closed
            if t["outcome_metrics"].get("theta_captured_pct") is not None
        ]
        all_pnl = [t["outcome_metrics"]["realized_pnl"] for t in closed]

        return {
            "total_trades": len(closed),
            "total_pnl": float(np.sum(all_pnl)),
            "win_rate_by_strategy": win_rate_by_strategy,
            "avg_pnl_by_strategy": avg_pnl_by_strategy,
            "best_performing_strategies": best_performing_strategies,
            "vol_prediction_accuracy": float(np.mean(all_iv_acc)) if all_iv_acc else 0.0,
            "theta_capture_rate": float(np.mean(all_theta_cap)) if all_theta_cap else 0.0,
        }

    # ==================================================================================
    # Internal helpers
    # ==================================================================================

    def _compute_greeks_accuracy(
        self, record: Dict[str, Any], exit_data: Dict[str, Any]
    ) -> float:
        """
        Compute how accurately the entry Greeks predicted the actual P&L move.

        Compares delta-predicted P&L to actual P&L from underlying move.
        """
        entry_greeks = record.get("entry_greeks", {})
        delta = entry_greeks.get("delta", 0)
        gamma = entry_greeks.get("gamma", 0)

        underlying_at_entry = record.get("underlying_price", 0)
        underlying_at_exit = exit_data.get("underlying_price_at_exit", underlying_at_entry)

        if not underlying_at_entry or not underlying_at_exit:
            return 0.0

        underlying_move = underlying_at_exit - underlying_at_entry

        # Delta-gamma predicted P&L
        predicted_pnl = delta * underlying_move + 0.5 * gamma * underlying_move ** 2
        actual_pnl = exit_data.get("exit_price", 0) - record["entry_price"]

        if abs(actual_pnl) < 0.001:
            return 1.0 if abs(predicted_pnl) < 0.001 else 0.0

        # Accuracy = 1 - |error| / |actual|, clamped to [0, 1]
        error = abs(predicted_pnl - actual_pnl)
        accuracy = max(0.0, 1.0 - error / abs(actual_pnl))
        return accuracy

    def _save_trade(self, tracking_id: str) -> None:
        """Persist a single trade to JSON."""
        record = self._trades.get(tracking_id)
        if record is None:
            return
        path = self.storage_path / f"{tracking_id}.json"
        path.write_text(json.dumps(record, indent=2, default=str))

    def _load_trades(self) -> None:
        """Load all trade JSON files from storage."""
        if not self.storage_path.exists():
            return
        for path in self.storage_path.glob("opt_*.json"):
            try:
                data = json.loads(path.read_text())
                tid = data.get("tracking_id")
                if tid:
                    self._trades[tid] = data
            except Exception as e:
                logger.warning(f"Failed to load trade from {path}: {e}")


# ======================================================================================
# Module-level helpers
# ======================================================================================

def _to_iso(val) -> str:
    """Convert a datetime or string to ISO format string."""
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, str):
        return val
    return datetime.now().isoformat()


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default
