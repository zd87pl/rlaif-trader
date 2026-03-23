"""
Backtesting engine — validate before you risk real money.

Simulates the full pipeline against historical data:
- Walk-forward (no lookahead)
- Multi-strategy ensemble
- Realistic transaction costs
- Position sizing from risk manager
- Full P&L, Sharpe, drawdown, win rate
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .data import AlpacaDataClient
from .features.technical import TechnicalFeatureEngine
from .strategies import (
    Strategy,
    MomentumStrategy,
    MeanReversionStrategy,
    AgentStrategy,
    StrategyEnsemble,
    Signal,
)
from .execution.risk_manager import RiskManager
from .utils.logging import get_logger

logger = get_logger(__name__)


class BacktestResult:
    """Holds backtest results and computes metrics."""

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def add_trade(self, trade: Dict[str, Any]) -> None:
        self.trades.append(trade)

    def compute_metrics(self, initial_capital: float) -> Dict[str, Any]:
        if not self.trades:
            return {"error": "no trades"}

        capital = initial_capital
        peak = capital
        max_dd = 0.0
        returns = []
        wins = 0
        losses = 0

        for t in self.trades:
            pnl = t["pnl"]
            capital += pnl
            returns.append(pnl / (capital - pnl) if (capital - pnl) > 0 else 0)

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

        returns_arr = np.array(returns)
        total_return = (capital - initial_capital) / initial_capital

        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252)

        sortino = 0.0
        downside = returns_arr[returns_arr < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns_arr.mean() / downside.std() * np.sqrt(252)

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0

        winning_trades = [t["pnl"] for t in self.trades if t["pnl"] > 0]
        losing_trades = [t["pnl"] for t in self.trades if t["pnl"] < 0]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        profit_factor = (
            sum(winning_trades) / abs(sum(losing_trades))
            if losing_trades
            else float("inf")
        )

        # Strategy breakdown
        strategy_stats = {}
        for t in self.trades:
            sname = t.get("strategy", "unknown")
            if sname not in strategy_stats:
                strategy_stats[sname] = {"trades": 0, "pnl": 0, "wins": 0}
            strategy_stats[sname]["trades"] += 1
            strategy_stats[sname]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                strategy_stats[sname]["wins"] += 1

        return {
            "initial_capital": initial_capital,
            "final_capital": capital,
            "total_return": total_return,
            "total_pnl": capital - initial_capital,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t["pnl"] for t in self.trades]),
            "strategy_breakdown": strategy_stats,
        }

    def print_report(self, initial_capital: float) -> None:
        metrics = self.compute_metrics(initial_capital)
        if "error" in metrics:
            print("No trades to report.")
            return

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Initial Capital:   ${metrics['initial_capital']:>12,.2f}")
        print(f"  Final Capital:     ${metrics['final_capital']:>12,.2f}")
        print(f"  Total Return:       {metrics['total_return']:>11.2%}")
        print(f"  Total P&L:         ${metrics['total_pnl']:>12,.2f}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>11.2f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>11.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>11.2%}")
        print(f"  Total Trades:       {metrics['total_trades']:>11d}")
        print(f"  Win Rate:           {metrics['win_rate']:>11.1%}")
        print(f"  Avg Win:           ${metrics['avg_win']:>12,.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:>12,.2f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:>11.2f}")

        # Strategy breakdown
        strat_stats = metrics.get("strategy_breakdown", {})
        if strat_stats:
            print(f"\n  {'STRATEGY BREAKDOWN':^40}")
            print(f"  {'-' * 40}")
            for sname, stats in strat_stats.items():
                wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                print(
                    f"  {sname:20s}  trades={stats['trades']:3d}  "
                    f"P&L=${stats['pnl']:>10,.2f}  win={wr:.0f}%"
                )

        print("=" * 60)


class Backtester:
    """
    Walk-forward backtester with multi-strategy support.

    Can test individual strategies or the full ensemble.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        signal_interval_days: int = 5,
        strategies: Optional[List[str]] = None,
        ensemble_mode: str = "conviction_weighted",
        use_agents: bool = False,
        risk_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            symbols: Stocks to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting cash
            transaction_cost_pct: Cost per trade as fraction
            signal_interval_days: Days between signal generation
            strategies: List of strategy names to include
                Options: "momentum", "mean_reversion", "agent"
                Default: ["momentum", "mean_reversion"]
            ensemble_mode: How to combine strategies
            use_agents: If True, include Claude agent strategy (costs API credits)
            risk_config: Risk manager config overrides
        """
        load_dotenv()

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.signal_interval_days = signal_interval_days

        self.data_client = AlpacaDataClient()
        self.tech_engine = TechnicalFeatureEngine()

        # Build strategy list
        strategy_names = strategies or ["momentum", "mean_reversion"]
        if use_agents and "agent" not in strategy_names:
            strategy_names.append("agent")

        strategy_objects = []
        for name in strategy_names:
            if name == "momentum":
                strategy_objects.append(MomentumStrategy(weight=1.0))
            elif name == "mean_reversion":
                strategy_objects.append(MeanReversionStrategy(weight=0.8))
            elif name == "agent":
                strategy_objects.append(AgentStrategy(weight=1.5))

        self.ensemble = StrategyEnsemble(
            strategies=strategy_objects,
            mode=ensemble_mode,
        )

        risk_config = risk_config or {}
        self.risk_manager = RiskManager(**risk_config)

    def run(self) -> BacktestResult:
        """Run the backtest."""
        logger.info(
            f"Backtest: {self.symbols}, {self.start_date} to {self.end_date}, "
            f"${self.initial_capital:,.0f}, "
            f"strategies={[s.name for s in self.ensemble.strategies]}, "
            f"ensemble={self.ensemble.mode}"
        )

        result = BacktestResult()
        capital = self.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}

        # Download all data upfront
        all_data = {}
        for symbol in self.symbols:
            df = self.data_client.download_bars(
                symbols=symbol,
                start=self.start_date,
                end=self.end_date,
                timeframe="1Day",
            )
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            all_data[symbol] = self.tech_engine.compute_all(df)
            logger.info(f"Loaded {len(all_data[symbol])} bars for {symbol}")

        # Walk forward
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        for i, date in enumerate(dates):
            if i % self.signal_interval_days != 0:
                continue

            for symbol in self.symbols:
                df = all_data[symbol]

                # Only use data up to current date (no lookahead)
                available = df[df.index <= date]
                if len(available) < 50:
                    continue

                current_price = float(available.iloc[-1]["close"])

                # Generate ensemble signal
                signal = self.ensemble.generate_signal(
                    symbol=symbol,
                    price_data=available,
                    features=available,
                    news_data=None,
                    fundamental_data=None,
                )

                action = signal.action
                score = signal.score
                confidence = signal.confidence

                # Execute
                if action == "buy" and symbol not in positions:
                    position_size = capital * self.risk_manager.max_position_pct
                    position_size *= min(confidence, 1.0) * min(abs(score), 1.0)
                    qty = int(position_size / current_price)

                    if qty >= 1:
                        cost = qty * current_price * (1 + self.transaction_cost_pct)
                        if cost <= capital:
                            capital -= cost
                            positions[symbol] = {
                                "qty": qty,
                                "entry_price": current_price,
                                "entry_date": date,
                                "score": score,
                                "confidence": confidence,
                                "strategy": signal.strategy_name,
                            }
                            logger.info(
                                f"  BUY {qty} {symbol} @ ${current_price:.2f} "
                                f"({signal.strategy_name} score={score:+.2f})"
                            )

                elif action == "sell" and symbol in positions:
                    pos = positions.pop(symbol)
                    proceeds = pos["qty"] * current_price * (1 - self.transaction_cost_pct)
                    pnl = proceeds - (pos["qty"] * pos["entry_price"])
                    capital += proceeds

                    result.add_trade({
                        "symbol": symbol,
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(date.date()),
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "qty": pos["qty"],
                        "pnl": pnl,
                        "return": pnl / (pos["qty"] * pos["entry_price"]),
                        "hold_days": (date - pos["entry_date"]).days,
                        "entry_score": pos["score"],
                        "strategy": pos.get("strategy", "unknown"),
                    })

                    logger.info(
                        f"  SELL {pos['qty']} {symbol} @ ${current_price:.2f} "
                        f"P&L=${pnl:,.2f} ({pnl / (pos['qty'] * pos['entry_price']):.1%})"
                    )

        # Close remaining positions
        for symbol, pos in list(positions.items()):
            df = all_data[symbol]
            if len(df) > 0:
                final_price = float(df.iloc[-1]["close"])
                proceeds = pos["qty"] * final_price * (1 - self.transaction_cost_pct)
                pnl = proceeds - (pos["qty"] * pos["entry_price"])
                capital += proceeds

                result.add_trade({
                    "symbol": symbol,
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": self.end_date,
                    "entry_price": pos["entry_price"],
                    "exit_price": final_price,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "return": pnl / (pos["qty"] * pos["entry_price"]),
                    "hold_days": 0,
                    "entry_score": pos["score"],
                    "strategy": pos.get("strategy", "unknown"),
                })

        result.print_report(self.initial_capital)
        return result
