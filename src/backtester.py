"""
Backtesting engine — validate before you risk real money.

Simulates the full pipeline against historical data:
- Walk-forward (no lookahead)
- Realistic transaction costs
- Position sizing from risk manager
- Full P&L, Sharpe, drawdown, win rate
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .agents import ManagerAgent, ClaudeClient
from .data import AlpacaDataClient
from .features.technical import TechnicalFeatureEngine
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

        # Build equity curve from trades
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
        print("=" * 60)


class Backtester:
    """
    Walk-forward backtester.

    Uses the multi-agent system to generate signals on historical data,
    then simulates execution with transaction costs.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        signal_interval_days: int = 5,
        use_agents: bool = True,
        risk_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            symbols: Stocks to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting cash
            transaction_cost_pct: Cost per trade as fraction (0.001 = 0.1%)
            signal_interval_days: Days between signal generation
            use_agents: If True, use Claude agents. If False, use technical-only signals.
            risk_config: Risk manager config overrides
        """
        load_dotenv()

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.signal_interval_days = signal_interval_days
        self.use_agents = use_agents

        self.data_client = AlpacaDataClient()
        self.tech_engine = TechnicalFeatureEngine()

        if use_agents:
            self.claude_client = ClaudeClient()
            self.manager = ManagerAgent(
                claude_client=self.claude_client, debate_rounds=1
            )
        else:
            self.manager = None

        risk_config = risk_config or {}
        self.risk_manager = RiskManager(**risk_config)

    def run(self) -> BacktestResult:
        """Run the backtest."""
        logger.info(
            f"Backtest: {self.symbols}, {self.start_date} to {self.end_date}, "
            f"${self.initial_capital:,.0f}, agents={'ON' if self.use_agents else 'OFF'}"
        )

        result = BacktestResult()
        capital = self.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {qty, entry_price, entry_date}

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

                latest = available.iloc[-1]
                current_price = float(latest["close"])

                # Generate signal
                if self.use_agents:
                    signal = self._agent_signal(symbol, available, latest)
                else:
                    signal = self._technical_signal(latest)

                action = signal["action"]
                score = signal["score"]
                confidence = signal["confidence"]

                # Execute
                if action == "buy" and symbol not in positions:
                    # Position sizing
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
                            }
                            logger.info(
                                f"  BUY {qty} {symbol} @ ${current_price:.2f} "
                                f"(score={score:.2f})"
                            )

                elif action == "sell" and symbol in positions:
                    pos = positions.pop(symbol)
                    proceeds = (
                        pos["qty"] * current_price * (1 - self.transaction_cost_pct)
                    )
                    pnl = proceeds - (pos["qty"] * pos["entry_price"])
                    capital += proceeds

                    result.add_trade(
                        {
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
                        }
                    )

                    logger.info(
                        f"  SELL {pos['qty']} {symbol} @ ${current_price:.2f} "
                        f"P&L=${pnl:,.2f} ({pnl / (pos['qty'] * pos['entry_price']):.1%})"
                    )

        # Close any remaining positions at end
        for symbol, pos in list(positions.items()):
            df = all_data[symbol]
            if len(df) > 0:
                final_price = float(df.iloc[-1]["close"])
                proceeds = pos["qty"] * final_price * (1 - self.transaction_cost_pct)
                pnl = proceeds - (pos["qty"] * pos["entry_price"])
                capital += proceeds

                result.add_trade(
                    {
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
                    }
                )

        result.print_report(self.initial_capital)
        return result

    def _agent_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        latest: pd.Series,
    ) -> Dict[str, Any]:
        """Use multi-agent system for signal."""
        # Build agent input (same as pipeline)
        def safe_float(val, default=0.0):
            try:
                v = float(val)
                return v if v == v else default
            except (ValueError, TypeError):
                return default

        technical = {
            "current_price": safe_float(latest.get("close")),
            "rsi": safe_float(latest.get("rsi", 50)),
            "macd": safe_float(latest.get("macd", 0)),
            "trend": "uptrend" if safe_float(latest.get("close")) > safe_float(latest.get("sma_50", 0)) else "downtrend",
        }

        returns = df["close"].pct_change().dropna()
        risk = {
            "volatility": safe_float(returns.std() * (252 ** 0.5)),
            "max_drawdown": 0.15,
            "sharpe_ratio": safe_float(
                returns.mean() / (returns.std() + 1e-9) * (252 ** 0.5)
            ),
            "beta": 1.0,
        }

        agent_data = {
            "technical": technical,
            "risk": risk,
            "sentiment": {"news_sentiment": 0.0, "social_sentiment": 0.0},
            "fundamentals": {"symbol": symbol},
        }

        try:
            response = self.manager.analyze(symbol=symbol, data=agent_data)
            action = "buy" if response.score >= 0.3 else "sell" if response.score <= -0.3 else "hold"
            return {
                "action": action,
                "score": response.score,
                "confidence": response.confidence,
            }
        except Exception as e:
            logger.error(f"Agent signal failed for {symbol}: {e}")
            return self._technical_signal(latest)

    def _technical_signal(self, latest: pd.Series) -> Dict[str, Any]:
        """Simple technical signal (no Claude API calls)."""
        score = 0.0
        signals = 0

        # RSI
        rsi = float(latest.get("rsi", 50))
        if rsi < 30:
            score += 0.5
            signals += 1
        elif rsi > 70:
            score -= 0.5
            signals += 1

        # MACD
        macd_hist = float(latest.get("macd_hist", 0))
        if macd_hist > 0:
            score += 0.3
            signals += 1
        elif macd_hist < 0:
            score -= 0.3
            signals += 1

        # Price vs SMA
        close = float(latest.get("close", 0))
        sma_50 = float(latest.get("sma_50", close))
        sma_200 = float(latest.get("sma_200", close))

        if close > sma_50 > sma_200:
            score += 0.4
            signals += 1
        elif close < sma_50 < sma_200:
            score -= 0.4
            signals += 1

        # Normalize
        if signals > 0:
            score /= signals

        confidence = min(abs(score) + 0.3, 1.0)
        action = "buy" if score >= 0.3 else "sell" if score <= -0.3 else "hold"

        return {"action": action, "score": score, "confidence": confidence}
