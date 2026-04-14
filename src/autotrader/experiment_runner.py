"""Experiment runner: sandboxed, time-budgeted strategy backtesting.

Core autoresearch-mlx adaptation. Each experiment:
1. Writes the generated signal function to a temp file
2. Executes a backtest in a subprocess with timeout
3. Computes composite metric
4. Returns ExperimentResult with keep/discard recommendation

The subprocess sandbox restricts imports and prevents side effects.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import CompositeMetric, ExperimentResult
from .safety import SafetyGuard
from .strategy_spec import StrategySpec
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Template for the sandboxed backtest subprocess
_SANDBOX_SCRIPT = '''\
"""Sandboxed backtest subprocess.

Reads strategy spec from stdin, runs backtest, writes result JSON to stdout.
This runs in a separate process with restricted imports.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def main():
    spec_json = sys.stdin.read()
    spec = json.loads(spec_json)

    signal_code = spec["signal_code"]
    parameters = spec.get("parameters", {})
    ohlcv_json = spec["ohlcv_data"]
    commission_bps = spec.get("commission_bps", 5.0)
    slippage_bps = spec.get("slippage_bps", 2.0)

    # Reconstruct DataFrame
    df = pd.DataFrame(ohlcv_json)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # Execute the generated signal function
    exec_globals = {"pd": pd, "np": np, "math": __import__("math")}
    exec(signal_code, exec_globals)
    generate_signals = exec_globals["generate_signals"]

    signals = generate_signals(df, parameters)
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=df.index)

    # Run backtest
    df["signal"] = signals.reindex(df.index).fillna(0).astype(int)
    df["next_return"] = df["close"].pct_change().shift(-1).fillna(0.0)

    # Apply transaction costs
    gross_return = df["signal"].astype(float) * df["next_return"].astype(float)
    turnover = df["signal"].astype(float).diff().abs().fillna(df["signal"].astype(float).abs())
    total_cost = turnover * ((commission_bps + slippage_bps) / 10_000.0)
    net_return = gross_return - total_cost

    # Equity curve
    equity = (1.0 + net_return).cumprod()

    # Compute metrics
    signaled = df[df["signal"] != 0]
    n_trades = int(turnover[turnover > 0].sum())
    hit_rate = 0.0
    avg_trade_return = 0.0
    if not signaled.empty:
        trade_returns = signaled["signal"].astype(float) * signaled["next_return"]
        hit_rate = float((trade_returns > 0).mean())
        avg_trade_return = float(trade_returns.mean())

    cumulative_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

    # Sharpe ratio (annualized)
    daily_returns = net_return[net_return != 0]
    if len(daily_returns) > 1 and daily_returns.std() > 1e-10:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino ratio
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 1e-10:
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = sharpe

    # Max drawdown
    running_max = equity.cummax()
    drawdowns = (equity / running_max.clip(lower=1e-12)) - 1.0
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    # Calmar ratio
    calmar = 0.0
    if abs(max_drawdown) > 1e-10:
        annual_return = cumulative_return  # approximate for sub-year
        calmar = float(annual_return / abs(max_drawdown))

    result = {
        "sharpe_ratio": round(sharpe, 6),
        "sortino_ratio": round(sortino, 6),
        "calmar_ratio": round(calmar, 6),
        "max_drawdown": round(max_drawdown, 6),
        "hit_rate": round(hit_rate, 6),
        "cumulative_return": round(cumulative_return, 6),
        "num_trades": n_trades,
        "avg_trade_return": round(avg_trade_return, 8),
        "equity_curve_final": round(float(equity.iloc[-1]), 6) if not equity.empty else 1.0,
        "total_bars": len(df),
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


class ExperimentRunner:
    """Run time-budgeted strategy experiments in sandboxed subprocesses."""

    def __init__(
        self,
        data_client: Any = None,
        preprocessor: Any = None,
        technical_engine: Any = None,
        composite_metric: Optional[CompositeMetric] = None,
        safety: Optional[SafetyGuard] = None,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        default_symbols: Optional[List[str]] = None,
        default_lookback_months: int = 6,
    ):
        self.data_client = data_client
        self.preprocessor = preprocessor
        self.technical_engine = technical_engine
        self.metric = composite_metric or CompositeMetric()
        self.safety = safety or SafetyGuard()
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.default_symbols = default_symbols or ["AAPL", "MSFT", "SPY"]
        self.default_lookback_months = default_lookback_months

    def _get_backtest_data(
        self,
        symbols: Optional[List[str]] = None,
        lookback_months: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and prepare OHLCV + features data for backtesting."""
        symbols = symbols or self.default_symbols
        lookback = lookback_months or self.default_lookback_months

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (
            pd.Timestamp.now() - pd.DateOffset(months=lookback)
        ).strftime("%Y-%m-%d")

        datasets = {}
        for symbol in symbols:
            try:
                if self.data_client:
                    history = self.data_client.download_bars(
                        symbols=symbol,
                        start=start_date,
                        end=end_date,
                        timeframe="1Day",
                    )
                    if self.preprocessor:
                        history = self.preprocessor.preprocess(history, symbol=symbol)
                    if self.technical_engine:
                        history = self.technical_engine.compute_all(history)
                    datasets[symbol] = history
                else:
                    # Generate synthetic data for testing
                    datasets[symbol] = self._generate_synthetic_data(
                        symbol, start_date, end_date
                    )
            except Exception as e:
                logger.warning("Failed to fetch data for %s: %s", symbol, e)

        return datasets

    @staticmethod
    def _generate_synthetic_data(
        symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing when no data client is available."""
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        np.random.seed(hash(symbol) % 2**31)

        returns = np.random.normal(0.0005, 0.02, n)
        close = 100.0 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": close * (1 + np.random.normal(0, 0.005, n)),
                "high": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
                "low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
                "close": close,
                "volume": np.random.lognormal(15, 1, n).astype(int),
            },
            index=dates,
        )

        # Add basic technical features
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["rsi"] = _compute_rsi(df["close"], 14)
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["atr"] = _compute_atr(df, 14)
        df["adx"] = 25.0  # placeholder
        df["obv"] = (df["volume"] * np.sign(df["close"].diff())).cumsum()
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        return df.dropna()

    def run_experiment(
        self,
        spec: StrategySpec,
        time_budget_seconds: int = 300,
        symbols: Optional[List[str]] = None,
    ) -> ExperimentResult:
        """Run a single experiment: backtest the strategy spec in a sandbox.

        Parameters
        ----------
        spec : StrategySpec
            The strategy to test (with signal_code).
        time_budget_seconds : int
            Max wall-clock seconds for the backtest subprocess.
        symbols : list[str], optional
            Override default symbols.

        Returns
        -------
        ExperimentResult
            Contains composite score, component metrics, and status.
        """
        result = ExperimentResult(
            spec_id=spec.spec_id,
            description=spec.description,
        )
        start_time = time.time()

        # Validate code safety
        valid, reason = self.safety.validate_signal_code(spec.signal_code)
        if not valid:
            result.status = "crash"
            result.error = f"Code validation failed: {reason}"
            result.backtest_duration_seconds = time.time() - start_time
            logger.warning("Experiment rejected by safety guard: %s", reason)
            return result

        # Check rate limits
        rate_ok, rate_msg = self.safety.check_experiment_rate()
        if not rate_ok:
            result.status = "crash"
            result.error = rate_msg
            result.backtest_duration_seconds = time.time() - start_time
            return result

        # Fetch data
        try:
            datasets = self._get_backtest_data(symbols=symbols)
        except Exception as e:
            result.status = "crash"
            result.error = f"Data fetch failed: {e}"
            result.backtest_duration_seconds = time.time() - start_time
            self.safety.record_crash()
            return result

        if not datasets:
            result.status = "crash"
            result.error = "No data available for any symbol"
            result.backtest_duration_seconds = time.time() - start_time
            self.safety.record_crash()
            return result

        # Run backtest per symbol in sandbox
        all_metrics: List[Dict[str, float]] = []
        for symbol, df in datasets.items():
            try:
                symbol_result = self._run_sandboxed_backtest(
                    spec, df, symbol, time_budget_seconds
                )
                if symbol_result:
                    all_metrics.append(symbol_result)
            except Exception as e:
                logger.warning("Backtest failed for %s: %s", symbol, e)

        if not all_metrics:
            result.status = "crash"
            result.error = "All symbol backtests failed"
            result.backtest_duration_seconds = time.time() - start_time
            self.safety.record_crash()
            return result

        # Aggregate metrics across symbols
        agg = self._aggregate_metrics(all_metrics)
        result.sharpe_ratio = agg["sharpe_ratio"]
        result.sortino_ratio = agg["sortino_ratio"]
        result.calmar_ratio = agg["calmar_ratio"]
        result.max_drawdown = agg["max_drawdown"]
        result.hit_rate = agg["hit_rate"]
        result.cumulative_return = agg["cumulative_return"]
        result.num_trades = agg["num_trades"]
        result.avg_trade_return = agg["avg_trade_return"]
        result.composite_score = self.metric.compute(agg)
        result.backtest_duration_seconds = time.time() - start_time
        result.backtest_details = {"per_symbol": all_metrics, "aggregate": agg}
        result.status = "pending"  # caller decides keep/discard

        self.safety.record_experiment()
        self.safety.record_success()
        logger.info(
            "Experiment %s completed: composite=%.6f sharpe=%.4f return=%.4f dd=%.4f [%.1fs]",
            result.experiment_id,
            result.composite_score,
            result.sharpe_ratio,
            result.cumulative_return,
            result.max_drawdown,
            result.backtest_duration_seconds,
        )
        return result

    def _run_sandboxed_backtest(
        self,
        spec: StrategySpec,
        df: pd.DataFrame,
        symbol: str,
        timeout: int,
    ) -> Optional[Dict[str, float]]:
        """Run backtest in a subprocess sandbox."""
        # Prepare data for subprocess
        df_for_json = df.reset_index()
        # Convert timestamps to strings for JSON serialization
        for col in df_for_json.columns:
            if pd.api.types.is_datetime64_any_dtype(df_for_json[col]):
                df_for_json[col] = df_for_json[col].astype(str)

        payload = json.dumps({
            "signal_code": spec.signal_code,
            "parameters": spec.parameters,
            "ohlcv_data": df_for_json.to_dict(orient="list"),
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
        })

        # Write sandbox script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="autotrader_sandbox_"
        ) as f:
            f.write(_SANDBOX_SCRIPT)
            sandbox_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, sandbox_path],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode != 0:
                logger.warning(
                    "Sandbox backtest failed for %s: %s",
                    symbol,
                    proc.stderr[:500] if proc.stderr else "unknown error",
                )
                return None

            result = json.loads(proc.stdout.strip())
            result["symbol"] = symbol
            return result

        except subprocess.TimeoutExpired:
            logger.warning("Sandbox backtest timed out for %s (%ds)", symbol, timeout)
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Sandbox backtest error for %s: %s", symbol, e)
            return None
        finally:
            Path(sandbox_path).unlink(missing_ok=True)

    @staticmethod
    def _aggregate_metrics(
        per_symbol: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Average metrics across symbols."""
        if not per_symbol:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "cumulative_return": 0.0,
                "num_trades": 0,
                "avg_trade_return": 0.0,
            }

        return {
            "sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in per_symbol])),
            "sortino_ratio": float(np.mean([m.get("sortino_ratio", 0) for m in per_symbol])),
            "calmar_ratio": float(np.mean([m.get("calmar_ratio", 0) for m in per_symbol])),
            "max_drawdown": float(np.min([m["max_drawdown"] for m in per_symbol])),
            "hit_rate": float(np.mean([m["hit_rate"] for m in per_symbol])),
            "cumulative_return": float(np.mean([m["cumulative_return"] for m in per_symbol])),
            "num_trades": int(np.sum([m["num_trades"] for m in per_symbol])),
            "avg_trade_return": float(np.mean([m["avg_trade_return"] for m in per_symbol])),
        }

    def run_experiment_inline(
        self,
        spec: StrategySpec,
        df: pd.DataFrame,
    ) -> ExperimentResult:
        """Run experiment inline (no subprocess) for testing.

        WARNING: No sandboxing. Only use for trusted code during development.
        """
        result = ExperimentResult(spec_id=spec.spec_id, description=spec.description)
        start_time = time.time()

        try:
            exec_globals = {"pd": pd, "np": np, "math": __import__("math")}
            exec(spec.signal_code, exec_globals)
            generate_signals = exec_globals["generate_signals"]

            signals = generate_signals(df, spec.parameters)
            if not isinstance(signals, pd.Series):
                signals = pd.Series(signals, index=df.index)

            df = df.copy()
            df["signal"] = signals.reindex(df.index).fillna(0).astype(int)
            df["next_return"] = df["close"].pct_change().shift(-1).fillna(0.0)

            gross = df["signal"].astype(float) * df["next_return"]
            turnover = df["signal"].astype(float).diff().abs().fillna(
                df["signal"].astype(float).abs()
            )
            cost = turnover * ((self.commission_bps + self.slippage_bps) / 10_000.0)
            net = gross - cost
            equity = (1.0 + net).cumprod()

            signaled = df[df["signal"] != 0]
            if not signaled.empty:
                trade_returns = signaled["signal"].astype(float) * signaled["next_return"]
                result.hit_rate = float((trade_returns > 0).mean())
                result.avg_trade_return = float(trade_returns.mean())

            result.cumulative_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
            result.num_trades = int(turnover[turnover > 0].sum())

            daily = net[net != 0]
            if len(daily) > 1 and daily.std() > 1e-10:
                result.sharpe_ratio = float(daily.mean() / daily.std() * np.sqrt(252))
            result.sortino_ratio = result.sharpe_ratio

            running_max = equity.cummax()
            dd = (equity / running_max.clip(lower=1e-12)) - 1.0
            result.max_drawdown = float(dd.min()) if not dd.empty else 0.0

            result.composite_score = self.metric.compute({
                "sharpe_ratio": result.sharpe_ratio,
                "cumulative_return": result.cumulative_return,
                "max_drawdown": result.max_drawdown,
                "hit_rate": result.hit_rate,
            })
            result.status = "pending"

        except Exception as e:
            result.status = "crash"
            result.error = f"{type(e).__name__}: {e}"

        result.backtest_duration_seconds = time.time() - start_time
        return result


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()
