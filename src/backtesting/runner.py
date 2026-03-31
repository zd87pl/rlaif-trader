"""Deterministic benchmark backtesting runner for the stabilized MVP path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkBacktestRunner:
    data_client: Optional[Any] = None
    preprocessor: Optional[Any] = None
    technical_engine: Optional[Any] = None
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.data_client is None:
            from ..data.ingestion.market_data import AlpacaDataClient

            self.data_client = AlpacaDataClient()
        if self.preprocessor is None:
            from ..data.processing.preprocessor import DataPreprocessor

            self.preprocessor = DataPreprocessor(min_trading_days=1)
        if self.technical_engine is None:
            from ..features.technical import TechnicalFeatureEngine

            self.technical_engine = TechnicalFeatureEngine()

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        benchmark_curves = {
            "strategy": [1.0],
            "buy_and_hold": [1.0],
            "sma_cross": [1.0],
        }

        for symbol in symbols:
            try:
                history = self.data_client.download_bars(
                    symbols=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe=timeframe,
                )
                processed = self.preprocessor.preprocess(history, symbol=symbol)
                enriched = self.technical_engine.compute_all(processed)
                symbol_result = self._evaluate_symbol(symbol, enriched)
                results.append(symbol_result)
                for key in benchmark_curves:
                    benchmark_curves[key].extend(symbol_result["equity_curves"][key][1:])
            except Exception as exc:
                logger.warning("Benchmark backtest error on %s: %s", symbol, exc)
                results.append({"symbol": symbol, "error": str(exc)})

        successful = [r for r in results if "error" not in r]
        signals_generated = int(sum(r.get("signals_generated", 0) for r in successful))
        total_rows = int(sum(r.get("rows_processed", 0) for r in successful))
        avg_hit_rate = (
            float(np.mean([r.get("hit_rate", 0.0) for r in successful]))
            if successful
            else 0.0
        )

        metrics = {
            "symbols_processed": len(successful),
            "symbols_requested": len(symbols),
            "signals_generated": signals_generated,
            "rows_processed": total_rows,
            "avg_hit_rate": round(avg_hit_rate, 4),
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
        }

        for name, curve in benchmark_curves.items():
            total_return = curve[-1] - 1.0 if curve else 0.0
            max_drawdown = self._max_drawdown(np.array(curve, dtype=float))
            metrics[f"{name}_cumulative_return"] = round(float(total_return), 6)
            metrics[f"{name}_max_drawdown"] = round(float(max_drawdown), 6)

        metrics["alpha_vs_buy_hold"] = round(
            metrics["strategy_cumulative_return"] - metrics["buy_and_hold_cumulative_return"],
            6,
        )
        metrics["alpha_vs_sma_cross"] = round(
            metrics["strategy_cumulative_return"] - metrics["sma_cross_cumulative_return"],
            6,
        )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "trades": results,
            "equity_curve": benchmark_curves["strategy"],
            "benchmark_curves": benchmark_curves,
            "metrics": metrics,
        }

    def _evaluate_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        frame = df.copy()
        frame["next_return"] = frame["close"].pct_change().shift(-1).fillna(0.0)

        frame["strategy_signal"] = (
            (frame["close"] > frame.get("sma_20"))
            & (frame.get("sma_20") > frame.get("sma_50"))
            & (frame.get("macd_hist") > 0)
            & (frame.get("rsi") > 50)
        ).astype(int)
        frame["buy_hold_signal"] = 1
        frame["sma_cross_signal"] = (
            (frame["close"] > frame.get("sma_20")) & (frame.get("sma_20") > frame.get("sma_50"))
        ).astype(int)

        strategy_returns = self._apply_costs(frame["strategy_signal"], frame["next_return"])
        buy_hold_returns = self._apply_costs(frame["buy_hold_signal"], frame["next_return"])
        sma_cross_returns = self._apply_costs(frame["sma_cross_signal"], frame["next_return"])

        frame["strategy_equity"] = (1.0 + strategy_returns).cumprod()
        frame["buy_hold_equity"] = (1.0 + buy_hold_returns).cumprod()
        frame["sma_cross_equity"] = (1.0 + sma_cross_returns).cumprod()

        signaled = frame[frame["strategy_signal"] == 1]
        hit_rate = float((signaled["next_return"] > 0).mean()) if not signaled.empty else 0.0
        avg_next_return = float(signaled["next_return"].mean()) if not signaled.empty else 0.0

        return {
            "symbol": symbol,
            "rows_processed": int(len(frame)),
            "signals_generated": int(frame["strategy_signal"].sum()),
            "hit_rate": round(hit_rate, 4),
            "avg_next_return": round(avg_next_return, 6),
            "strategy_cumulative_return": round(float(frame["strategy_equity"].iloc[-1] - 1.0), 6) if not frame.empty else 0.0,
            "buy_hold_cumulative_return": round(float(frame["buy_hold_equity"].iloc[-1] - 1.0), 6) if not frame.empty else 0.0,
            "sma_cross_cumulative_return": round(float(frame["sma_cross_equity"].iloc[-1] - 1.0), 6) if not frame.empty else 0.0,
            "equity_curves": {
                "strategy": frame["strategy_equity"].round(6).tolist() if not frame.empty else [1.0],
                "buy_and_hold": frame["buy_hold_equity"].round(6).tolist() if not frame.empty else [1.0],
                "sma_cross": frame["sma_cross_equity"].round(6).tolist() if not frame.empty else [1.0],
            },
        }

    def _apply_costs(self, signal: pd.Series, returns: pd.Series) -> pd.Series:
        gross = signal.astype(float) * returns.astype(float)
        turnover = signal.astype(float).diff().abs().fillna(signal.astype(float))
        total_cost_bps = self.commission_bps + self.slippage_bps
        cost = turnover * (total_cost_bps / 10_000.0)
        return gross - cost

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if equity.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity / np.maximum(running_max, 1e-12)) - 1.0
        return float(drawdowns.min())


def run_benchmark_backtest(
    data_client: Any,
    preprocessor: Any,
    technical_engine: Any,
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1Day",
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Dict[str, Any]:
    runner = BenchmarkBacktestRunner(
        data_client=data_client,
        preprocessor=preprocessor,
        technical_engine=technical_engine,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    return runner.run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run deterministic benchmark backtest")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    runner = BenchmarkBacktestRunner(
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    result = runner.run(symbols, args.start, args.end, timeframe=args.timeframe)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
