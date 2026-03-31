import pandas as pd

from src.backtesting.runner import BenchmarkBacktestRunner


class FakeDataClient:
    def download_bars(self, symbols, start, end, timeframe="1Day"):
        idx = pd.date_range("2024-01-01", periods=80, freq="D")
        base = pd.Series(range(100, 180), index=idx, dtype=float)
        return pd.DataFrame(
            {
                "open": base,
                "high": base + 1,
                "low": base - 1,
                "close": base,
                "volume": 1000,
            },
            index=idx,
        )


class FakePreprocessor:
    def preprocess(self, df, symbol=None):
        return df


class FakeTechnicalEngine:
    def compute_all(self, df):
        frame = df.copy()
        frame["sma_20"] = frame["close"].rolling(20).mean()
        frame["sma_50"] = frame["close"].rolling(50).mean()
        frame["macd_hist"] = 1.0
        frame["rsi"] = 60.0
        return frame.bfill()



def test_benchmark_backtest_runner_returns_metrics_and_baselines():
    runner = BenchmarkBacktestRunner(
        data_client=FakeDataClient(),
        preprocessor=FakePreprocessor(),
        technical_engine=FakeTechnicalEngine(),
        commission_bps=1.0,
        slippage_bps=2.0,
    )

    result = runner.run(["AAPL", "MSFT"], "2024-01-01", "2024-03-31")

    assert result["metrics"]["symbols_processed"] == 2
    assert result["metrics"]["signals_generated"] > 0
    assert "strategy_cumulative_return" in result["metrics"]
    assert "buy_and_hold_cumulative_return" in result["metrics"]
    assert "sma_cross_cumulative_return" in result["metrics"]
    assert "alpha_vs_buy_hold" in result["metrics"]
    assert len(result["trades"]) == 2
    assert result["trades"][0]["rows_processed"] == 80
    assert result["metrics"]["commission_bps"] == 1.0
    assert result["metrics"]["slippage_bps"] == 2.0
