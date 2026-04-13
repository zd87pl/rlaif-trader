from pathlib import Path

import pandas as pd

from main import TradingPipeline
from src.execution.broker import PaperBroker
from src.execution.oms import OrderManagementSystem
from src.execution.risk_engine import RiskEngine
from src.execution.scheduler import TradingScheduler


class FakePipeline:
    def analyze(self, symbol):
        if symbol == "AAPL":
            return {
                "symbol": symbol,
                "manager_decision": {"decision": "buy", "confidence": 0.9},
                "agents": {},
            }
        return {
            "symbol": symbol,
            "manager_decision": {"decision": "hold", "confidence": 0.2},
            "agents": {},
        }


class FakePreferenceGenerator:
    def __init__(self):
        self.decisions = []

    def record_decision(self, **kwargs):
        decision = type("Decision", (), kwargs)()
        self.decisions.append(decision)
        return decision


class FakeOutcomeTracker:
    def __init__(self):
        self.tracked = []

    def track_decision(self, decision, quantity):
        self.tracked.append((decision.symbol, quantity))


class FakeTelemetryPipeline(FakePipeline):
    def __init__(self):
        self.preference_generator = FakePreferenceGenerator()
        self.outcome_tracker = FakeOutcomeTracker()


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



def test_scheduler_executes_paper_signal_flow_with_telemetry(tmp_path):
    broker = PaperBroker(initial_cash=100_000)
    broker.connect()
    broker.set_price("AAPL", 100.0)

    risk_engine = RiskEngine(config={"allowed_underlyings": ["AAPL", "MSFT"]})
    oms = OrderManagementSystem(broker=broker, risk_engine=None)
    pipeline = FakeTelemetryPipeline()
    scheduler = TradingScheduler(
        config={
            "symbols": ["AAPL", "MSFT"],
            "dry_run": False,
            "max_positions": 5,
            "telemetry_dir": str(tmp_path),
        },
        pipeline=pipeline,
        oms=oms,
        risk_engine=risk_engine,
        alert_manager=None,
    )

    signals = scheduler.pre_market_analysis(["AAPL", "MSFT"])
    assert len(signals) == 1
    assert signals[0]["symbol"] == "AAPL"
    assert signals[0]["action"] == "buy"

    results = scheduler.execute_signals(signals)
    assert len(results) == 1
    assert results[0]["status"] == "filled"
    assert broker.get_positions()[0]["symbol"] == "AAPL"
    assert len(pipeline.preference_generator.decisions) == 1
    assert pipeline.outcome_tracker.tracked == [("AAPL", 1.0)]
    telemetry_file = Path(tmp_path) / "paper_signal_executed.jsonl"
    assert telemetry_file.exists()
    assert "AAPL" in telemetry_file.read_text()



def test_pipeline_backtest_uses_benchmark_runner(monkeypatch):
    def fake_init_llm_client(self):
        self.llm_client = None

    def fake_init_data(self):
        self.data_client = FakeDataClient()
        self.preprocessor = FakePreprocessor()

    def fake_init_features(self):
        self.tech_features = FakeTechnicalEngine()
        self.sent_features = None
        self.fund_features = None

    def fake_init_foundation_model(self):
        self.foundation_model = None

    def fake_init_agents(self):
        self.fundamental_analyst = None
        self.sentiment_analyst = None
        self.technical_analyst = None
        self.risk_analyst = None
        self.manager = None
        self.rag = None

    def fake_init_options(self):
        self.options_analyst = None
        self.options_chains = None
        self.greeks_calc = None
        self.vol_surface = None
        self.options_strategies = None
        self.options_flow = None

    def fake_init_execution(self):
        self.broker = None
        self.oms = None
        self.risk_engine = None
        self.alert_manager = None

    def fake_init_rlaif(self):
        self.preference_gen = None
        self.preference_generator = None
        self.reward_model = None
        self.rlaif_finetuner = None
        self.outcome_tracker = None
        self.options_outcome_tracker = None

    def fake_init_scheduler(self):
        self.scheduler = None

    monkeypatch.setattr(TradingPipeline, "_init_llm_client", fake_init_llm_client)
    monkeypatch.setattr(TradingPipeline, "_init_data", fake_init_data)
    monkeypatch.setattr(TradingPipeline, "_init_features", fake_init_features)
    monkeypatch.setattr(TradingPipeline, "_init_foundation_model", fake_init_foundation_model)
    monkeypatch.setattr(TradingPipeline, "_init_agents", fake_init_agents)
    monkeypatch.setattr(TradingPipeline, "_init_options", fake_init_options)
    monkeypatch.setattr(TradingPipeline, "_init_execution", fake_init_execution)
    monkeypatch.setattr(TradingPipeline, "_init_rlaif", fake_init_rlaif)
    monkeypatch.setattr(TradingPipeline, "_init_scheduler", fake_init_scheduler)

    pipeline = TradingPipeline(config_path="configs/config.yaml", mode="paper")
    result = pipeline.backtest(
        start_date="2024-01-01",
        end_date="2024-03-31",
        symbols=["AAPL", "MSFT"],
    )

    assert result["metrics"]["symbols_processed"] == 2
    assert result["metrics"]["signals_generated"] > 0
    assert "strategy_cumulative_return" in result["metrics"]
    assert "buy_and_hold_cumulative_return" in result["metrics"]
    assert len(result["trades"]) == 2
