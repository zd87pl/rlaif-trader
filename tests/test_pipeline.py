import pandas as pd

from main import TradingPipeline


class FakeDataClient:
    def download_latest(self, symbols, days, timeframe, use_cache=True):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=idx,
        )


class FakePreprocessor:
    def preprocess(self, df, symbol=None):
        return df.assign(returns=df["close"].pct_change().fillna(0.0))


class FakeTechFeatures:
    def compute_all(self, df):
        return df.assign(rsi=55.0, macd=1.2)


class FakeFoundationModel:
    def predict(self, df):
        return {"direction": "up", "confidence": 0.61}


class FakeAgent:
    def __init__(self, name):
        self.name = name

    def analyze(self, context):
        return {
            "agent": self.name,
            "signal": "buy",
            "confidence": 0.7,
            "symbol": context["symbol"],
        }


class FakeManager:
    def synthesize(self, agent_results, context):
        return {
            "decision": "buy",
            "confidence": 0.82,
            "reasoning": f"Consensus long on {context['symbol']}",
            "trades": [f"BUY {context['symbol']}"],
        }


class FakeRiskEngine:
    def status(self):
        return "ok"


class FakePaperBroker:
    def is_connected(self):
        return True


class FakeScheduler:
    def start(self, pipeline, symbols, mode):
        return None

    def stop(self):
        return None


class FakeAlertManager:
    def send(self, level, message):
        return None



def patch_pipeline_inits(monkeypatch):
    def fake_init_llm_client(self):
        self.llm_client = object()

    def fake_init_data(self):
        self.data_client = FakeDataClient()
        self.preprocessor = FakePreprocessor()

    def fake_init_features(self):
        self.tech_features = FakeTechFeatures()
        self.sent_features = None
        self.fund_features = None

    def fake_init_foundation_model(self):
        self.foundation_model = FakeFoundationModel()

    def fake_init_agents(self):
        self.fundamental_analyst = FakeAgent("fundamental")
        self.sentiment_analyst = FakeAgent("sentiment")
        self.technical_analyst = FakeAgent("technical")
        self.risk_analyst = FakeAgent("risk")
        self.manager = FakeManager()
        self.rag = None

    def fake_init_options(self):
        self.options_analyst = FakeAgent("options")
        self.options_chains = None
        self.greeks_calc = None
        self.vol_surface = None
        self.options_strategies = None
        self.options_flow = None

    def fake_init_execution(self):
        self.broker = FakePaperBroker()
        self.oms = None
        self.risk_engine = FakeRiskEngine()
        self.alert_manager = FakeAlertManager()

    def fake_init_rlaif(self):
        self.preference_gen = None
        self.reward_model = None
        self.rlaif_finetuner = None
        self.outcome_tracker = None
        self.options_outcome_tracker = None

    def fake_init_scheduler(self):
        self.scheduler = FakeScheduler()

    monkeypatch.setattr(TradingPipeline, "_init_llm_client", fake_init_llm_client)
    monkeypatch.setattr(TradingPipeline, "_init_data", fake_init_data)
    monkeypatch.setattr(TradingPipeline, "_init_features", fake_init_features)
    monkeypatch.setattr(TradingPipeline, "_init_foundation_model", fake_init_foundation_model)
    monkeypatch.setattr(TradingPipeline, "_init_agents", fake_init_agents)
    monkeypatch.setattr(TradingPipeline, "_init_options", fake_init_options)
    monkeypatch.setattr(TradingPipeline, "_init_execution", fake_init_execution)
    monkeypatch.setattr(TradingPipeline, "_init_rlaif", fake_init_rlaif)
    monkeypatch.setattr(TradingPipeline, "_init_scheduler", fake_init_scheduler)



def test_pipeline_status_smoke(monkeypatch):
    patch_pipeline_inits(monkeypatch)
    pipeline = TradingPipeline(config_path="configs/config.yaml", mode="paper")

    status = pipeline.status()

    assert status["mode"] == "paper"
    assert status["broker"] == "FakePaperBroker"
    assert status["broker_connected"] is True
    assert status["risk_state"] == "ok"
    assert "ManagerAgent" in status["agents"]
    assert status["components"]["data_client"] is True



def test_pipeline_analyze_smoke(monkeypatch):
    patch_pipeline_inits(monkeypatch)
    pipeline = TradingPipeline(config_path="configs/config.yaml", mode="paper")

    result = pipeline.analyze("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["foundation_forecast"]["direction"] == "up"
    assert result["features"]["technical"]["status"] == "ok"
    assert result["agents"]["technical"]["signal"] == "buy"
    assert result["manager_decision"]["decision"] == "buy"
    assert result["recommended_trades"] == ["BUY AAPL"]
