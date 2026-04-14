from types import SimpleNamespace

from src.autotrader.orchestrator import ExperimentOrchestrator
from src.execution.risk_engine import RiskEngine


class DummySentinel:
    def __init__(self):
        self.updated = []
        self.check_calls = 0
        self.check_interval = 10

    def update_timing(self, check_interval=None, scheduled_interval=None):
        self.updated.append((check_interval, scheduled_interval))

    def check(self, portfolio_state, market_data):
        self.check_calls += 1
        return []


class DummyMetric:
    def __init__(self):
        self.updated = None

    def update_weights(self, **kwargs):
        self.updated = kwargs


class DummyRunner:
    def __init__(self):
        self.metric = DummyMetric()
        self.default_symbols = ["SPY"]


class DummySafety:
    is_halted = False

    def check_experiment_rate(self):
        return True, "ok"


class DummySwapper:
    active_spec = None


class DummyLog:
    def recent(self, limit):
        return []


class DummyStrategist:
    def __init__(self):
        self._current_directive = SimpleNamespace(risk_preference="moderate")
        self.assess_calls = 0

    def needs_reassessment(self):
        return True

    def assess(self, **kwargs):
        self.assess_calls += 1
        return SimpleNamespace(
            check_interval_seconds=30,
            composite_weights={"sharpe": 0.4, "return": 0.2, "drawdown": 0.3, "hit_rate": 0.1},
            improvement_threshold=0.01,
            experiment_frequency="hourly",
            thesis_guidance="",
            risk_budget_pct=0.2,
            max_position_pct=0.01,
            symbols_focus=["AAPL"],
            strategy_style="swing",
        )


def make_orchestrator(**kwargs):
    return ExperimentOrchestrator(
        sentinel=kwargs.get("sentinel", DummySentinel()),
        thesis_gen=SimpleNamespace(),
        runner=kwargs.get("runner", DummyRunner()),
        swapper=DummySwapper(),
        safety=DummySafety(),
        log=DummyLog(),
        mode=kwargs.get("mode", "continuous"),
        strategist=kwargs.get("strategist"),
        risk_engine=kwargs.get("risk_engine"),
        auto_reassess=kwargs.get("auto_reassess", True),
        reassess_interval_minutes=kwargs.get("reassess_interval_minutes"),
    )


def test_risk_engine_dynamic_limits_do_not_relax_base_config():
    engine = RiskEngine(
        config={
            "max_total_exposure_pct": 0.25,
            "max_position_risk_pct": 0.01,
            "max_trades_per_day": 2,
        }
    )

    applied = engine.set_dynamic_limits(
        max_exposure_pct=0.50,
        max_position_pct=0.02,
        max_trades_per_day=8,
    )

    assert applied["max_total_exposure_pct"] == 0.25
    assert applied["max_position_risk_pct"] == 0.01
    assert applied["max_trades_per_day"] == 2
    assert engine.config["max_total_exposure_pct"] == 0.25
    assert engine.config["max_position_risk_pct"] == 0.01
    assert engine.config["max_trades_per_day"] == 2


def test_orchestrator_preserves_daily_directives():
    sentinel = DummySentinel()
    runner = DummyRunner()
    risk_engine = SimpleNamespace(calls=[])

    def set_dynamic_limits(**kwargs):
        risk_engine.calls.append(kwargs)

    risk_engine.set_dynamic_limits = set_dynamic_limits

    orchestrator = make_orchestrator(
        sentinel=sentinel,
        runner=runner,
        risk_engine=risk_engine,
    )
    directive = SimpleNamespace(
        check_interval_seconds=120,
        composite_weights={"sharpe": 0.4, "return": 0.2, "drawdown": 0.3, "hit_rate": 0.1},
        improvement_threshold=0.02,
        experiment_frequency="daily",
        thesis_guidance="be patient",
        risk_budget_pct=0.2,
        max_position_pct=0.01,
        symbols_focus=["AAPL", "MSFT"],
        strategy_style="swing",
    )

    orchestrator._apply_directive(directive)

    assert orchestrator.mode == "daily"
    assert runner.default_symbols == ["AAPL", "MSFT"]
    assert sentinel.updated[-1] == (120, 240)
    assert risk_engine.calls[-1] == {
        "max_exposure_pct": 0.2,
        "max_position_pct": 0.01,
    }


def test_orchestrator_can_disable_auto_reassessment(monkeypatch):
    strategist = DummyStrategist()
    sentinel = DummySentinel()
    orchestrator = make_orchestrator(
        strategist=strategist,
        sentinel=sentinel,
        mode="daily",
        auto_reassess=False,
    )
    orchestrator._last_experiment_time = 1_000.0

    monkeypatch.setattr("time.time", lambda: 1_100.0)
    monkeypatch.setattr("time.sleep", lambda _: None)

    orchestrator._iteration(portfolio_state_fn=None, market_data_fn=None)

    assert strategist.assess_calls == 0
    assert sentinel.check_calls == 0
