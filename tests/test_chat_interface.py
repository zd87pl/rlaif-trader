import json
from pathlib import Path

from src.chat_interface import ChatRouter


class FakeScheduler:
    def pre_market_analysis(self, symbols):
        return [
            {"symbol": symbols[0], "action": "buy", "confidence": 0.8, "position_size": 1}
        ]

    def execute_signals(self, signals):
        return [
            {
                "symbol": signals[0]["symbol"],
                "status": "simulated",
                "order_id": "paper-1",
            }
        ]


class FakePipeline:
    def __init__(self):
        self.scheduler = FakeScheduler()

    def status(self):
        return {
            "mode": "paper",
            "broker": "PaperBroker",
            "foundation_model": "unavailable",
            "broker_connected": True,
            "asset_universe": ["AAPL", "MSFT", "NVDA"],
        }

    def analyze(self, symbol):
        return {
            "symbol": symbol,
            "manager_decision": {
                "decision": "buy",
                "confidence": 0.77,
                "reasoning": f"Momentum is favorable for {symbol}",
            },
            "recommended_trades": [f"BUY {symbol}"],
            "features": {
                "technical": {
                    "latest": {
                        "rsi": 61.2,
                        "macd_hist": 1.4,
                    }
                }
            },
        }

    def analyze_options(self, symbol):
        return {
            "symbol": symbol,
            "recommended_strategies": [{"name": "call_spread"}],
            "unusual_flow": [{"strike": 100}],
        }

    def backtest(self, start_date, end_date, symbols=None):
        return {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols or ["AAPL"],
            "metrics": {
                "symbols_processed": len(symbols or ["AAPL"]),
                "strategy_cumulative_return": 0.12,
                "buy_and_hold_cumulative_return": 0.08,
                "alpha_vs_buy_hold": 0.04,
            },
        }


class FakeOpportunityPipeline(FakePipeline):
    def analyze(self, symbol):
        decision = "buy" if symbol in {"AAPL", "NVDA"} else "hold"
        confidence = 0.81 if symbol == "NVDA" else 0.62 if symbol == "AAPL" else 0.1
        return {
            "symbol": symbol,
            "manager_decision": {
                "decision": decision,
                "confidence": confidence,
                "reasoning": f"Reason for {symbol}",
            },
            "recommended_trades": [f"BUY {symbol}"] if decision == "buy" else [],
            "features": {
                "technical": {
                    "latest": {
                        "rsi": 58.0,
                        "macd_hist": 0.9,
                    }
                }
            },
        }


def make_router(tmp_path, pipeline_cls=FakePipeline):
    return ChatRouter(
        pipeline_factory=lambda config: pipeline_cls(),
        config_path="configs/config.yaml",
        notes_path=str(tmp_path / "alpha_notes.jsonl"),
        telemetry_dir=str(tmp_path / "telemetry"),
        state_path=str(tmp_path / "session_state.json"),
    )


def test_parse_intents(tmp_path):
    router = make_router(tmp_path)

    assert router.parse_intent("status").name == "status"
    assert router.parse_intent("analyze AAPL").params["symbol"] == "AAPL"
    assert router.parse_intent("options for nvda").name == "options"
    assert router.parse_intent("backtest AAPL,MSFT last year").name == "backtest"
    assert router.parse_intent("record this alpha: buy quality after panic").name == "alpha_note"
    assert router.parse_intent("show my alpha notes").name == "show_alpha_notes"
    assert router.parse_intent("show today's telemetry").name == "show_telemetry"
    assert router.parse_intent("show recent paper trades").name == "show_paper_trades"
    assert router.parse_intent("show latest eod summary").name == "show_eod_summary"
    assert router.parse_intent("start paper trading").name == "start_paper_trading"
    assert router.parse_intent("stop paper trading").name == "stop_paper_trading"
    assert router.parse_intent("start live trading").name == "live_trading_guardrail"


def test_handle_status_and_analysis(tmp_path):
    router = make_router(tmp_path)

    status = router.handle("status")
    analysis = router.handle("analyze AAPL")

    assert "Mode: paper" in status
    assert "Analysis for AAPL" in analysis
    assert "BUY AAPL" in analysis


def test_contextual_follow_up_for_options(tmp_path):
    router = make_router(tmp_path)
    router.handle("analyze AAPL")
    response = router.handle("what about options?")

    assert "Options analysis for AAPL" in response


def test_persistent_context_across_router_instances(tmp_path):
    router1 = make_router(tmp_path)
    router1.handle("analyze AAPL")

    router2 = make_router(tmp_path)
    response = router2.handle("what about options?")

    assert "Options analysis for AAPL" in response


def test_handle_opportunities_with_explanation(tmp_path):
    router = make_router(tmp_path, pipeline_cls=FakeOpportunityPipeline)
    response = router.handle("what are the best opportunities today?")

    assert "Top opportunities" in response
    assert "NVDA" in response
    assert "RSI" in response or "Reason for NVDA" in response


def test_save_and_show_alpha_note(tmp_path):
    router = make_router(tmp_path)
    response = router.handle("record this alpha: semis lead before AI infra earnings")
    notes_response = router.handle("show my alpha notes")

    notes_path = Path(tmp_path / "alpha_notes.jsonl")
    assert notes_path.exists()
    saved = [json.loads(line) for line in notes_path.read_text().splitlines()]
    assert saved[0]["note"] == "semis lead before AI infra earnings"
    assert "Saved alpha note" in response
    assert "Recent alpha notes" in notes_response


def test_show_today_telemetry(tmp_path):
    router = make_router(tmp_path)
    telemetry_dir = Path(tmp_path / "telemetry")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    event_file = telemetry_dir / "paper_signal_executed.jsonl"
    event_file.write_text(json.dumps({"timestamp": "2099-01-01T00:00:00", "event": "x"}) + "\n")

    response = router.handle("show today's telemetry")
    assert "No telemetry events recorded" in response


def test_show_recent_events(tmp_path):
    router = make_router(tmp_path)
    telemetry_dir = Path(tmp_path / "telemetry")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    trades_file = telemetry_dir / "paper_signal_executed.jsonl"
    eod_file = telemetry_dir / "paper_eod_summary.jsonl"
    trades_file.write_text(json.dumps({"timestamp": "2026-03-31T00:00:00", "signal": {"symbol": "AAPL", "action": "buy"}, "order_result": {"status": "simulated"}}) + "\n")
    eod_file.write_text(json.dumps({"timestamp": "2026-03-31T00:00:00", "summary": {"trades_executed": 2, "daily_pnl": 123.45}}) + "\n")

    trades_response = router.handle("show recent paper trades")
    eod_response = router.handle("show latest eod summary")

    assert "Recent paper trades" in trades_response
    assert "AAPL" in trades_response
    assert "Latest EOD summaries" in eod_response
    assert "123.45" in eod_response


def test_start_and_stop_paper_trading(tmp_path):
    router = make_router(tmp_path)
    start_response = router.handle("start paper trading")
    state_file = Path(tmp_path / "session_state.json")
    assert state_file.exists()

    stop_response = router.handle("stop paper trading")

    assert "Paper trading executed" in start_response
    assert router.state.paper_trading_active is False
    assert "stopped" in stop_response.lower()


def test_live_trading_guardrail(tmp_path):
    router = make_router(tmp_path)
    response = router.handle("start live trading")
    assert "blocked" in response.lower()
    assert "live" in response.lower()
