"""Plain-English chat interface for the trading CLI."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ChatIntent:
    name: str
    params: Dict[str, Any]
    raw_text: str


@dataclass
class ChatSessionState:
    last_symbol: Optional[str] = None
    last_symbols: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    paper_trading_active: bool = False
    last_paper_results: List[Dict[str, Any]] = field(default_factory=list)


class ChatRouter:
    def __init__(
        self,
        pipeline_factory: Callable[..., Any],
        config_path: str = "configs/config.yaml",
        notes_path: str = "./data/chat/alpha_notes.jsonl",
        telemetry_dir: str = "./data/telemetry",
        state_path: str = "./data/chat/session_state.json",
    ):
        self.pipeline_factory = pipeline_factory
        self.config_path = config_path
        self.notes_path = Path(notes_path)
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def parse_intent(self, text: str) -> ChatIntent:
        raw = text.strip()
        lower = raw.lower()

        if lower in {"exit", "quit", "bye", ":q"}:
            return ChatIntent("exit", {}, raw)

        if lower in {"help", "?"} or "what can you do" in lower:
            return ChatIntent("help", {}, raw)

        if any(phrase in lower for phrase in ["show my alpha notes", "show alpha notes", "list alpha notes", "what alpha notes do i have"]):
            return ChatIntent("show_alpha_notes", {}, raw)

        if any(phrase in lower for phrase in ["show recent paper trades", "show paper trades", "recent paper trades", "show recent fills", "show fills"]):
            return ChatIntent("show_paper_trades", {}, raw)

        if any(phrase in lower for phrase in ["show latest eod", "show eod summary", "latest eod summary", "show latest eod summary"]):
            return ChatIntent("show_eod_summary", {}, raw)

        if any(phrase in lower for phrase in ["today's telemetry", "todays telemetry", "show telemetry", "show today's telemetry", "what happened today"]):
            return ChatIntent("show_telemetry", {}, raw)

        if any(phrase in lower for phrase in ["start live trading", "go live", "place real orders", "trade live"]):
            return ChatIntent("live_trading_guardrail", {}, raw)

        if any(phrase in lower for phrase in ["start paper trading", "paper trade the top", "paper trade top", "paper trade opportunities"]):
            top_n = self._extract_top_n(lower) or 3
            return ChatIntent("start_paper_trading", {"top_n": top_n}, raw)

        if any(phrase in lower for phrase in ["stop paper trading", "stop the paper trading", "stop paper mode"]):
            return ChatIntent("stop_paper_trading", {}, raw)

        if "status" in lower or "system health" in lower:
            return ChatIntent("status", {}, raw)

        if lower.startswith("analyze ") or lower.startswith("analyse "):
            symbol = self._extract_symbol_with_context(raw)
            return ChatIntent("analyze", {"symbol": symbol}, raw)

        if ("options" in lower or "options angle" in lower) and self._extract_symbol_with_context(raw):
            return ChatIntent("options", {"symbol": self._extract_symbol_with_context(raw)}, raw)

        if any(phrase in lower for phrase in ["best opportunities", "current opportunities", "top opportunities", "opportunities today", "scan opportunities"]):
            return ChatIntent("opportunities", {}, raw)

        if any(phrase in lower for phrase in ["remember this", "record this alpha", "save this alpha", "alpha:"]):
            note = self._extract_alpha_note(raw)
            return ChatIntent("alpha_note", {"note": note}, raw)

        if "backtest" in lower or "benchmark" in lower:
            symbols = self._extract_symbols_with_context(raw)
            start, end = self._extract_date_range(lower)
            return ChatIntent(
                "backtest",
                {
                    "symbols": symbols,
                    "start": start,
                    "end": end,
                },
                raw,
            )

        symbol = self._extract_symbol_with_context(raw)
        if symbol:
            return ChatIntent("analyze", {"symbol": symbol}, raw)

        return ChatIntent("unknown", {}, raw)

    def handle(self, text: str) -> str:
        intent = self.parse_intent(text)

        if intent.name == "exit":
            return "Goodbye."
        if intent.name == "help":
            return self._help_text()
        if intent.name == "unknown":
            return (
                "I didn't understand that yet. Try things like: 'status', 'analyze AAPL', "
                "'what about options?', 'best opportunities today', 'backtest AAPL,MSFT last year', "
                "'start paper trading', 'show my alpha notes', 'show recent paper trades', or "
                "'show today's telemetry'."
            )

        if intent.name == "alpha_note":
            response = self._save_alpha_note(intent.params["note"])
            self._remember_intent(intent)
            return response
        if intent.name == "show_alpha_notes":
            self._remember_intent(intent)
            return self._show_alpha_notes()
        if intent.name == "show_telemetry":
            self._remember_intent(intent)
            return self._show_today_telemetry()
        if intent.name == "show_paper_trades":
            self._remember_intent(intent)
            return self._show_recent_events("paper_signal_executed", title="Recent paper trades")
        if intent.name == "show_eod_summary":
            self._remember_intent(intent)
            return self._show_recent_events("paper_eod_summary", title="Latest EOD summaries")
        if intent.name == "live_trading_guardrail":
            self._remember_intent(intent)
            return (
                "Live trading is blocked from plain-English chat for safety. "
                "Use the explicit 'live' CLI command with confirmations if you really want to do that."
            )
        if intent.name == "stop_paper_trading":
            self.state.paper_trading_active = False
            self.state.last_paper_results = []
            self._save_state()
            self._remember_intent(intent)
            return "Paper trading session marked as stopped in chat mode."

        pipe = None
        if intent.name in {
            "status",
            "analyze",
            "options",
            "opportunities",
            "backtest",
            "start_paper_trading",
        }:
            pipe = self.pipeline_factory(config=self.config_path)

        if intent.name == "status":
            response = self._render_status(pipe.status())
        elif intent.name == "analyze":
            result = pipe.analyze(intent.params["symbol"])
            response = self._render_analysis(result)
        elif intent.name == "options":
            result = pipe.analyze_options(intent.params["symbol"])
            response = self._render_options(result)
        elif intent.name == "opportunities":
            response = self._render_opportunities(pipe)
        elif intent.name == "backtest":
            result = pipe.backtest(
                start_date=intent.params["start"],
                end_date=intent.params["end"],
                symbols=intent.params["symbols"],
            )
            response = self._render_backtest(result)
        elif intent.name == "start_paper_trading":
            response = self._start_paper_trading(pipe, top_n=intent.params.get("top_n", 3))
        else:
            response = "Unhandled intent."

        self._remember_intent(intent)
        return response

    def _help_text(self) -> str:
        return (
            "You can ask for:\n"
            "- status\n"
            "- analyze AAPL\n"
            "- what about options?\n"
            "- best opportunities today\n"
            "- backtest AAPL,MSFT last year\n"
            "- start paper trading\n"
            "- stop paper trading\n"
            "- show recent paper trades\n"
            "- show latest eod summary\n"
            "- record this alpha: post-earnings IV crush on quality names\n"
            "- show my alpha notes\n"
            "- show today's telemetry\n"
            "- exit"
        )

    def _remember_intent(self, intent: ChatIntent) -> None:
        self.state.last_intent = intent.name
        if intent.params.get("symbol"):
            self.state.last_symbol = intent.params["symbol"]
        if intent.params.get("symbols"):
            self.state.last_symbols = list(intent.params["symbols"])
        elif intent.params.get("symbol"):
            self.state.last_symbols = [intent.params["symbol"]]
        self._save_state()

    def _render_status(self, status: Dict[str, Any]) -> str:
        return (
            f"Mode: {status.get('mode')}\n"
            f"Broker: {status.get('broker')}\n"
            f"Foundation model: {status.get('foundation_model')}\n"
            f"Broker connected: {status.get('broker_connected')}\n"
            f"Universe: {', '.join(status.get('asset_universe', []))}"
        )

    def _render_analysis(self, result: Dict[str, Any]) -> str:
        manager = result.get("manager_decision", {})
        action = manager.get("action") or manager.get("decision") or "unknown"
        confidence = manager.get("confidence", "?")
        trades = result.get("recommended_trades", [])
        trades_text = ", ".join(str(t) for t in trades) if trades else "none"
        return (
            f"Analysis for {result.get('symbol')}:\n"
            f"- action: {action}\n"
            f"- confidence: {confidence}\n"
            f"- recommended trades: {trades_text}"
        )

    def _render_options(self, result: Dict[str, Any]) -> str:
        if result.get("error"):
            return f"Options analysis unavailable: {result['error']}"
        strategies = result.get("recommended_strategies") or []
        flow = result.get("unusual_flow") or []
        return (
            f"Options analysis for {result.get('symbol')}:\n"
            f"- unusual flow items: {len(flow)}\n"
            f"- recommended strategies: {len(strategies)}"
        )

    def _render_opportunities(self, pipe: Any) -> str:
        status = pipe.status()
        universe = status.get("asset_universe", [])[:5]
        ranked: List[tuple] = []
        for symbol in universe:
            try:
                result = pipe.analyze(symbol)
                manager = result.get("manager_decision", {})
                action = str(manager.get("action") or manager.get("decision") or "hold").lower()
                confidence = float(manager.get("confidence", 0.0) or 0.0)
                if action not in {"hold", "unknown"}:
                    reason = self._summarize_opportunity_reason(result)
                    ranked.append((confidence, symbol, action, reason))
            except Exception:
                continue
        ranked.sort(reverse=True)
        if not ranked:
            return "No strong opportunities found right now."
        lines = ["Top opportunities:"]
        for confidence, symbol, action, reason in ranked[:3]:
            lines.append(f"- {symbol}: {action} (confidence {confidence:.2f}) — {reason}")
        return "\n".join(lines)

    def _render_backtest(self, result: Dict[str, Any]) -> str:
        metrics = result.get("metrics", {})
        return (
            f"Backtest {result.get('start_date')} -> {result.get('end_date')}\n"
            f"- symbols processed: {metrics.get('symbols_processed', '?')}\n"
            f"- strategy return: {metrics.get('strategy_cumulative_return', metrics.get('cumulative_return', '?'))}\n"
            f"- buy and hold return: {metrics.get('buy_and_hold_cumulative_return', 'n/a')}\n"
            f"- alpha vs buy and hold: {metrics.get('alpha_vs_buy_hold', 'n/a')}"
        )

    def _start_paper_trading(self, pipe: Any, top_n: int = 3) -> str:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            return "Paper trading scheduler is unavailable in this environment."
        symbols = pipe.status().get("asset_universe", [])[: max(top_n, 1)]
        signals = scheduler.pre_market_analysis(symbols)
        if not signals:
            self.state.paper_trading_active = False
            self.state.last_paper_results = []
            self._save_state()
            return "No actionable paper-trading signals found right now."
        results = scheduler.execute_signals(signals[:top_n])
        self.state.paper_trading_active = True
        self.state.last_paper_results = results
        self._save_state()
        filled = [r for r in results if r.get("status") in {"filled", "submitted", "simulated"}]
        return (
            f"Paper trading executed {len(results)} signal(s). "
            f"Successful/simulated actions: {len(filled)}."
        )

    def _save_alpha_note(self, note: str) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "note": note,
        }
        with open(self.notes_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return f"Saved alpha note: {note}"

    def _show_alpha_notes(self, limit: int = 5) -> str:
        if not self.notes_path.exists():
            return "No alpha notes saved yet."
        lines = self.notes_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return "No alpha notes saved yet."
        records = [json.loads(line) for line in lines[-limit:]]
        output = ["Recent alpha notes:"]
        for record in records:
            output.append(f"- {record.get('timestamp')}: {record.get('note')}")
        return "\n".join(output)

    def _show_today_telemetry(self) -> str:
        today = datetime.utcnow().date().isoformat()
        counts: Dict[str, int] = {}
        for path in self.telemetry_dir.glob("*.jsonl"):
            count = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if today in line:
                    count += 1
            if count:
                counts[path.stem] = count
        if not counts:
            return "No telemetry events recorded for today yet."
        lines = ["Today's telemetry:"]
        for name, count in sorted(counts.items()):
            lines.append(f"- {name}: {count}")
        return "\n".join(lines)

    def _show_recent_events(self, event_type: str, title: str, limit: int = 5) -> str:
        path = self.telemetry_dir / f"{event_type}.jsonl"
        if not path.exists():
            return f"No {event_type.replace('_', ' ')} events recorded yet."
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return f"No {event_type.replace('_', ' ')} events recorded yet."
        records = [json.loads(line) for line in lines[-limit:]]
        output = [f"{title}:"]
        for record in records:
            if event_type == "paper_signal_executed":
                signal = record.get("signal", {})
                order = record.get("order_result", {})
                output.append(
                    f"- {record.get('timestamp')}: {signal.get('symbol')} {signal.get('action')} -> {order.get('status')}"
                )
            else:
                summary = record.get("summary", {})
                output.append(
                    f"- {record.get('timestamp')}: trades={summary.get('trades_executed', '?')} pnl={summary.get('daily_pnl', '?')}"
                )
        return "\n".join(output)

    def _summarize_opportunity_reason(self, result: Dict[str, Any]) -> str:
        technical = result.get("features", {}).get("technical", {})
        latest = technical.get("latest", {}) if isinstance(technical, dict) else {}
        hints = []
        if latest.get("rsi") is not None:
            hints.append(f"RSI {latest.get('rsi')}")
        if latest.get("macd_hist") is not None:
            hints.append(f"MACD hist {latest.get('macd_hist')}")
        manager = result.get("manager_decision", {})
        if manager.get("reasoning"):
            hints.append(str(manager.get("reasoning"))[:80])
        return "; ".join(hints) if hints else "ranked by manager confidence"

    def _extract_symbol_with_context(self, text: str) -> Optional[str]:
        symbol = self._extract_symbol(text)
        if symbol:
            return symbol
        lower = text.lower()
        if any(token in lower for token in ["it", "that", "same symbol", "what about options", "what about it"]):
            return self.state.last_symbol
        return None

    def _extract_symbols_with_context(self, text: str) -> Optional[List[str]]:
        symbols = self._extract_symbols(text)
        if symbols:
            return symbols
        lower = text.lower()
        if any(token in lower for token in ["that", "same symbols", "same universe", "that idea"]):
            return self.state.last_symbols or ([self.state.last_symbol] if self.state.last_symbol else None)
        return None

    @staticmethod
    def _extract_symbol(text: str) -> Optional[str]:
        stopwords = {
            "WHAT",
            "ABOUT",
            "OPTIONS",
            "OPTION",
            "ANALYZE",
            "ANALYSE",
            "STATUS",
            "BACKTEST",
            "BENCHMARK",
            "SHOW",
            "TODAY",
            "PAPER",
            "TRADING",
            "START",
            "STOP",
            "FOR",
            "THE",
            "TOP",
            "RECENT",
            "FILLS",
            "LATEST",
            "EOD",
            "SUMMARY",
            "LIVE",
            "GO",
            "REAL",
            "ORDERS",
        }
        matches = [m for m in re.findall(r"\b[A-Z]{1,5}\b", text.upper()) if m not in stopwords]
        return matches[0] if matches else None

    @staticmethod
    def _extract_symbols(text: str) -> Optional[List[str]]:
        stopwords = {
            "WHAT",
            "ABOUT",
            "OPTIONS",
            "OPTION",
            "ANALYZE",
            "ANALYSE",
            "STATUS",
            "BACKTEST",
            "BENCHMARK",
            "SHOW",
            "TODAY",
            "PAPER",
            "TRADING",
            "START",
            "STOP",
            "FOR",
            "THE",
            "TOP",
            "LAST",
            "YEAR",
            "RECENT",
            "FILLS",
            "LATEST",
            "EOD",
            "SUMMARY",
            "LIVE",
            "GO",
            "REAL",
            "ORDERS",
        }
        matches = [m for m in re.findall(r"\b[A-Z]{1,5}\b", text.upper()) if m not in stopwords]
        return matches or None

    @staticmethod
    def _extract_top_n(lower_text: str) -> Optional[int]:
        match = re.search(r"top\s+(\d+)", lower_text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_alpha_note(text: str) -> str:
        lowered = text.lower()
        for prefix in ["remember this:", "record this alpha:", "save this alpha:", "alpha:"]:
            if prefix in lowered:
                start = lowered.index(prefix) + len(prefix)
                return text[start:].strip()
        return text.strip()

    @staticmethod
    def _extract_date_range(lower_text: str) -> tuple[str, str]:
        today = datetime.utcnow().date()
        if "last year" in lower_text:
            end = today
            start = end - timedelta(days=365)
            return start.isoformat(), end.isoformat()
        return (today - timedelta(days=365)).isoformat(), today.isoformat()

    def _load_state(self) -> ChatSessionState:
        if not self.state_path.exists():
            return ChatSessionState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ChatSessionState(**data)
        except Exception:
            return ChatSessionState()

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")


def run_chat_session(router: ChatRouter, input_fn=input, output_fn=print) -> None:
    output_fn("RLAIF chat mode. Type 'help' for examples or 'exit' to quit.")
    while True:
        try:
            user_text = input_fn("rlaif> ")
        except EOFError:
            output_fn("Goodbye.")
            return
        response = router.handle(user_text)
        output_fn(response)
        if router.parse_intent(user_text).name == "exit":
            return
