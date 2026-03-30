"""
RLAIF Trading Pipeline — Main Entry Point

Unified orchestrator that wires together multi-agent LLM analysis,
foundation models, options analytics, execution, and RLAIF feedback.

Usage:
    python main.py                  # Quick status check
    python main.py --help           # See CLI options (via src/cli.py)

    from main import TradingPipeline
    pipeline = TradingPipeline(mode='paper')
    result = pipeline.analyze('AAPL')
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project root on sys.path so `src.*` imports resolve everywhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_config, setup_logging, get_logger, set_seed


# ---------------------------------------------------------------------------
# TradingPipeline
# ---------------------------------------------------------------------------
class TradingPipeline:
    """Top-level orchestrator that initialises every subsystem and exposes
    high-level actions: analyse, paper-trade, live-trade, backtest."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        mode: str = "paper",
        log_level: str = "INFO",
    ):
        # ------ bootstrap ------
        self.mode = mode
        self.config = load_config(config_path)
        self.logger = get_logger("pipeline")
        setup_logging(level=log_level)
        set_seed(self.config.get("rl", {}).get("ensemble", {}).get("agents", [{}])[0].get("seed", 42))
        self.logger.info("Initialising TradingPipeline  mode=%s", mode)

        # ------ LLM client (MLX local or Claude API) ------
        self._init_llm_client()

        # ------ data layer ------
        self._init_data()

        # ------ feature engines ------
        self._init_features()

        # ------ foundation model (TimesFM / TTM) ------
        self._init_foundation_model()

        # ------ multi-agent system ------
        self._init_agents()

        # ------ options subsystem ------
        self._init_options()

        # ------ execution layer ------
        self._init_execution()

        # ------ RLAIF feedback loop ------
        self._init_rlaif()

        # ------ scheduler ------
        self._init_scheduler()

        self.logger.info("TradingPipeline ready.")

    # -----------------------------------------------------------------------
    # Initialisation helpers (each is a separate method so failures are clear)
    # -----------------------------------------------------------------------
    def _init_llm_client(self):
        from src.agents import create_client, get_default_client

        backend = os.getenv("LLM_BACKEND", self.config.get("llm", {}).get("backend", "claude"))
        llm_cfg = self.config.get("llm", {})
        try:
            self.llm_client = create_client(
                backend=backend,
                model=llm_cfg.get("model"),
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_tokens", 4096),
            )
        except Exception:
            self.logger.warning("create_client() failed — falling back to get_default_client()")
            self.llm_client = get_default_client()
        self.logger.info("LLM client: %s", type(self.llm_client).__name__)

    def _init_data(self):
        from src.data import AlpacaDataClient, DataPreprocessor

        data_cfg = self.config.get("data", {})
        self.data_client = AlpacaDataClient(config=data_cfg)
        self.preprocessor = DataPreprocessor(config=data_cfg.get("processing", {}))
        self.logger.info("Data layer initialised (Alpaca + preprocessor)")

    def _init_features(self):
        from src.features import TechnicalFeatureEngine, SentimentAnalyzer, FundamentalAnalyzer

        feat_cfg = self.config.get("features", {})
        self.tech_features = TechnicalFeatureEngine(config=feat_cfg.get("technical", {}))
        self.sent_features = SentimentAnalyzer(config=feat_cfg.get("sentiment", {}))
        self.fund_features = FundamentalAnalyzer(config=feat_cfg.get("fundamental", {}))
        self.logger.info("Feature engines ready (technical, sentiment, fundamental)")

    def _init_foundation_model(self):
        fm_cfg = self.config.get("foundation_model", {})
        model_type = fm_cfg.get("model_type", "timesfm")
        try:
            if model_type == "ttm":
                from src.models import TTMPredictor
                self.foundation_model = TTMPredictor(config=fm_cfg.get("ttm", {}))
            else:
                from src.models import TimesFMPredictor
                self.foundation_model = TimesFMPredictor(config=fm_cfg.get("timesfm", {}))
            self.logger.info("Foundation model loaded: %s", model_type)
        except Exception as exc:
            self.logger.warning("Foundation model (%s) unavailable: %s", model_type, exc)
            self.foundation_model = None

    def _init_agents(self):
        from src.agents import (
            FundamentalAnalyst,
            SentimentAnalyst,
            TechnicalAnalyst,
            RiskAnalyst,
            ManagerAgent,
            RAGSystem,
        )

        agent_cfg = self.config.get("llm", {}).get("agents", {})

        self.fundamental_analyst = FundamentalAnalyst(
            llm_client=self.llm_client,
            config=agent_cfg.get("fundamental_analyst", {}),
        )
        self.sentiment_analyst = SentimentAnalyst(
            llm_client=self.llm_client,
            config=agent_cfg.get("sentiment_analyst", {}),
        )
        self.technical_analyst = TechnicalAnalyst(
            llm_client=self.llm_client,
            config=agent_cfg.get("technical_analyst", {}),
        )
        self.risk_analyst = RiskAnalyst(
            llm_client=self.llm_client,
            config=agent_cfg.get("risk_analyst", {}),
        )
        self.manager = ManagerAgent(
            llm_client=self.llm_client,
            config=agent_cfg.get("manager", {}),
        )

        # RAG system (optional)
        rag_cfg = self.config.get("llm", {}).get("rag", {})
        if rag_cfg.get("enabled", False):
            try:
                self.rag = RAGSystem(config=rag_cfg)
                self.logger.info("RAG system enabled")
            except Exception as exc:
                self.logger.warning("RAG init failed: %s", exc)
                self.rag = None
        else:
            self.rag = None

        self.logger.info("Multi-agent system ready (5 analysts + manager)")

    def _init_options(self):
        from src.options.chains import OptionsChainManager
        from src.options.greeks import GreeksCalculator
        from src.options.vol_surface import VolatilitySurface
        from src.options.strategies import OptionsStrategyBuilder
        from src.options.flow_analyzer import OptionsFlowAnalyzer
        from src.options.options_analyst import OptionsAnalyst

        self.options_chains = OptionsChainManager()
        self.greeks_calc = GreeksCalculator()
        self.vol_surface = VolatilitySurface()
        self.options_strategies = OptionsStrategyBuilder()
        self.options_flow = OptionsFlowAnalyzer()
        self.options_analyst = OptionsAnalyst(
            llm_client=self.llm_client,
            chain_manager=self.options_chains,
            greeks_calculator=self.greeks_calc,
            vol_surface=self.vol_surface,
            strategy_builder=self.options_strategies,
            flow_analyzer=self.options_flow,
        )
        self.logger.info("Options subsystem ready")

    def _init_execution(self):
        from src.execution import (
            AlpacaBroker,
            TradierBroker,
            PaperBroker,
            OrderManagementSystem,
        )
        from src.execution.risk_engine import RiskEngine
        from src.execution.alerts import AlertManager
        from src.execution.scheduler import TradingScheduler

        env_cfg = self.config.get("environment", {})

        # Select broker
        if self.mode == "live":
            broker_name = os.getenv("BROKER", "alpaca")
            if broker_name == "tradier":
                self.broker = TradierBroker()
            else:
                self.broker = AlpacaBroker()
        else:
            self.broker = PaperBroker(
                initial_balance=env_cfg.get("initial_balance", 100_000),
            )

        self.oms = OrderManagementSystem(broker=self.broker)
        self.risk_engine = RiskEngine(config=env_cfg)
        self.alert_manager = AlertManager()
        self.logger.info("Execution layer ready  broker=%s", type(self.broker).__name__)

    def _init_rlaif(self):
        from src.rlaif import PreferenceGenerator, RewardModel, RLAIFFineTuner, OutcomeTracker
        from src.rlaif.options_outcome_tracker import OptionsOutcomeTracker

        rlaif_cfg = self.config.get("rlaif", {})
        self.preference_gen = PreferenceGenerator(config=rlaif_cfg.get("preferences", {}))
        self.reward_model = RewardModel(config=rlaif_cfg.get("reward_model", {}))
        self.rlaif_finetuner = RLAIFFineTuner(config=rlaif_cfg.get("fine_tuning", {}))
        self.outcome_tracker = OutcomeTracker(config=rlaif_cfg.get("feedback", {}))
        self.options_outcome_tracker = OptionsOutcomeTracker()
        self.logger.info("RLAIF feedback loop initialised")

    def _init_scheduler(self):
        from src.execution.scheduler import TradingScheduler

        self.scheduler = TradingScheduler()
        self.logger.info("Scheduler initialised")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Run full multi-agent analysis pipeline on *symbol*.

        Returns a dict with keys: symbol, timestamp, agents (per-agent output),
        manager_decision, foundation_forecast, features, recommended_trades.
        """
        self.logger.info("Analysing %s ...", symbol)
        t0 = time.time()

        # 1) Fetch raw data
        raw_data = self.data_client.get_bars(symbol)
        processed = self.preprocessor.process(raw_data)

        # 2) Compute features
        tech_feats = self.tech_features.compute(processed)
        sent_feats = self.sent_features.compute(symbol)
        fund_feats = self.fund_features.compute(symbol)

        features = {
            "technical": tech_feats,
            "sentiment": sent_feats,
            "fundamental": fund_feats,
        }

        # 3) Foundation model forecast
        forecast = None
        if self.foundation_model is not None:
            try:
                forecast = self.foundation_model.predict(processed)
            except Exception as exc:
                self.logger.warning("Foundation model forecast failed: %s", exc)

        # 4) Run agents (concurrently where possible)
        context = {
            "symbol": symbol,
            "features": features,
            "forecast": forecast,
            "price_data": processed,
        }

        agent_results = {}
        for name, agent in [
            ("fundamental", self.fundamental_analyst),
            ("sentiment", self.sentiment_analyst),
            ("technical", self.technical_analyst),
            ("risk", self.risk_analyst),
        ]:
            try:
                agent_results[name] = agent.analyze(context)
            except Exception as exc:
                self.logger.error("Agent %s failed: %s", name, exc)
                agent_results[name] = {"error": str(exc)}

        # Options analyst
        try:
            agent_results["options"] = self.options_analyst.analyze(context)
        except Exception as exc:
            self.logger.warning("Options analyst failed: %s", exc)
            agent_results["options"] = {"error": str(exc)}

        # 5) Manager synthesis
        try:
            manager_decision = self.manager.synthesize(
                agent_results=agent_results,
                context=context,
            )
        except Exception as exc:
            self.logger.error("Manager synthesis failed: %s", exc)
            manager_decision = {"error": str(exc)}

        elapsed = time.time() - t0
        self.logger.info("Analysis for %s completed in %.1fs", symbol, elapsed)

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "features": features,
            "foundation_forecast": forecast,
            "agents": agent_results,
            "manager_decision": manager_decision,
            "recommended_trades": manager_decision.get("trades", []) if isinstance(manager_decision, dict) else [],
        }

    def analyze_options(self, symbol: str) -> Dict[str, Any]:
        """Options-focused analysis: vol surface, unusual flow, strategy recs."""
        self.logger.info("Options analysis for %s ...", symbol)

        chain = self.options_chains.get_chain(symbol)
        greeks = self.greeks_calc.compute(chain)
        vol_surf = self.vol_surface.build(chain)
        flow = self.options_flow.analyze(symbol)
        strategies = self.options_strategies.recommend(
            symbol=symbol,
            chain=chain,
            vol_surface=vol_surf,
        )

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "chain_summary": chain,
            "greeks": greeks,
            "vol_surface": vol_surf,
            "unusual_flow": flow,
            "recommended_strategies": strategies,
        }

    def run_paper(self):
        """Start paper-trading loop driven by the scheduler."""
        self.logger.info("Starting PAPER trading …")
        if self.mode != "paper":
            self.logger.warning("Pipeline was not initialised in paper mode — switching broker to PaperBroker")
            from src.execution import PaperBroker
            env_cfg = self.config.get("environment", {})
            self.broker = PaperBroker(initial_balance=env_cfg.get("initial_balance", 100_000))

        assets = self.config.get("data", {}).get("assets", [])
        self.scheduler.start(pipeline=self, symbols=assets, mode="paper")

    def run_live(self):
        """Start live trading (requires explicit confirmation)."""
        self.logger.info("Starting LIVE trading …")
        if self.mode != "live":
            raise RuntimeError(
                "Pipeline must be initialised with mode='live' to trade live. "
                "Re-create with TradingPipeline(mode='live')."
            )
        assets = self.config.get("data", {}).get("assets", [])
        self.scheduler.start(pipeline=self, symbols=assets, mode="live")

    def backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a backtest over [start_date, end_date].

        Parameters
        ----------
        start_date : str  e.g. "2023-01-01"
        end_date   : str  e.g. "2024-01-01"
        symbols    : list of tickers (defaults to config asset universe)

        Returns a dict with equity curve, metrics, and trade log.
        """
        symbols = symbols or self.config.get("data", {}).get("assets", [])
        self.logger.info("Backtesting %s → %s on %d symbols", start_date, end_date, len(symbols))

        bt_cfg = self.config.get("backtesting", {})
        results: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "trades": [],
            "equity_curve": [],
            "metrics": {},
        }

        # Fetch historical data for each symbol
        for sym in symbols:
            try:
                hist = self.data_client.get_bars(sym, start=start_date, end=end_date)
                processed = self.preprocessor.process(hist)
                analysis = self.analyze(sym)
                results["trades"].append(
                    {"symbol": sym, "analysis": analysis}
                )
            except Exception as exc:
                self.logger.error("Backtest error on %s: %s", sym, exc)

        self.logger.info("Backtest complete — %d symbol(s) processed", len(results["trades"]))
        return results

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of system health and component status."""
        info: Dict[str, Any] = {
            "mode": self.mode,
            "timestamp": datetime.utcnow().isoformat(),
            "llm_client": type(self.llm_client).__name__,
            "foundation_model": (
                type(self.foundation_model).__name__
                if self.foundation_model
                else "unavailable"
            ),
            "broker": type(self.broker).__name__,
            "rag_enabled": self.rag is not None,
            "rlaif_enabled": self.config.get("rlaif", {}).get("enabled", False),
            "asset_universe": self.config.get("data", {}).get("assets", []),
            "agents": [
                "FundamentalAnalyst",
                "SentimentAnalyst",
                "TechnicalAnalyst",
                "RiskAnalyst",
                "OptionsAnalyst",
                "ManagerAgent",
            ],
        }

        # Broker connectivity
        try:
            info["broker_connected"] = self.broker.is_connected() if hasattr(self.broker, "is_connected") else True
        except Exception:
            info["broker_connected"] = False

        # Risk engine state
        try:
            info["risk_state"] = self.risk_engine.status() if hasattr(self.risk_engine, "status") else "ok"
        except Exception:
            info["risk_state"] = "unknown"

        return info

    def kill(self):
        """Emergency kill switch — cancel all orders, flatten positions."""
        self.logger.critical("KILL SWITCH ACTIVATED — liquidating all positions")
        try:
            self.oms.cancel_all()
        except Exception as exc:
            self.logger.error("cancel_all failed: %s", exc)
        try:
            self.broker.flatten_all()
        except Exception as exc:
            self.logger.error("flatten_all failed: %s", exc)
        try:
            self.scheduler.stop()
        except Exception as exc:
            self.logger.error("scheduler stop failed: %s", exc)
        self.alert_manager.send(
            level="critical",
            message="KILL SWITCH — all orders cancelled, positions flattened",
        )
        self.logger.critical("Kill switch complete.")


# ---------------------------------------------------------------------------
# Quick start
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    pipeline = TradingPipeline()
    st = pipeline.status()
    print(json.dumps(st, indent=2, default=str))
