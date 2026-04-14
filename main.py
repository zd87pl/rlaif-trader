"""
RLAIF Trading Pipeline — Main Entry Point

Unified orchestrator that wires together multi-agent LLM analysis,
foundation models, options analytics, execution, and RLAIF feedback.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in local envs
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()

# ---------------------------------------------------------------------------
# Project root on sys.path so `src.*` imports resolve everywhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, get_settings, load_config, set_seed, setup_logging


class TradingPipeline:
    """Top-level orchestrator for analysis, paper trading, and backtesting."""

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        mode: str = "paper",
        log_level: str = "INFO",
    ):
        self.mode = mode
        self.config = load_config(config_path)
        setup_logging(log_level=log_level)
        self.logger = get_logger("pipeline")

        seed = (
            self.config.get("rl", {})
            .get("ensemble", {})
            .get("agents", [{}])[0]
            .get("seed", 42)
        )
        set_seed(seed)

        self.llm_client = None
        self.data_client = None
        self.preprocessor = None
        self.tech_features = None
        self.sent_features = None
        self.fund_features = None
        self.foundation_model = None
        self.fundamental_analyst = None
        self.sentiment_analyst = None
        self.technical_analyst = None
        self.risk_analyst = None
        self.manager = None
        self.rag = None
        self.options_chains = None
        self.greeks_calc = None
        self.vol_surface = None
        self.options_strategies = None
        self.options_flow = None
        self.options_analyst = None
        self.broker = None
        self.oms = None
        self.risk_engine = None
        self.alert_manager = None
        self.preference_gen = None
        self.reward_model = None
        self.rlaif_finetuner = None
        self.outcome_tracker = None
        self.options_outcome_tracker = None
        self.scheduler = None
        self.autotrader = None

        self.logger.info("Initialising TradingPipeline mode=%s", mode)

        self._init_llm_client()
        self._init_data()
        self._init_features()
        self._init_foundation_model()
        self._init_agents()
        self._init_options()
        self._init_execution()
        self._init_rlaif()
        self._init_scheduler()
        self._init_autotrader()

        self.logger.info("TradingPipeline ready")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_llm_client(self) -> None:
        try:
            from src.agents import create_client, get_default_client
        except Exception as exc:
            self.logger.warning("LLM client factory unavailable: %s", exc)
            self.llm_client = None
            return

        backend = os.getenv(
            "LLM_BACKEND", self.config.get("llm", {}).get("backend", "claude")
        )
        llm_cfg = self.config.get("llm", {})
        try:
            self.llm_client = create_client(
                backend=backend,
                model=llm_cfg.get("model"),
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_tokens", 4096),
            )
        except Exception as exc:
            self.logger.warning(
                "create_client() failed (%s) — trying get_default_client()", exc
            )
            try:
                self.llm_client = get_default_client()
            except Exception as inner_exc:
                self.logger.warning("Default LLM client unavailable: %s", inner_exc)
                self.llm_client = None

    def _init_data(self) -> None:
        try:
            from src.data import AlpacaDataClient, DataPreprocessor
        except Exception as exc:
            self.logger.warning("Data modules unavailable: %s", exc)
            return

        data_cfg = self.config.get("data", {})
        proc_cfg = data_cfg.get("processing", {})

        try:
            self.preprocessor = DataPreprocessor(
                fill_method=proc_cfg.get("fill_method", "forward"),
                outlier_threshold=proc_cfg.get("outlier_threshold", 5.0),
                min_trading_days=proc_cfg.get("min_trading_days", 252),
            )
        except Exception as exc:
            self.logger.warning("Preprocessor unavailable: %s", exc)
            self.preprocessor = None

        # Data client selection: CCXT (free, crypto) -> Alpaca (equities) -> None
        data_source = os.getenv("DATA_SOURCE", "auto").lower()
        self.data_client = None

        if data_source in ("ccxt", "auto"):
            try:
                from src.data.ingestion.ccxt_client import CCXTDataClient
                exchange = os.getenv("CCXT_EXCHANGE", "kraken")
                self.data_client = CCXTDataClient(exchange=exchange)
                self.logger.info("Data client: CCXT (%s) — free, no API key", exchange)
            except Exception as exc:
                if data_source == "ccxt":
                    self.logger.warning("CCXT data client failed: %s", exc)
                else:
                    self.logger.debug("CCXT not available, trying Alpaca: %s", exc)

        if self.data_client is None and data_source in ("alpaca", "auto"):
            try:
                settings = get_settings()
                self.data_client = AlpacaDataClient(
                    api_key=settings.alpaca_api_key or None,
                    secret_key=settings.alpaca_secret_key or None,
                )
                self.logger.info("Data client: Alpaca")
            except Exception as exc:
                self.logger.warning("Market data client unavailable: %s", exc)
                self.data_client = None

    def _init_features(self) -> None:
        feat_cfg = self.config.get("features", {})

        try:
            from src.features import (
                FundamentalAnalyzer,
                SentimentAnalyzer,
                TechnicalFeatureEngine,
            )
        except Exception as exc:
            self.logger.warning("Feature modules unavailable: %s", exc)
            return

        try:
            self.tech_features = TechnicalFeatureEngine(
                config=feat_cfg.get("technical", {})
            )
        except Exception as exc:
            self.logger.warning("Technical features unavailable: %s", exc)
            self.tech_features = None

        try:
            sent_cfg = feat_cfg.get("sentiment", {})
            self.sent_features = SentimentAnalyzer(
                model_name=sent_cfg.get("model", "yiyanghkust/finbert-tone"),
                batch_size=sent_cfg.get("batch_size", 32),
                max_length=sent_cfg.get("max_length", 512),
            )
        except Exception as exc:
            self.logger.warning("Sentiment features unavailable: %s", exc)
            self.sent_features = None

        try:
            fund_cfg = feat_cfg.get("fundamental", {})
            lookback_periods = fund_cfg.get("lookback_periods", 4)
            self.fund_features = FundamentalAnalyzer(
                lookback_periods=lookback_periods
            )
        except Exception as exc:
            self.logger.warning("Fundamental features unavailable: %s", exc)
            self.fund_features = None

    def _init_foundation_model(self) -> None:
        fm_cfg = self.config.get("foundation_model", {})
        model_type = fm_cfg.get("model_type", "timesfm")
        try:
            if model_type == "ttm":
                from src.models import TTMPredictor

                self.foundation_model = TTMPredictor(**fm_cfg.get("ttm", {}))
            else:
                from src.models import TimesFMPredictor

                self.foundation_model = TimesFMPredictor(**fm_cfg.get("timesfm", {}))
            self.logger.info("Foundation model loaded: %s", model_type)
        except Exception as exc:
            self.logger.warning("Foundation model unavailable: %s", exc)
            self.foundation_model = None

    def _init_agents(self) -> None:
        try:
            from src.agents import (
                FundamentalAnalyst,
                ManagerAgent,
                RAGSystem,
                RiskAnalyst,
                SentimentAnalyst,
                TechnicalAnalyst,
            )
        except Exception as exc:
            self.logger.warning("Agent modules unavailable: %s", exc)
            return

        agent_cfg = self.config.get("llm", {}).get("agents", {})

        self.fundamental_analyst = self._try_construct(
            FundamentalAnalyst,
            "fundamental analyst",
            llm_client=self.llm_client,
            config=agent_cfg.get("fundamental_analyst", {}),
        )
        self.sentiment_analyst = self._try_construct(
            SentimentAnalyst,
            "sentiment analyst",
            llm_client=self.llm_client,
            config=agent_cfg.get("sentiment_analyst", {}),
        )
        self.technical_analyst = self._try_construct(
            TechnicalAnalyst,
            "technical analyst",
            llm_client=self.llm_client,
            config=agent_cfg.get("technical_analyst", {}),
        )
        self.risk_analyst = self._try_construct(
            RiskAnalyst,
            "risk analyst",
            llm_client=self.llm_client,
            config=agent_cfg.get("risk_analyst", {}),
        )
        self.manager = self._try_construct(
            ManagerAgent,
            "manager agent",
            llm_client=self.llm_client,
            config=agent_cfg.get("manager", {}),
        )

        rag_cfg = self.config.get("llm", {}).get("rag", {})
        if rag_cfg.get("enabled", False):
            self.rag = self._try_construct(RAGSystem, "RAG system", config=rag_cfg)

    def _init_options(self) -> None:
        try:
            from src.options import (
                GreeksCalculator,
                IVSForecaster,
                OptionsAnalyst,
                OptionsChainManager,
                OptionsFlowAnalyzer,
                OptionsStrategyBuilder,
                VolatilitySurface,
            )
        except Exception as exc:
            self.logger.warning("Options modules unavailable: %s", exc)
            return

        self.options_chains = self._try_construct(
            OptionsChainManager,
            "options chain manager",
            backend="yfinance",
        )
        self.greeks_calc = self._try_construct(
            GreeksCalculator,
            "greeks calculator",
        )
        self.vol_surface = self._try_construct(
            VolatilitySurface,
            "volatility surface",
            chains_provider=self.options_chains,
            greeks_provider=self.greeks_calc,
            history_provider=self.data_client,
        )
        self.options_strategies = self._try_construct(
            OptionsStrategyBuilder,
            "options strategy builder",
            chains_provider=self.options_chains,
            greeks_provider=self.greeks_calc,
        )
        self.options_flow = self._try_construct(
            OptionsFlowAnalyzer,
            "options flow analyzer",
            chain_manager=self.options_chains,
            greeks_calculator=self.greeks_calc,
        )
        self.ivs_forecaster = self._try_construct(
            IVSForecaster,
            "IVS forecaster",
        )
        self.options_analyst = self._try_construct(
            OptionsAnalyst,
            "options analyst",
            llm_client=self.llm_client,
            chain_manager=self.options_chains,
            greeks_calculator=self.greeks_calc,
            vol_surface=self.vol_surface,
            strategy_builder=self.options_strategies,
            flow_analyzer=self.options_flow,
        )

    def _init_execution(self) -> None:
        try:
            from src.execution import (
                AlpacaBroker,
                OrderManagementSystem,
                PaperBroker,
                TradierBroker,
            )
            from src.execution.alerts import AlertManager
            from src.execution.risk_engine import RiskEngine
        except Exception as exc:
            self.logger.warning("Execution modules unavailable: %s", exc)
            return

        env_cfg = self.config.get("environment", {})
        settings = None
        try:
            settings = get_settings()
        except Exception:
            settings = None

        try:
            if self.mode == "live":
                broker_name = os.getenv("BROKER", "alpaca")
                if broker_name == "binance":
                    try:
                        from src.execution.binance_broker import BinanceBroker
                        self.broker = BinanceBroker(
                            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
                            futures=os.getenv("BINANCE_FUTURES", "false").lower() == "true",
                        )
                    except ImportError:
                        self.logger.error("python-binance not installed for Binance broker")
                        self.broker = None
                elif broker_name == "tradier":
                    self.broker = TradierBroker()
                else:
                    api_key = os.getenv("ALPACA_API_KEY") or getattr(
                        settings, "alpaca_api_key", None
                    )
                    secret_key = os.getenv("ALPACA_SECRET_KEY") or getattr(
                        settings, "alpaca_secret_key", None
                    )
                    self.broker = AlpacaBroker(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper=False,
                    )
            else:
                self.broker = PaperBroker(
                    initial_cash=env_cfg.get("initial_balance", 100_000)
                )
        except Exception as exc:
            self.logger.warning("Broker unavailable: %s", exc)
            self.broker = None

        self.risk_engine = self._try_construct(
            RiskEngine,
            "risk engine",
            config=env_cfg,
        )
        self.oms = self._try_construct(
            OrderManagementSystem,
            "order management system",
            broker=self.broker,
            risk_engine=self.risk_engine,
        )
        self.alert_manager = self._try_construct(
            AlertManager,
            "alert manager",
        )

    def _init_rlaif(self) -> None:
        rlaif_cfg = self.config.get("rlaif", {})
        base_storage = Path(rlaif_cfg.get("storage_path", "./data/rlaif"))
        base_storage.mkdir(parents=True, exist_ok=True)

        try:
            from src.rlaif.preference_generator import PreferenceGenerator
        except Exception as exc:
            self.logger.warning("Preference generator unavailable: %s", exc)
            self.preference_gen = None
            self.preference_generator = None
            self.outcome_tracker = None
            self.options_outcome_tracker = None
            self.reward_model = None
            self.rlaif_finetuner = None
            return

        preferences_cfg = rlaif_cfg.get("preferences", {})
        self.preference_gen = self._try_construct(
            PreferenceGenerator,
            "preference generator",
            storage_path=base_storage,
            min_preference_margin=preferences_cfg.get("min_preference_margin", 0.05),
            hold_period_days=preferences_cfg.get("hold_period_days", 5),
        )
        self.preference_generator = self.preference_gen

        try:
            from src.rlaif.outcome_tracker import OutcomeTracker
        except Exception as exc:
            self.logger.warning("Outcome tracker unavailable: %s", exc)
            OutcomeTracker = None

        try:
            from src.rlaif.options_outcome_tracker import OptionsOutcomeTracker
        except Exception as exc:
            self.logger.warning("Options outcome tracker unavailable: %s", exc)
            OptionsOutcomeTracker = None

        feedback_cfg = rlaif_cfg.get("feedback", {})
        self.outcome_tracker = None
        if OutcomeTracker is not None and self.preference_gen is not None and self.data_client is not None:
            self.outcome_tracker = self._try_construct(
                OutcomeTracker,
                "outcome tracker",
                preference_generator=self.preference_gen,
                data_client=self.data_client,
                storage_path=base_storage / "positions",
                update_interval_seconds=feedback_cfg.get("update_interval_seconds", 60),
                auto_close_after_days=feedback_cfg.get("auto_close_after_days", 5),
            )

        self.options_outcome_tracker = None
        if OptionsOutcomeTracker is not None and self.preference_gen is not None:
            self.options_outcome_tracker = self._try_construct(
                OptionsOutcomeTracker,
                "options outcome tracker",
                preference_generator=self.preference_gen,
                storage_path=base_storage / "options_outcomes",
            )

        self.reward_model = None
        self.rlaif_finetuner = None

    def _init_autotrader(self) -> None:
        try:
            from src.autotrader import (
                CompositeMetric,
                ExperimentLog,
                ExperimentOrchestrator,
                ExperimentRunner,
                MarketSentinel,
                SafetyGuard,
                StrategyHotSwapper,
                ThesisGenerator,
            )
        except Exception as exc:
            self.logger.warning("AutoTrader modules unavailable: %s", exc)
            return

        at_cfg = {}
        try:
            at_cfg = load_config("configs/autotrader.yaml").get("autotrader", {})
        except Exception:
            self.logger.debug("autotrader.yaml not found, using defaults")

        if not at_cfg.get("enabled", False):
            self.logger.info("AutoTrader disabled in config")
            return

        exp_cfg = at_cfg.get("experiment", {})
        weights = exp_cfg.get("composite_weights", {})
        metric = CompositeMetric(
            sharpe_weight=weights.get("sharpe", 0.35),
            return_weight=weights.get("return", 0.30),
            drawdown_weight=weights.get("drawdown", 0.20),
            hit_rate_weight=weights.get("hit_rate", 0.15),
        )

        safety_cfg = at_cfg.get("safety", {})
        safety = SafetyGuard(
            max_experiments_per_hour=safety_cfg.get("max_experiments_per_hour", 12),
            max_swaps_per_day=safety_cfg.get("max_swaps_per_day", 6),
            min_improvement_threshold=safety_cfg.get("min_improvement_threshold", 0.01),
            max_consecutive_crashes=safety_cfg.get("max_consecutive_crashes", 10),
        )
        if self.risk_engine:
            safety.set_risk_engine(self.risk_engine)

        log_cfg = at_cfg.get("logging", {})
        experiment_log = ExperimentLog(
            path=log_cfg.get("results_tsv", "data/autotrader/experiment_results.tsv")
        )

        runner = ExperimentRunner(
            data_client=self.data_client,
            preprocessor=self.preprocessor,
            technical_engine=self.tech_features,
            composite_metric=metric,
            safety=safety,
            commission_bps=exp_cfg.get("commission_bps", 5.0),
            slippage_bps=exp_cfg.get("slippage_bps", 2.0),
            default_symbols=exp_cfg.get("symbols", ["AAPL", "MSFT", "SPY"]),
            default_lookback_months=exp_cfg.get("backtest_lookback_months", 6),
        )

        sentinel = MarketSentinel(
            regime_detector=getattr(self, "regime_detector", None),
            technical_engine=self.tech_features,
            data_client=self.data_client,
            config=at_cfg.get("sentinel", {}),
        )

        thesis_cfg = at_cfg.get("thesis", {})
        thesis_gen = ThesisGenerator(
            claude_client=self.llm_client,
            model=thesis_cfg.get("llm_model", "claude-sonnet-4-6-20250514"),
            temperature=thesis_cfg.get("temperature", 0.8),
        )

        swapper = StrategyHotSwapper(
            oms=self.oms,
            risk_engine=self.risk_engine,
            safety=safety,
            strategies_dir=log_cfg.get("strategies_dir", "data/autotrader/strategies"),
            audit_log_path=log_cfg.get("audit_log", "data/autotrader/audit.jsonl"),
        )

        # Portfolio strategist: AI-driven capital allocation & timing
        self.strategist = None
        try:
            from src.autotrader.strategist import PortfolioStrategist
            from src.autotrader.settings_manager import SettingsManager
            settings_mgr = SettingsManager()
            risk_pref = settings_mgr.get("RISK_PREFERENCE") or "moderate"
            auto_reassess = settings_mgr.get("AUTO_REASSESS").strip().lower() != "false"
            try:
                reassess_interval_min = int(settings_mgr.get("REASSESS_INTERVAL_MIN") or "60")
            except ValueError:
                reassess_interval_min = 60

            self.strategist = PortfolioStrategist(
                llm_client=self.llm_client,
                broker=self.broker,
            )
            # Initial assessment
            directive = self.strategist.assess(
                risk_preference=risk_pref,
                reassess_after_minutes_override=reassess_interval_min,
            )
            self.logger.info(
                "Portfolio strategist: style=%s, risk=%.0f%%, tier=%s",
                directive.strategy_style,
                directive.risk_budget_pct * 100,
                directive.capital_tier,
            )
        except Exception as exc:
            self.logger.debug("Portfolio strategist unavailable: %s", exc)
            auto_reassess = True
            reassess_interval_min = 60

        # RLAIF bridge: connect experiment outcomes to reward model training
        rlaif_callback = None
        try:
            from src.autotrader.rlaif_bridge import RLAIFBridge
            self.rlaif_bridge = RLAIFBridge(
                preference_generator=self.preference_gen,
                reward_model=self.reward_model,
            )
            rlaif_callback = self.rlaif_bridge.get_rlaif_callback()
            self.logger.info("RLAIF bridge connected to autotrader")
        except Exception as exc:
            self.logger.debug("RLAIF bridge unavailable: %s", exc)
            self.rlaif_bridge = None

        self.autotrader = ExperimentOrchestrator(
            sentinel=sentinel,
            thesis_gen=thesis_gen,
            runner=runner,
            swapper=swapper,
            safety=safety,
            log=experiment_log,
            mode=at_cfg.get("mode", "continuous"),
            improvement_threshold=exp_cfg.get("improvement_threshold", 0.01),
            time_budget_seconds=exp_cfg.get("time_budget_seconds", 300),
            rlaif_callback=rlaif_callback,
            strategist=self.strategist,
            risk_engine=self.risk_engine,
            auto_reassess=auto_reassess,
            reassess_interval_minutes=reassess_interval_min,
        )

        # Apply initial directive if available
        if self.strategist and self.strategist.current_directive:
            self.autotrader._apply_directive(self.strategist.current_directive)

        self.logger.info(
            "AutoTrader initialised (mode=%s, symbols=%s)",
            at_cfg.get("mode", "continuous"),
            exp_cfg.get("symbols", []),
        )

    def _init_scheduler(self) -> None:
        try:
            from src.execution.scheduler import TradingScheduler
        except Exception as exc:
            self.logger.warning("Scheduler unavailable: %s", exc)
            return

        scheduler_cfg = self.config.get("scheduler", {})
        scheduler_cfg.setdefault("symbols", self.config.get("data", {}).get("assets", []))
        scheduler_cfg.setdefault("dry_run", self.mode != "live")

        self.scheduler = self._try_construct(
            TradingScheduler,
            "trading scheduler",
            config=scheduler_cfg,
            pipeline=self,
            oms=self.oms,
            risk_engine=self.risk_engine,
            alert_manager=self.alert_manager,
        )

    # ------------------------------------------------------------------
    # Internal utility helpers
    # ------------------------------------------------------------------
    def _try_construct(self, cls, label: str, **kwargs):
        try:
            return cls(**kwargs)
        except Exception as exc:
            self.logger.warning("%s unavailable: %s", label, exc)
            return None

    def _safe_call(self, obj: Any, method_names: List[str], *args, **kwargs):
        if obj is None:
            return None
        for method_name in method_names:
            method = getattr(obj, method_name, None)
            if callable(method):
                return method(*args, **kwargs)
        raise AttributeError(
            f"{type(obj).__name__} has none of the methods {method_names}"
        )

    def _fetch_market_data(self, symbol: str):
        if self.data_client is None:
            raise RuntimeError("market data client is unavailable")

        data_cfg = self.config.get("data", {})
        interval = (
            data_cfg.get("sources", {})
            .get("market_data", {})
            .get("bar_interval", "1Day")
        )
        lookback_days = (
            data_cfg.get("sources", {})
            .get("market_data", {})
            .get("lookback_days", 365)
        )

        if hasattr(self.data_client, "download_latest"):
            return self.data_client.download_latest(
                symbols=symbol,
                days=lookback_days,
                timeframe=interval,
                use_cache=True,
            )

        if hasattr(self.data_client, "get_bars"):
            return self.data_client.get_bars(symbol)

        raise AttributeError(
            f"Unsupported market data client: {type(self.data_client).__name__}"
        )

    def _preprocess_market_data(self, raw_data, symbol: str):
        if self.preprocessor is None:
            return raw_data

        try:
            return self._safe_call(
                self.preprocessor,
                ["preprocess", "process"],
                raw_data,
                symbol=symbol,
            )
        except TypeError:
            return self._safe_call(
                self.preprocessor,
                ["preprocess", "process"],
                raw_data,
            )

    def _compute_technical_features(self, processed) -> Dict[str, Any]:
        if self.tech_features is None:
            return {"status": "unavailable", "reason": "technical engine not loaded"}

        try:
            result = self._safe_call(self.tech_features, ["compute_all", "compute"], processed)
            if hasattr(result, "tail"):
                return {
                    "status": "ok",
                    "columns": list(result.columns),
                    "latest": result.tail(1).to_dict(orient="records")[0],
                }
            return {"status": "ok", "value": result}
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def _compute_sentiment_features(self, symbol: str) -> Dict[str, Any]:
        if self.sent_features is None:
            return {"status": "unavailable", "reason": "sentiment engine not loaded"}

        if hasattr(self.sent_features, "compute"):
            try:
                return {"status": "ok", "value": self.sent_features.compute(symbol)}
            except Exception as exc:
                return {"status": "error", "reason": str(exc)}

        return {
            "status": "unavailable",
            "reason": "pipeline does not yet wire news text into sentiment analyzer",
        }

    def _compute_fundamental_features(self, symbol: str) -> Dict[str, Any]:
        if self.fund_features is None:
            return {
                "status": "unavailable",
                "reason": "fundamental engine not loaded",
            }

        if hasattr(self.fund_features, "compute"):
            try:
                return {"status": "ok", "value": self.fund_features.compute(symbol)}
            except Exception as exc:
                return {"status": "error", "reason": str(exc)}

        return {
            "status": "unavailable",
            "reason": "pipeline does not yet wire financial statement data into fundamental analyzer",
        }

    def _run_foundation_forecast(self, processed):
        if self.foundation_model is None:
            return None
        try:
            return self._safe_call(self.foundation_model, ["predict"], processed)
        except Exception as exc:
            self.logger.warning("Foundation model forecast failed: %s", exc)
            return None

    def _run_agent(self, agent: Any, context: Dict[str, Any]):
        if agent is None:
            return {"status": "unavailable"}
        try:
            return self._safe_call(agent, ["analyze", "run"], context)
        except Exception as exc:
            self.logger.warning("Agent %s failed: %s", type(agent).__name__, exc)
            return {"error": str(exc)}

    def _run_manager(self, agent_results: Dict[str, Any], context: Dict[str, Any]):
        if self.manager is None:
            return {
                "decision": "hold",
                "confidence": 0.0,
                "reasoning": "manager agent unavailable",
                "trades": [],
            }

        try:
            if hasattr(self.manager, "synthesize"):
                return self.manager.synthesize(
                    agent_results=agent_results,
                    context=context,
                )
            if hasattr(self.manager, "analyze"):
                mgr_context = dict(context)
                mgr_context["agent_results"] = agent_results
                return self.manager.analyze(mgr_context)
        except Exception as exc:
            self.logger.warning("Manager synthesis failed: %s", exc)
            return {"error": str(exc), "trades": []}

        return {"error": "manager has no supported entrypoint", "trades": []}

    def _get_options_chain(self, symbol: str):
        if self.options_chains is None:
            raise RuntimeError("options chain manager unavailable")

        expirations = []
        if hasattr(self.options_chains, "get_expirations"):
            expirations = self.options_chains.get_expirations(symbol)
        expiration = expirations[0] if expirations else None

        if hasattr(self.options_chains, "fetch_chain"):
            return self.options_chains.fetch_chain(symbol, expiration=expiration)
        if hasattr(self.options_chains, "get_chain"):
            if expiration is not None:
                return self.options_chains.get_chain(symbol, expiration)
            return self.options_chains.get_chain(symbol)

        raise AttributeError(
            f"Unsupported options chain manager: {type(self.options_chains).__name__}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, symbol: str) -> Dict[str, Any]:
        self.logger.info("Analysing %s", symbol)
        started = time.time()

        base_result: Dict[str, Any] = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": 0.0,
            "features": {},
            "foundation_forecast": None,
            "agents": {},
            "manager_decision": {"decision": "hold", "trades": []},
            "recommended_trades": [],
        }

        try:
            raw_data = self._fetch_market_data(symbol)
            processed = self._preprocess_market_data(raw_data, symbol)
        except Exception as exc:
            base_result["manager_decision"] = {
                "error": f"market data unavailable: {exc}",
                "trades": [],
            }
            base_result["elapsed_seconds"] = round(time.time() - started, 2)
            return base_result

        features = {
            "technical": self._compute_technical_features(processed),
            "sentiment": self._compute_sentiment_features(symbol),
            "fundamental": self._compute_fundamental_features(symbol),
        }
        forecast = self._run_foundation_forecast(processed)

        context = {
            "symbol": symbol,
            "features": features,
            "forecast": forecast,
            "price_data": processed,
        }

        agent_results = {
            "fundamental": self._run_agent(self.fundamental_analyst, context),
            "sentiment": self._run_agent(self.sentiment_analyst, context),
            "technical": self._run_agent(self.technical_analyst, context),
            "risk": self._run_agent(self.risk_analyst, context),
        }

        if self.options_analyst is not None:
            agent_results["options"] = self._run_agent(self.options_analyst, context)
        else:
            agent_results["options"] = {"status": "unavailable"}

        manager_decision = self._run_manager(agent_results, context)

        base_result.update(
            {
                "elapsed_seconds": round(time.time() - started, 2),
                "features": features,
                "foundation_forecast": forecast,
                "agents": agent_results,
                "manager_decision": manager_decision,
                "recommended_trades": (
                    manager_decision.get("trades", [])
                    if isinstance(manager_decision, dict)
                    else []
                ),
            }
        )
        return base_result

    def analyze_options(self, symbol: str) -> Dict[str, Any]:
        self.logger.info("Options analysis for %s", symbol)

        result: Dict[str, Any] = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chain_summary": None,
            "greeks": None,
            "vol_surface": None,
            "unusual_flow": [],
            "recommended_strategies": [],
        }

        try:
            chain = self._get_options_chain(symbol)
            result["chain_summary"] = chain
        except Exception as exc:
            result["error"] = f"options data unavailable: {exc}"
            return result

        try:
            if self.greeks_calc is not None:
                if hasattr(self.greeks_calc, "compute"):
                    result["greeks"] = self.greeks_calc.compute(chain)
                elif hasattr(self.greeks_calc, "compute_greeks"):
                    result["greeks"] = {"status": "manual_api_required"}
        except Exception as exc:
            result["greeks"] = {"error": str(exc)}

        try:
            if self.vol_surface is not None:
                if hasattr(self.vol_surface, "build_surface"):
                    result["vol_surface"] = self.vol_surface.build_surface(symbol)
                elif hasattr(self.vol_surface, "build"):
                    result["vol_surface"] = self.vol_surface.build(chain)
        except Exception as exc:
            result["vol_surface"] = {"error": str(exc)}

        try:
            if self.options_flow is not None:
                if hasattr(self.options_flow, "detect_unusual_activity"):
                    result["unusual_flow"] = self.options_flow.detect_unusual_activity(symbol)
                elif hasattr(self.options_flow, "analyze"):
                    result["unusual_flow"] = self.options_flow.analyze(symbol)
        except Exception as exc:
            result["unusual_flow"] = [{"error": str(exc)}]

        try:
            if self.options_strategies is not None and hasattr(
                self.options_strategies, "recommend"
            ):
                result["recommended_strategies"] = self.options_strategies.recommend(
                    symbol=symbol,
                    chain=chain,
                    vol_surface=result.get("vol_surface"),
                )
        except Exception as exc:
            result["recommended_strategies"] = [{"error": str(exc)}]

        return result

    def run_autotrader(
        self,
        mode: Optional[str] = None,
    ) -> None:
        """Start the autonomous strategy experimentation loop.

        This is the main entry point for the self-improving quant trader.
        It runs the autoresearch-style NEVER STOP loop: detect market events,
        generate strategy modifications via Claude, backtest them, and
        hot-swap winners into live trading.
        """
        if self.autotrader is None:
            raise RuntimeError(
                "AutoTrader not initialised. Check configs/autotrader.yaml"
            )

        if mode:
            self.autotrader.mode = mode

        self.logger.info(
            "Starting AutoTrader (mode=%s)", self.autotrader.mode
        )

        def portfolio_state_fn():
            if self.oms:
                return self.oms.get_portfolio_state()
            return None

        self.autotrader.run(
            portfolio_state_fn=portfolio_state_fn,
        )

    def autotrader_status(self) -> Dict[str, Any]:
        """Return autotrader status for CLI/API."""
        if self.autotrader is None:
            return {"enabled": False, "reason": "not initialised"}
        return self.autotrader.status()

    def run_paper(self):
        self.logger.info("Starting PAPER trading")
        if self.scheduler is None:
            raise RuntimeError("scheduler unavailable")

        assets = self.config.get("data", {}).get("assets", [])
        self.scheduler.start(pipeline=self, symbols=assets, mode="paper")

    def run_live(self):
        self.logger.info("Starting LIVE trading")
        if self.mode != "live":
            raise RuntimeError(
                "Pipeline must be initialised with mode='live' to trade live"
            )
        if self.scheduler is None:
            raise RuntimeError("scheduler unavailable")

        assets = self.config.get("data", {}).get("assets", [])
        self.scheduler.start(pipeline=self, symbols=assets, mode="live")

    def backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        symbols = symbols or self.config.get("data", {}).get("assets", [])
        self.logger.info(
            "Backtesting %s → %s on %d symbols", start_date, end_date, len(symbols)
        )

        results: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
            "trades": [],
            "equity_curve": [],
            "metrics": {},
        }

        if self.data_client is None or self.preprocessor is None:
            results["error"] = "market data stack unavailable"
            return results

        try:
            from src.backtesting.runner import run_benchmark_backtest

            cost_cfg = self.config.get("backtesting", {}).get("costs", {})
            return run_benchmark_backtest(
                data_client=self.data_client,
                preprocessor=self.preprocessor,
                technical_engine=self.tech_features,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Day",
                commission_bps=float(cost_cfg.get("commission_bps", 0.0) or 0.0),
                slippage_bps=float(cost_cfg.get("slippage_bps", 0.0) or 0.0),
            )
        except Exception as exc:
            self.logger.warning("Benchmark backtest unavailable, falling back: %s", exc)

        errors = 0
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0

        for sym in symbols:
            try:
                if hasattr(self.data_client, "download_bars"):
                    hist = self.data_client.download_bars(
                        symbols=sym,
                        start=start_date,
                        end=end_date,
                        timeframe="1Day",
                    )
                elif hasattr(self.data_client, "get_bars"):
                    hist = self.data_client.get_bars(sym, start=start_date, end=end_date)
                else:
                    raise AttributeError("unsupported market data client API")

                processed = self._preprocess_market_data(hist, sym)
                analysis = self.analyze(sym)
                manager = analysis.get("manager_decision", {})
                decision = str(
                    manager.get("action") or manager.get("decision") or "hold"
                ).lower()
                if decision in {"buy", "long"}:
                    buy_signals += 1
                elif decision in {"sell", "short"}:
                    sell_signals += 1
                else:
                    hold_signals += 1

                results["trades"].append(
                    {
                        "symbol": sym,
                        "rows": len(processed),
                        "decision": decision,
                        "analysis": analysis,
                    }
                )
            except Exception as exc:
                errors += 1
                self.logger.warning("Backtest error on %s: %s", sym, exc)
                results["trades"].append({"symbol": sym, "error": str(exc)})

        total = len(symbols)
        results["metrics"] = {
            "symbols_processed": total,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "error_count": errors,
            "success_rate": round((total - errors) / total, 4) if total else 0.0,
        }
        return results

    def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_client": type(self.llm_client).__name__ if self.llm_client else "unavailable",
            "foundation_model": (
                type(self.foundation_model).__name__
                if self.foundation_model is not None
                else "unavailable"
            ),
            "broker": type(self.broker).__name__ if self.broker else "unavailable",
            "rag_enabled": self.rag is not None,
            "rlaif_enabled": self.config.get("rlaif", {}).get("enabled", False),
            "asset_universe": self.config.get("data", {}).get("assets", []),
            "agents": [
                name
                for name, agent in [
                    ("FundamentalAnalyst", self.fundamental_analyst),
                    ("SentimentAnalyst", self.sentiment_analyst),
                    ("TechnicalAnalyst", self.technical_analyst),
                    ("RiskAnalyst", self.risk_analyst),
                    ("OptionsAnalyst", self.options_analyst),
                    ("ManagerAgent", self.manager),
                ]
                if agent is not None
            ],
            "components": {
                "data_client": self.data_client is not None,
                "preprocessor": self.preprocessor is not None,
                "technical_features": self.tech_features is not None,
                "sentiment_features": self.sent_features is not None,
                "fundamental_features": self.fund_features is not None,
                "scheduler": self.scheduler is not None,
            },
        }

        try:
            info["broker_connected"] = (
                self.broker.is_connected() if hasattr(self.broker, "is_connected") else False
            )
        except Exception:
            info["broker_connected"] = False

        try:
            info["risk_state"] = (
                self.risk_engine.status() if hasattr(self.risk_engine, "status") else "ok"
            )
        except Exception:
            info["risk_state"] = "unknown"

        return info

    def kill(self):
        self.logger.critical("KILL SWITCH ACTIVATED")

        if self.oms is not None and hasattr(self.oms, "cancel_all"):
            try:
                self.oms.cancel_all()
            except Exception as exc:
                self.logger.error("cancel_all failed: %s", exc)

        if self.broker is not None and hasattr(self.broker, "flatten_all"):
            try:
                self.broker.flatten_all()
            except Exception as exc:
                self.logger.error("flatten_all failed: %s", exc)

        if self.scheduler is not None and hasattr(self.scheduler, "stop"):
            try:
                self.scheduler.stop()
            except Exception as exc:
                self.logger.error("scheduler stop failed: %s", exc)

        if self.alert_manager is not None and hasattr(self.alert_manager, "send"):
            try:
                self.alert_manager.send(
                    level="critical",
                    message="KILL SWITCH — all orders cancelled, positions flattened",
                )
            except Exception as exc:
                self.logger.error("alert send failed: %s", exc)


if __name__ == "__main__":
    import json

    pipeline = TradingPipeline()
    print(json.dumps(pipeline.status(), indent=2, default=str))
