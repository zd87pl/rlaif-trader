"""
Trading Pipeline — the main loop that makes money.

Data → Features → News + Fundamentals → Strategy Ensemble → Risk Check → Execution → Logging

Supports:
- Multi-strategy ensemble (momentum, mean reversion, agent)
- Live news from Finnhub/Polygon/Alpaca
- Live fundamentals from yfinance + SEC EDGAR
- Alpaca and IBKR brokers (paper + live)
- Dry-run mode (signals only, no orders)
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from .agents import ClaudeClient
from .data import AlpacaDataClient, NewsAggregator, FundamentalDataAggregator
from .features.technical import TechnicalFeatureEngine
from .strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    AgentStrategy,
    StrategyEnsemble,
    Signal,
)
from .execution.risk_manager import RiskManager
from .execution.portfolio import Portfolio
from .utils.logging import get_logger

logger = get_logger(__name__)


class TradingPipeline:
    """
    End-to-end trading pipeline with multi-strategy ensemble.

    Modes:
    - "dry_run": Generate signals only, no orders (default)
    - "alpaca_paper": Execute on Alpaca paper trading
    - "ibkr_paper": Execute on IBKR paper trading (port 7497)
    - "ibkr_live": Execute on IBKR live trading (port 7496) — USE WITH CAUTION
    """

    def __init__(
        self,
        symbols: List[str],
        mode: str = "dry_run",
        risk_config: Optional[Dict[str, Any]] = None,
        enable_news: bool = True,
        enable_fundamentals: bool = True,
        enable_agents: bool = True,
        ensemble_mode: str = "conviction_weighted",
    ):
        load_dotenv()

        self.symbols = symbols
        self.mode = mode

        # Data
        self.data_client = AlpacaDataClient()
        self.tech_engine = TechnicalFeatureEngine()

        # News & fundamentals (graceful degradation if APIs unavailable)
        self.news_client = None
        self.fundamental_client = None

        if enable_news:
            try:
                self.news_client = NewsAggregator()
                logger.info("News aggregator initialized")
            except Exception as e:
                logger.warning(f"News aggregator unavailable: {e}")

        if enable_fundamentals:
            try:
                self.fundamental_client = FundamentalDataAggregator()
                logger.info("Fundamental data aggregator initialized")
            except Exception as e:
                logger.warning(f"Fundamental data unavailable: {e}")

        # Build strategy ensemble
        strategies = [
            MomentumStrategy(weight=1.0),
            MeanReversionStrategy(weight=0.8),
        ]

        if enable_agents:
            try:
                strategies.append(AgentStrategy(weight=1.5))
                logger.info("Agent strategy enabled (Claude API)")
            except Exception as e:
                logger.warning(f"Agent strategy unavailable: {e}")

        self.ensemble = StrategyEnsemble(
            strategies=strategies,
            mode=ensemble_mode,
        )

        # Risk
        risk_config = risk_config or {}
        self.risk_manager = RiskManager(**risk_config)

        # Portfolio tracking (always active)
        self.portfolio = Portfolio()

        # Broker (conditional)
        self.broker = None
        if mode == "alpaca_paper":
            from .execution.broker import AlpacaBroker
            self.broker = AlpacaBroker(paper=True)
        elif mode == "ibkr_paper":
            from .execution.ibkr_broker import IBKRBroker
            self.broker = IBKRBroker(port=7497)
        elif mode == "ibkr_live":
            from .execution.ibkr_broker import IBKRBroker
            self.broker = IBKRBroker(port=7496)

        logger.info(
            f"Pipeline initialized: {len(symbols)} symbols, mode={mode}, "
            f"{len(strategies)} strategies, ensemble={ensemble_mode}"
        )

    def run_once(self) -> List[Dict[str, Any]]:
        """
        Run one full cycle: analyze all symbols, generate signals, execute.

        Returns list of actions taken.
        """
        logger.info(f"=== Pipeline cycle start: {datetime.now().isoformat()} ===")

        results = []
        for symbol in self.symbols:
            try:
                result = self._process_symbol(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results.append({"symbol": symbol, "action": "error", "error": str(e)})

        logger.info(f"=== Pipeline cycle complete: {len(results)} symbols processed ===")
        return results

    def run_loop(self, interval_minutes: int = 60) -> None:
        """Run the pipeline on a loop."""
        logger.info(f"Starting pipeline loop (interval={interval_minutes}min)")

        while True:
            try:
                if self.broker and not self.broker.is_market_open():
                    logger.info("Market closed. Waiting...")
                    time.sleep(300)
                    continue

                self.run_once()

                logger.info(f"Sleeping {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user")
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                time.sleep(60)

    def _process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Full pipeline for one symbol."""
        logger.info(f"Processing {symbol}...")

        # 1. Fetch price data
        df = self._fetch_data(symbol)
        if df is None or len(df) < 50:
            return {"symbol": symbol, "action": "skip", "reason": "insufficient data"}

        # 2. Compute technical features
        features_df = self.tech_engine.compute_all(df)

        # 3. Fetch news & sentiment
        news_data = self._fetch_news(symbol)

        # 4. Fetch fundamentals
        fundamental_data = self._fetch_fundamentals(symbol)

        # 5. Run strategy ensemble
        signal = self.ensemble.generate_signal(
            symbol=symbol,
            price_data=df,
            features=features_df,
            news_data=news_data,
            fundamental_data=fundamental_data,
        )

        current_price = float(features_df.iloc[-1]["close"])

        logger.info(
            f"{symbol}: ensemble={signal.action} score={signal.score:+.2f} "
            f"confidence={signal.confidence:.0%} price=${current_price:.2f}"
        )

        result = {
            "symbol": symbol,
            "action": signal.action,
            "score": signal.score,
            "confidence": signal.confidence,
            "price": current_price,
            "reasoning": signal.reasoning[:400],
            "strategy": signal.strategy_name,
            "timestamp": datetime.now().isoformat(),
            "data_sources": {
                "news_articles": news_data.get("news_count", 0),
                "has_fundamentals": bool(fundamental_data.get("fundamentals")),
            },
        }

        if signal.metadata.get("individual_signals"):
            result["strategy_signals"] = signal.metadata["individual_signals"]

        # 6. Execute (if not dry run and not HOLD)
        if signal.action != "hold" and self.broker:
            execution_result = self._execute(symbol, signal, current_price)
            result["execution"] = execution_result

        return result

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent price data."""
        try:
            df = self.data_client.download_latest(
                symbols=symbol, days=120, timeframe="1Day", use_cache=False,
            )
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol].copy()
            return df
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            return None

    def _fetch_news(self, symbol: str) -> Dict[str, Any]:
        """Fetch and process news for a symbol."""
        if not self.news_client:
            return {"news_sentiment": 0.0, "social_sentiment": 0.0, "analyst_ratings": "Not available"}

        try:
            articles = self.news_client.get_news(symbol, days_back=7, max_articles=30)
            headlines = [a["headline"] for a in articles if a.get("headline")]

            # Quick keyword sentiment
            news_sentiment = 0.0
            if headlines:
                positive = {"beat", "surge", "profit", "growth", "upgrade", "strong", "record", "rally"}
                negative = {"miss", "decline", "loss", "downgrade", "weak", "cut", "fall", "risk"}
                pos = sum(1 for h in headlines for w in positive if w in h.lower())
                neg = sum(1 for h in headlines for w in negative if w in h.lower())
                total = pos + neg
                if total > 0:
                    news_sentiment = (pos - neg) / total

            social = self.news_client.get_social_sentiment(symbol)
            social_sentiment = (
                social.get("reddit_sentiment", 0) + social.get("twitter_sentiment", 0)
            ) / 2

            analyst_ratings = self.news_client.get_analyst_ratings(symbol)

            return {
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "analyst_ratings": analyst_ratings,
                "recent_headlines": headlines[:10],
                "news_count": len(headlines),
            }
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return {"news_sentiment": 0.0, "social_sentiment": 0.0, "analyst_ratings": "Not available"}

    def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data."""
        if not self.fundamental_client:
            return {}
        try:
            return self.fundamental_client.get_fundamentals(symbol)
        except Exception as e:
            logger.warning(f"Fundamentals fetch failed for {symbol}: {e}")
            return {}

    def _execute(
        self,
        symbol: str,
        signal: Signal,
        current_price: float,
    ) -> Dict[str, Any]:
        """Execute a trade through the broker with risk checks."""
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        check = self.risk_manager.check_signal(
            symbol=symbol,
            action=signal.action,
            score=signal.score,
            confidence=signal.confidence,
            current_price=current_price,
            account=account,
            open_positions=positions,
        )

        if not check["approved"]:
            return {"status": "rejected", "reason": check["reason"]}

        try:
            if signal.action == "sell":
                order = self.broker.close_position(symbol)
            else:
                order = self.broker.market_order(
                    symbol=symbol, qty=check["qty"], side=signal.action,
                )

            self.portfolio.log_trade(
                symbol=symbol,
                action=signal.action,
                qty=check["qty"],
                price=current_price,
                order_id=order["order_id"],
                signal_score=signal.score,
                signal_confidence=signal.confidence,
                reasoning=signal.reasoning[:200],
            )

            return {"status": "executed", "order": order}

        except Exception as e:
            logger.error(f"Execution failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}

    def status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        result = {
            "mode": self.mode,
            "symbols": self.symbols,
            "strategies": [s.name for s in self.ensemble.strategies],
            "ensemble_mode": self.ensemble.mode,
            "portfolio_summary": self.portfolio.get_summary(),
        }
        if self.broker:
            result["account"] = self.broker.get_account()
            result["positions"] = self.broker.get_positions()
        return result
