"""
Trading Pipeline — the main loop that makes money.

Data → Features → Agent Analysis → Risk Check → Execution → Logging

Supports:
- Alpaca (paper trading, free)
- IBKR (paper + live)
- Dry-run mode (signals only, no orders)
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from .agents import ManagerAgent, ClaudeClient, AgentResponse
from .data import AlpacaDataClient
from .features.technical import TechnicalFeatureEngine
from .execution.risk_manager import RiskManager
from .execution.portfolio import Portfolio
from .utils.logging import get_logger

logger = get_logger(__name__)


class TradingPipeline:
    """
    End-to-end trading pipeline.

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
    ):
        load_dotenv()

        self.symbols = symbols
        self.mode = mode

        # Data
        self.data_client = AlpacaDataClient()
        self.tech_engine = TechnicalFeatureEngine()

        # Agents
        self.claude_client = ClaudeClient()
        self.manager = ManagerAgent(claude_client=self.claude_client, debate_rounds=2)

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
            f"Pipeline initialized: {len(symbols)} symbols, mode={mode}"
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
        """
        Run the pipeline on a loop.

        Args:
            interval_minutes: Minutes between cycles
        """
        logger.info(f"Starting pipeline loop (interval={interval_minutes}min)")

        while True:
            try:
                # Check market hours
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

        # 1. Fetch data
        df = self._fetch_data(symbol)
        if df is None or len(df) < 50:
            return {"symbol": symbol, "action": "skip", "reason": "insufficient data"}

        # 2. Compute features
        features_df = self.tech_engine.compute_all(df)
        latest = features_df.iloc[-1]

        # 3. Build agent input
        agent_data = self._build_agent_data(symbol, features_df, latest)

        # 4. Get multi-agent analysis
        response = self.manager.analyze(symbol=symbol, data=agent_data)

        # 5. Determine action
        action = self._score_to_action(response.score)
        current_price = float(latest["close"])

        logger.info(
            f"{symbol}: score={response.score:.2f}, confidence={response.confidence:.0%}, "
            f"action={action}, price=${current_price:.2f}"
        )

        result = {
            "symbol": symbol,
            "action": action,
            "score": response.score,
            "confidence": response.confidence,
            "price": current_price,
            "reasoning": response.analysis[:300],
            "timestamp": datetime.now().isoformat(),
        }

        # 6. Execute (if not dry run and not HOLD)
        if action != "hold" and self.broker:
            execution_result = self._execute(symbol, action, response, current_price)
            result["execution"] = execution_result

        return result

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent data for a symbol."""
        try:
            df = self.data_client.download_latest(
                symbols=symbol,
                days=120,
                timeframe="1Day",
                use_cache=False,
            )
            # Extract single symbol from multi-index
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol].copy()
            return df
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            return None

    def _build_agent_data(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        latest: pd.Series,
    ) -> Dict[str, Any]:
        """Build the data dict that the multi-agent system expects."""
        # Get safe values with defaults
        def safe_float(val, default=0.0):
            try:
                v = float(val)
                return v if v == v else default  # NaN check
            except (ValueError, TypeError):
                return default

        technical = {
            "current_price": safe_float(latest.get("close")),
            "rsi": safe_float(latest.get("rsi", 50)),
            "macd": safe_float(latest.get("macd", 0)),
            "macd_signal": safe_float(latest.get("macd_signal", 0)),
            "macd_hist": safe_float(latest.get("macd_hist", 0)),
        }

        # Add SMAs if available
        for period in [20, 50, 200]:
            key = f"sma_{period}"
            if key in latest.index:
                technical[key] = safe_float(latest[key])

        # Add Bollinger Bands if available
        for key in ["bb_upper", "bb_lower", "bb_middle"]:
            if key in latest.index:
                technical[key] = safe_float(latest[key])

        # Trend description
        price = safe_float(latest.get("close"))
        sma_20 = safe_float(latest.get("sma_20", price))
        sma_50 = safe_float(latest.get("sma_50", price))
        if price > sma_20 > sma_50:
            technical["trend"] = "uptrend"
        elif price < sma_20 < sma_50:
            technical["trend"] = "downtrend"
        else:
            technical["trend"] = "sideways"

        # Risk data from price history
        returns = features_df["close"].pct_change().dropna()
        risk = {
            "volatility": safe_float(returns.std() * (252 ** 0.5)),
            "max_drawdown": safe_float(self._calc_max_drawdown(features_df["close"])),
            "sharpe_ratio": safe_float(
                returns.mean() / (returns.std() + 1e-9) * (252 ** 0.5)
            ),
            "beta": 1.0,
        }

        # Sentiment placeholder — in production, integrate news API
        sentiment = {
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "analyst_ratings": "Not available",
        }

        # Fundamentals placeholder — integrate with financial data API
        fundamentals = {
            "symbol": symbol,
            "note": "Fundamental data requires additional API integration",
        }

        return {
            "technical": technical,
            "risk": risk,
            "sentiment": sentiment,
            "fundamentals": fundamentals,
        }

    def _execute(
        self,
        symbol: str,
        action: str,
        response: AgentResponse,
        current_price: float,
    ) -> Dict[str, Any]:
        """Execute a trade through the broker with risk checks."""
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        # Risk check
        check = self.risk_manager.check_signal(
            symbol=symbol,
            action=action,
            score=response.score,
            confidence=response.confidence,
            current_price=current_price,
            account=account,
            open_positions=positions,
        )

        if not check["approved"]:
            return {"status": "rejected", "reason": check["reason"]}

        # Place order
        try:
            if action == "sell":
                order = self.broker.close_position(symbol)
            else:
                order = self.broker.market_order(
                    symbol=symbol,
                    qty=check["qty"],
                    side=action,
                )

            # Log to portfolio
            self.portfolio.log_trade(
                symbol=symbol,
                action=action,
                qty=check["qty"],
                price=current_price,
                order_id=order["order_id"],
                signal_score=response.score,
                signal_confidence=response.confidence,
                reasoning=response.analysis[:200],
            )

            return {"status": "executed", "order": order}

        except Exception as e:
            logger.error(f"Execution failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}

    def _score_to_action(self, score: float) -> str:
        """Convert agent score to action."""
        if score >= 0.3:
            return "buy"
        elif score <= -0.3:
            return "sell"
        return "hold"

    def _calc_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate max drawdown from price series."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return float(drawdown.min())

    def status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        result = {
            "mode": self.mode,
            "symbols": self.symbols,
            "portfolio_summary": self.portfolio.get_summary(),
        }
        if self.broker:
            result["account"] = self.broker.get_account()
            result["positions"] = self.broker.get_positions()
        return result
