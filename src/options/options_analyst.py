"""
Options Analyst Agent

Specialized agent for options-market analysis within the multi-agent
trading system.  Integrates:

- Volatility surface construction and anomaly detection
- Greeks calculation and risk profiling
- Unusual options flow detection (sweeps, V/OI spikes, IV moves)
- Volatility risk premium (VRP) estimation
- Concrete trade recommendations with defined risk

Works with both Claude and MLX LLM backends via BaseAgentV2.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..agents.base_agent_v2 import BaseAgentV2
from ..agents.base_agent import AgentResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptionsAnalyst(BaseAgentV2):
    """
    Options-specialist agent that analyses vol surfaces, Greeks, flow,
    and generates actionable options-trade recommendations.
    """

    SYSTEM_PROMPT = """\
You are an expert options analyst and derivatives strategist.

Your expertise covers:
- Volatility surface analysis: term structure, skew, smile dynamics
- Greeks interpretation: delta, gamma, vega, theta risk profiles
- Unusual options flow: sweep orders, V/OI spikes, IV jumps
- Volatility risk premium (VRP): realised vs implied vol comparison
- Strategy construction: verticals, straddles, calendars, ratios, condors

Analysis framework:
1. Assess the volatility surface for anomalies (skew kinks, term-structure inversions).
2. Evaluate Greeks exposure and how it changes with spot/vol shifts.
3. Interpret options flow for institutional positioning signals.
4. Compare implied vol to realised vol (VRP) for edge identification.
5. Recommend specific strategies with defined risk/reward and Greeks profile.

Output format:
1. Volatility Assessment (surface shape, anomalies, term structure)
2. Flow Analysis (unusual activity, sweep direction, P/C ratio)
3. Greeks & Risk Profile (key exposures, scenario analysis)
4. VRP Analysis (IV vs RV, percentile context)
5. Trade Recommendations (strategy, strikes, expiry, size, max loss, target)
6. Score (-1 to 1, bearish to bullish) and Confidence (0% to 100%)
"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        chain_manager: Optional[Any] = None,
        greeks_calculator: Optional[Any] = None,
        vol_surface: Optional[Any] = None,
        flow_analyzer: Optional[Any] = None,
        strategy_builder: Optional[Any] = None,
        use_rag: bool = False,
    ):
        """
        Args:
            llm_client: LLM backend (ClaudeClient or MLXClient).
            chain_manager: OptionsChainManager instance.
            greeks_calculator: GreeksCalculator instance.
            vol_surface: VolatilitySurface instance.
            flow_analyzer: OptionsFlowAnalyzer instance.
            strategy_builder: OptionsStrategyBuilder instance.
            use_rag: Whether to augment prompts with RAG context.
        """
        super().__init__(
            name="OptionsAnalyst",
            system_prompt=self.SYSTEM_PROMPT,
            llm_client=llm_client,
            use_rag=use_rag,
        )
        self.chain_manager = chain_manager
        self.greeks_calculator = greeks_calculator
        self.vol_surface = vol_surface
        self.flow_analyzer = flow_analyzer
        self.strategy_builder = strategy_builder

    # ------------------------------------------------------------------
    # Core analysis (BaseAgentV2 contract)
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Perform comprehensive options analysis.

        Pipeline:
            1. Fetch options chain
            2. Calculate Greeks surface
            3. Build vol surface, detect anomalies
            4. Analyse flow for unusual activity
            5. Estimate VRP
            6. Format everything into an LLM prompt
            7. Get LLM analysis and trade recommendations
            8. Return AgentResponse with options-specific data

        Args:
            symbol: Underlying ticker.
            data: Dict with at least 'current_price'; may also contain
                  'historical_vol', 'price_history', etc.
            context: Optional additional context string.

        Returns:
            AgentResponse with options analysis.
        """
        logger.info(f"OptionsAnalyst analyzing {symbol}")
        spot = float(data.get("current_price", data.get("price", 0)))

        # --- 1. Options chain ---
        chain = self._fetch_chain(symbol)
        chain_summary = self._summarize_chain(chain)

        # --- 2. Greeks ---
        greeks_summary = self._compute_greeks_summary(symbol, chain, spot)

        # --- 3. Vol surface ---
        vol_summary = self._build_vol_surface_summary(symbol, chain, spot)

        # --- 4. Flow analysis ---
        flow_summary = self._analyze_flow(symbol, chain)

        # --- 5. VRP ---
        vrp = self._estimate_vrp(data)

        # --- Assemble prompt data ---
        options_data: Dict[str, Any] = {
            "spot_price": spot,
            "chain_summary": chain_summary,
            "greeks_summary": greeks_summary,
            "vol_surface": vol_summary,
            "flow_analysis": flow_summary,
            "vrp": vrp,
        }
        # Merge caller-provided data (price history, etc.)
        options_data.update({
            k: v for k, v in data.items()
            if k not in options_data
        })

        # --- 6. Build prompt ---
        steps = [
            "Assess the volatility surface: describe shape, skew, term structure, "
            "and flag any anomalies (kinks, inversions, unusual wings).",
            "Interpret options flow signals: unusual activity, sweep direction, "
            "put/call ratio, and what institutional positioning may look like.",
            "Evaluate Greeks risk profile: key delta/gamma/vega/theta exposures "
            "and how they shift under spot or vol changes.",
            "Compare implied volatility to realised volatility (VRP): is there "
            "an edge in selling or buying premium?",
            "Recommend 1-3 specific options strategies with exact strikes, "
            "expiration, position size rationale, max loss, and profit target.",
            "Provide an overall Score (-1 = strong bearish to 1 = strong bullish) "
            "and Confidence (0% to 100%).",
        ]

        prompt = self._build_cot_prompt(symbol, options_data, steps, context)
        prompt += (
            "\n\nIMPORTANT: End your response with:\n"
            "Score: [number from -1 to 1]\n"
            "Confidence: [percentage from 0% to 100%]\n"
        )

        # --- 7. LLM call ---
        try:
            response_text = self.llm.complete(
                prompt=prompt,
                system=self.system_prompt,
                max_tokens=2500,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"LLM call failed for {symbol}: {e}")
            response_text = (
                f"Options analysis for {symbol} could not be completed: {e}"
            )

        # --- 8. Parse & return ---
        parsed = self._parse_response(
            response_text, extract_score=True, extract_confidence=True
        )

        return AgentResponse(
            analysis=parsed["analysis"],
            score=parsed["score"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            data={
                "options_data": options_data,
                "flow_signal": flow_summary,
                "vrp": vrp,
            },
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Trade recommendations
    # ------------------------------------------------------------------

    def recommend_trades(
        self,
        symbol: str,
        data: Dict[str, Any],
        max_risk_per_trade: float = 0.02,
    ) -> List[Dict]:
        """
        Generate concrete options trade recommendations.

        Args:
            symbol: Underlying ticker.
            data: Must contain at least 'current_price' and 'portfolio_value'.
            max_risk_per_trade: Max fraction of portfolio to risk per trade.

        Returns:
            List of trade recommendation dicts:
            [
                {
                    'strategy': str,
                    'direction': 'bullish' | 'bearish' | 'neutral',
                    'legs': [...],
                    'max_loss': float,
                    'max_profit': float | 'unlimited',
                    'breakeven': float | List[float],
                    'greeks': {...},
                    'rationale': str,
                    'confidence': float,
                },
                ...
            ]
        """
        recommendations: List[Dict] = []
        spot = float(data.get("current_price", data.get("price", 0)))
        portfolio = float(data.get("portfolio_value", 100_000))
        max_risk = portfolio * max_risk_per_trade

        try:
            # Get flow signal for directional bias
            flow = self._analyze_flow(symbol, self._fetch_chain(symbol))
            composite = flow.get("composite_score", 0.0)
            signal = flow.get("signal", "neutral")

            # Get VRP for premium selling/buying bias
            vrp = self._estimate_vrp(data)
            vrp_value = vrp.get("vrp", 0.0)

            # --- Strategy selection logic ---

            if signal == "bullish" and composite > 0.3:
                # Strong bullish flow → bull call spread or long call
                recommendations.append(
                    self._build_recommendation(
                        strategy="bull_call_spread",
                        direction="bullish",
                        symbol=symbol,
                        spot=spot,
                        max_risk=max_risk,
                        rationale=(
                            f"Strong bullish flow (score={composite:.2f}): "
                            f"elevated call sweeps and low P/C ratio."
                        ),
                        confidence=min(abs(composite), 0.9),
                    )
                )

            elif signal == "bearish" and composite < -0.3:
                # Strong bearish flow → bear put spread
                recommendations.append(
                    self._build_recommendation(
                        strategy="bear_put_spread",
                        direction="bearish",
                        symbol=symbol,
                        spot=spot,
                        max_risk=max_risk,
                        rationale=(
                            f"Strong bearish flow (score={composite:.2f}): "
                            f"put sweeps and elevated P/C ratio."
                        ),
                        confidence=min(abs(composite), 0.9),
                    )
                )

            # VRP-based strategies
            if vrp_value > 0.05:
                # IV > RV → sell premium
                recommendations.append(
                    self._build_recommendation(
                        strategy="iron_condor",
                        direction="neutral",
                        symbol=symbol,
                        spot=spot,
                        max_risk=max_risk,
                        rationale=(
                            f"Positive VRP ({vrp_value:.1%}): implied vol exceeds "
                            f"realised vol — premium selling edge."
                        ),
                        confidence=min(vrp_value * 5, 0.85),
                    )
                )
            elif vrp_value < -0.05:
                # RV > IV → buy premium (straddle/strangle)
                recommendations.append(
                    self._build_recommendation(
                        strategy="long_straddle",
                        direction="neutral",
                        symbol=symbol,
                        spot=spot,
                        max_risk=max_risk,
                        rationale=(
                            f"Negative VRP ({vrp_value:.1%}): realised vol exceeds "
                            f"implied vol — cheap premium, expect movement."
                        ),
                        confidence=min(abs(vrp_value) * 5, 0.80),
                    )
                )

            # If strategy_builder is available, try to enrich with exact legs
            if self.strategy_builder is not None:
                for rec in recommendations:
                    try:
                        enriched = self.strategy_builder.build(
                            symbol=symbol,
                            strategy=rec["strategy"],
                            spot=spot,
                            max_risk=max_risk,
                        )
                        if enriched:
                            rec.update(enriched)
                    except Exception as e:
                        logger.warning(
                            f"Strategy builder enrichment failed: {e}"
                        )

            logger.info(
                f"Generated {len(recommendations)} trade recommendations "
                f"for {symbol}"
            )

        except Exception as e:
            logger.error(f"Error generating trade recommendations for {symbol}: {e}")

        return recommendations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_chain(self, symbol: str):
        """Fetch options chain via chain_manager."""
        if self.chain_manager is None:
            return None
        try:
            return self.chain_manager.get_chain(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch chain for {symbol}: {e}")
            return None

    @staticmethod
    def _summarize_chain(chain) -> Dict[str, Any]:
        """Create a compact summary of the options chain for the LLM."""
        if chain is None or (hasattr(chain, "empty") and chain.empty):
            return {"status": "no chain data available"}

        try:
            import pandas as pd

            type_col = (
                "option_type" if "option_type" in chain.columns else "type"
            )
            total_volume = int(chain["volume"].sum())
            total_oi = int(chain["open_interest"].sum())
            n_expiries = chain["expiry"].nunique() if "expiry" in chain.columns else 0
            strikes = chain["strike"].nunique() if "strike" in chain.columns else 0

            return {
                "total_contracts": len(chain),
                "total_volume": total_volume,
                "total_open_interest": total_oi,
                "num_expiries": n_expiries,
                "num_strikes": strikes,
                "avg_iv": round(
                    float(chain["implied_volatility"].mean()), 4
                ) if "implied_volatility" in chain.columns else None,
            }
        except Exception:
            return {"status": "chain data present but could not summarize"}

    def _compute_greeks_summary(
        self, symbol: str, chain, spot: float
    ) -> Dict[str, Any]:
        """Compute aggregate Greeks summary."""
        if self.greeks_calculator is None or chain is None:
            return {"status": "greeks calculator not available"}

        try:
            greeks = self.greeks_calculator.calculate_chain_greeks(
                chain, spot
            )
            if greeks is not None and not greeks.empty:
                return {
                    "avg_delta": round(float(greeks["delta"].mean()), 4),
                    "avg_gamma": round(float(greeks["gamma"].mean()), 6),
                    "avg_vega": round(float(greeks["vega"].mean()), 4),
                    "avg_theta": round(float(greeks["theta"].mean()), 4),
                    "max_gamma_strike": float(
                        greeks.loc[greeks["gamma"].idxmax(), "strike"]
                    ) if "strike" in greeks.columns else None,
                }
        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")

        return {"status": "greeks calculation failed"}

    def _build_vol_surface_summary(
        self, symbol: str, chain, spot: float
    ) -> Dict[str, Any]:
        """Build vol surface and detect anomalies."""
        if self.vol_surface is None or chain is None:
            return {"status": "vol surface module not available"}

        try:
            surface = self.vol_surface.build(chain, spot)
            anomalies = self.vol_surface.detect_anomalies(surface)
            return {
                "skew_slope": surface.get("skew_slope"),
                "term_structure": surface.get("term_structure"),
                "atm_iv": surface.get("atm_iv"),
                "anomalies": anomalies,
            }
        except Exception as e:
            logger.warning(f"Vol surface build failed: {e}")

        return {"status": "vol surface construction failed"}

    def _analyze_flow(self, symbol: str, chain) -> Dict[str, Any]:
        """Run flow analysis if analyzer is available."""
        if self.flow_analyzer is None:
            return {"status": "flow analyzer not available", "composite_score": 0.0}

        try:
            return self.flow_analyzer.aggregate_flow_signal(symbol)
        except Exception as e:
            logger.warning(f"Flow analysis failed: {e}")
            return {"status": "flow analysis failed", "composite_score": 0.0}

    @staticmethod
    def _estimate_vrp(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate Volatility Risk Premium (VRP = IV - RV).

        Positive VRP means implied vol is higher than realised vol,
        suggesting a premium-selling edge.
        """
        iv = data.get("implied_volatility", data.get("avg_iv"))
        rv = data.get("realised_volatility", data.get("historical_vol"))

        if iv is not None and rv is not None:
            iv = float(iv)
            rv = float(rv)
            vrp = iv - rv
            return {
                "implied_vol": round(iv, 4),
                "realised_vol": round(rv, 4),
                "vrp": round(vrp, 4),
                "vrp_pct": round(vrp / max(rv, 0.001), 4),
                "signal": (
                    "sell_premium" if vrp > 0.03
                    else "buy_premium" if vrp < -0.03
                    else "neutral"
                ),
            }

        return {"status": "insufficient vol data for VRP", "vrp": 0.0}

    @staticmethod
    def _build_recommendation(
        strategy: str,
        direction: str,
        symbol: str,
        spot: float,
        max_risk: float,
        rationale: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Build a skeleton trade recommendation dict."""
        # Approximate strike placement
        if strategy == "bull_call_spread":
            long_strike = round(spot * 0.99, 2)
            short_strike = round(spot * 1.04, 2)
            legs = [
                {"action": "buy", "type": "call", "strike": long_strike},
                {"action": "sell", "type": "call", "strike": short_strike},
            ]
            width = short_strike - long_strike
            max_loss = min(max_risk, width * 100)
            max_profit = (width * 100) - max_loss

        elif strategy == "bear_put_spread":
            long_strike = round(spot * 1.01, 2)
            short_strike = round(spot * 0.96, 2)
            legs = [
                {"action": "buy", "type": "put", "strike": long_strike},
                {"action": "sell", "type": "put", "strike": short_strike},
            ]
            width = long_strike - short_strike
            max_loss = min(max_risk, width * 100)
            max_profit = (width * 100) - max_loss

        elif strategy == "iron_condor":
            put_short = round(spot * 0.95, 2)
            put_long = round(spot * 0.92, 2)
            call_short = round(spot * 1.05, 2)
            call_long = round(spot * 1.08, 2)
            legs = [
                {"action": "buy", "type": "put", "strike": put_long},
                {"action": "sell", "type": "put", "strike": put_short},
                {"action": "sell", "type": "call", "strike": call_short},
                {"action": "buy", "type": "call", "strike": call_long},
            ]
            width = max(put_short - put_long, call_long - call_short)
            max_loss = min(max_risk, width * 100)
            max_profit = max_loss * 0.33  # typical IC credit ≈ 1/3 width

        elif strategy == "long_straddle":
            atm = round(spot, 2)
            legs = [
                {"action": "buy", "type": "call", "strike": atm},
                {"action": "buy", "type": "put", "strike": atm},
            ]
            max_loss = min(max_risk, spot * 0.05 * 100)
            max_profit = "unlimited"

        else:
            legs = []
            max_loss = max_risk
            max_profit = "undefined"

        return {
            "strategy": strategy,
            "direction": direction,
            "symbol": symbol,
            "spot_at_entry": spot,
            "legs": legs,
            "max_loss": round(float(max_loss) if isinstance(max_loss, (int, float)) else 0, 2),
            "max_profit": max_profit if isinstance(max_profit, str) else round(float(max_profit), 2),
            "rationale": rationale,
            "confidence": round(confidence, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
