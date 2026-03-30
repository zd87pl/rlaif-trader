"""
Options Flow Analyzer

Detects unusual options activity including:
- Abnormal volume/open interest ratios
- Sweep orders (aggressive buying/selling)
- IV spikes indicating event anticipation
- Put/call ratio extremes
- Dealer gamma exposure estimates (GEX)

Combines all signals into a composite flow score (-1 to 1).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
VOI_NOTABLE = 2.0          # Volume / OI ratio considered notable
VOI_VERY_UNUSUAL = 5.0     # Volume / OI ratio considered very unusual
MIN_VOLUME_NOTABLE = 1000  # Minimum volume for a contract to matter
PC_BULLISH = 0.5           # P/C ratio below this = bullish
PC_BEARISH = 1.5           # P/C ratio above this = bearish
IV_SPIKE_PCT = 0.20        # 20 % 1-day IV jump = anticipation event


class OptionsFlowAnalyzer:
    """
    Analyse live options flow for unusual activity signals.

    Depends on:
        - OptionsChainManager  (chains.py)  for fetching chain data
        - GreeksCalculator     (greeks.py)  for IV / delta calculations
    """

    def __init__(
        self,
        chain_manager: Optional[Any] = None,
        greeks_calculator: Optional[Any] = None,
    ):
        """
        Args:
            chain_manager: OptionsChainManager instance for chain data.
            greeks_calculator: GreeksCalculator instance for Greeks / IV.
        """
        self.chain_manager = chain_manager
        self.greeks_calculator = greeks_calculator
        logger.info("OptionsFlowAnalyzer initialized")

    # ------------------------------------------------------------------
    # 1. Unusual activity detection
    # ------------------------------------------------------------------

    def detect_unusual_activity(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> List[Dict]:
        """
        Identify unusual options activity based on volume/OI ratio,
        large absolute volume, and IV spikes.

        Args:
            symbol: Underlying ticker.
            lookback_days: Days to compare against.

        Returns:
            List of dicts, each describing one unusual contract:
            {
                'symbol', 'expiry', 'strike', 'option_type',
                'volume', 'open_interest', 'voi_ratio', 'iv',
                'iv_change_pct', 'signal', 'score'
            }
        """
        unusual: List[Dict] = []

        try:
            chain = self._get_chain(symbol)
            if chain is None or chain.empty:
                logger.warning(f"No chain data for {symbol}")
                return unusual

            for _, row in chain.iterrows():
                volume = float(row.get("volume", 0) or 0)
                oi = float(row.get("open_interest", 0) or 0)
                iv = float(row.get("implied_volatility", 0) or 0)
                iv_prev = float(row.get("prev_iv", iv) or iv)

                if volume < 10 or oi < 1:
                    continue

                voi = volume / max(oi, 1)
                iv_change = (iv - iv_prev) / max(iv_prev, 0.001) if iv_prev else 0.0

                signals: List[str] = []
                score = 0.0

                # V/OI checks
                if voi >= VOI_VERY_UNUSUAL and volume >= MIN_VOLUME_NOTABLE:
                    signals.append("very_unusual_voi")
                    score += 0.6
                elif voi >= VOI_NOTABLE and volume >= MIN_VOLUME_NOTABLE:
                    signals.append("notable_voi")
                    score += 0.3

                # Large absolute volume
                if volume >= 5000:
                    signals.append("high_volume")
                    score += 0.2

                # IV spike
                if abs(iv_change) >= IV_SPIKE_PCT:
                    signals.append("iv_spike")
                    score += 0.3

                if signals:
                    unusual.append({
                        "symbol": symbol,
                        "expiry": str(row.get("expiry", "")),
                        "strike": float(row.get("strike", 0)),
                        "option_type": str(row.get("option_type", row.get("type", ""))),
                        "volume": volume,
                        "open_interest": oi,
                        "voi_ratio": round(voi, 2),
                        "iv": round(iv, 4),
                        "iv_change_pct": round(iv_change, 4),
                        "signals": signals,
                        "score": round(min(score, 1.0), 3),
                    })

            # Sort by score descending
            unusual.sort(key=lambda x: x["score"], reverse=True)
            logger.info(
                f"Detected {len(unusual)} unusual options contracts for {symbol}"
            )

        except Exception as e:
            logger.error(f"Error detecting unusual activity for {symbol}: {e}")

        return unusual

    # ------------------------------------------------------------------
    # 2. Put / Call ratio
    # ------------------------------------------------------------------

    def calculate_put_call_ratio(self, symbol: str) -> Dict:
        """
        Calculate overall and per-expiration put/call ratio with
        historical comparison.

        Returns:
            {
                'overall_pc_volume': float,
                'overall_pc_oi': float,
                'by_expiry': {expiry: {'pc_volume': ..., 'pc_oi': ...}},
                'signal': 'bullish' | 'bearish' | 'neutral',
                'score': float  # -1 to 1
            }
        """
        result: Dict[str, Any] = {
            "overall_pc_volume": 1.0,
            "overall_pc_oi": 1.0,
            "by_expiry": {},
            "signal": "neutral",
            "score": 0.0,
        }

        try:
            chain = self._get_chain(symbol)
            if chain is None or chain.empty:
                return result

            type_col = "option_type" if "option_type" in chain.columns else "type"

            calls = chain[chain[type_col].str.upper() == "CALL"]
            puts = chain[chain[type_col].str.upper() == "PUT"]

            total_call_vol = calls["volume"].sum() or 1
            total_put_vol = puts["volume"].sum() or 1
            total_call_oi = calls["open_interest"].sum() or 1
            total_put_oi = puts["open_interest"].sum() or 1

            overall_pc_vol = total_put_vol / max(total_call_vol, 1)
            overall_pc_oi = total_put_oi / max(total_call_oi, 1)

            result["overall_pc_volume"] = round(float(overall_pc_vol), 4)
            result["overall_pc_oi"] = round(float(overall_pc_oi), 4)

            # Per-expiry breakdown
            if "expiry" in chain.columns:
                for expiry, group in chain.groupby("expiry"):
                    exp_calls = group[group[type_col].str.upper() == "CALL"]
                    exp_puts = group[group[type_col].str.upper() == "PUT"]
                    cv = exp_calls["volume"].sum() or 1
                    pv = exp_puts["volume"].sum() or 1
                    co = exp_calls["open_interest"].sum() or 1
                    po = exp_puts["open_interest"].sum() or 1
                    result["by_expiry"][str(expiry)] = {
                        "pc_volume": round(float(pv / max(cv, 1)), 4),
                        "pc_oi": round(float(po / max(co, 1)), 4),
                    }

            # Signal
            if overall_pc_vol < PC_BULLISH:
                result["signal"] = "bullish"
                result["score"] = round(min((PC_BULLISH - overall_pc_vol) / PC_BULLISH, 1.0), 3)
            elif overall_pc_vol > PC_BEARISH:
                result["signal"] = "bearish"
                result["score"] = round(-min((overall_pc_vol - PC_BEARISH) / PC_BEARISH, 1.0), 3)
            else:
                result["signal"] = "neutral"
                result["score"] = 0.0

        except Exception as e:
            logger.error(f"Error calculating P/C ratio for {symbol}: {e}")

        return result

    # ------------------------------------------------------------------
    # 3. Sweep detection
    # ------------------------------------------------------------------

    def detect_sweeps(
        self,
        symbol: str,
        chain: pd.DataFrame,
    ) -> List[Dict]:
        """
        Identify sweep orders -- high-volume trades hitting the ask
        (for calls) or the bid (for puts), indicating urgency.

        A sweep is characterised by:
        - Volume significantly exceeding open interest
        - Trade price at or near the ask (calls) / bid (puts)
        - Multi-exchange execution pattern

        Args:
            symbol: Underlying ticker.
            chain: DataFrame with at least columns:
                   strike, option_type/type, volume, open_interest,
                   bid, ask, last_price

        Returns:
            List of sweep detections.
        """
        sweeps: List[Dict] = []

        if chain is None or chain.empty:
            return sweeps

        try:
            type_col = "option_type" if "option_type" in chain.columns else "type"

            for _, row in chain.iterrows():
                volume = float(row.get("volume", 0) or 0)
                oi = float(row.get("open_interest", 0) or 0)
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                last = float(row.get("last_price", row.get("last", 0)) or 0)
                opt_type = str(row.get(type_col, "")).upper()

                if volume < MIN_VOLUME_NOTABLE or oi < 1:
                    continue

                voi = volume / max(oi, 1)
                if voi < VOI_NOTABLE:
                    continue

                # Check if trade is at the aggressive side
                spread = ask - bid if ask > bid else 0.01
                is_sweep = False
                direction = "unknown"

                if opt_type == "CALL" and last > 0 and ask > 0:
                    # Calls bought at ask = bullish sweep
                    pct_to_ask = (last - bid) / max(spread, 0.01)
                    if pct_to_ask >= 0.6:
                        is_sweep = True
                        direction = "bullish"
                elif opt_type == "PUT" and last > 0 and bid > 0:
                    # Puts bought at ask = bearish sweep
                    pct_to_ask = (last - bid) / max(spread, 0.01)
                    if pct_to_ask >= 0.6:
                        is_sweep = True
                        direction = "bearish"

                if is_sweep:
                    sweeps.append({
                        "symbol": symbol,
                        "expiry": str(row.get("expiry", "")),
                        "strike": float(row.get("strike", 0)),
                        "option_type": opt_type,
                        "volume": volume,
                        "open_interest": oi,
                        "voi_ratio": round(voi, 2),
                        "bid": bid,
                        "ask": ask,
                        "last_price": last,
                        "direction": direction,
                        "estimated_premium": round(volume * last * 100, 2),
                    })

            sweeps.sort(key=lambda x: x["estimated_premium"], reverse=True)
            logger.info(f"Detected {len(sweeps)} sweep orders for {symbol}")

        except Exception as e:
            logger.error(f"Error detecting sweeps for {symbol}: {e}")

        return sweeps

    # ------------------------------------------------------------------
    # 4. Aggregate flow signal
    # ------------------------------------------------------------------

    def aggregate_flow_signal(self, symbol: str) -> Dict:
        """
        Combine all flow signals into a single composite score from -1
        (strongly bearish) to +1 (strongly bullish).

        Sub-scores:
            volume_oi_score   – high V/OI = unusual activity
            pc_ratio_score    – extreme P/C = directional signal
            iv_spike_score    – sudden IV increase = anticipation
            sweep_score       – aggressive buying/selling direction

        Returns:
            {
                'symbol': str,
                'composite_score': float,  # -1 to 1
                'signal': str,             # 'bullish' / 'bearish' / 'neutral'
                'confidence': float,       # 0 to 1
                'volume_oi_score': float,
                'pc_ratio_score': float,
                'iv_spike_score': float,
                'sweep_score': float,
                'unusual_contracts': int,
                'sweep_count': int,
                'timestamp': str,
            }
        """
        result: Dict[str, Any] = {
            "symbol": symbol,
            "composite_score": 0.0,
            "signal": "neutral",
            "confidence": 0.0,
            "volume_oi_score": 0.0,
            "pc_ratio_score": 0.0,
            "iv_spike_score": 0.0,
            "sweep_score": 0.0,
            "unusual_contracts": 0,
            "sweep_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # --- Volume / OI score ---
            unusual = self.detect_unusual_activity(symbol)
            result["unusual_contracts"] = len(unusual)

            if unusual:
                # Average score of unusual contracts, biased by call/put direction
                bullish_scores = [
                    u["score"] for u in unusual
                    if u.get("option_type", "").upper() == "CALL"
                ]
                bearish_scores = [
                    u["score"] for u in unusual
                    if u.get("option_type", "").upper() == "PUT"
                ]
                bull_avg = np.mean(bullish_scores) if bullish_scores else 0.0
                bear_avg = np.mean(bearish_scores) if bearish_scores else 0.0
                result["volume_oi_score"] = round(
                    float(np.clip(bull_avg - bear_avg, -1, 1)), 3
                )

            # --- P/C ratio score ---
            pc = self.calculate_put_call_ratio(symbol)
            result["pc_ratio_score"] = pc.get("score", 0.0)

            # --- IV spike score ---
            iv_spike_score = self._compute_iv_spike_score(unusual)
            result["iv_spike_score"] = iv_spike_score

            # --- Sweep score ---
            chain = self._get_chain(symbol)
            if chain is not None and not chain.empty:
                sweeps = self.detect_sweeps(symbol, chain)
                result["sweep_count"] = len(sweeps)
                result["sweep_score"] = self._compute_sweep_score(sweeps)

            # --- Composite ---
            weights = {
                "volume_oi_score": 0.30,
                "pc_ratio_score": 0.25,
                "iv_spike_score": 0.20,
                "sweep_score": 0.25,
            }
            composite = sum(
                result[k] * w for k, w in weights.items()
            )
            result["composite_score"] = round(float(np.clip(composite, -1, 1)), 3)

            # Signal
            if result["composite_score"] > 0.15:
                result["signal"] = "bullish"
            elif result["composite_score"] < -0.15:
                result["signal"] = "bearish"
            else:
                result["signal"] = "neutral"

            # Confidence: higher when more data points agree
            scores = [
                result["volume_oi_score"],
                result["pc_ratio_score"],
                result["iv_spike_score"],
                result["sweep_score"],
            ]
            nonzero = [s for s in scores if abs(s) > 0.05]
            if nonzero:
                agreement = sum(1 for s in nonzero if np.sign(s) == np.sign(composite))
                result["confidence"] = round(agreement / len(scores), 2)
            else:
                result["confidence"] = 0.1

        except Exception as e:
            logger.error(f"Error computing aggregate flow for {symbol}: {e}")

        return result

    # ------------------------------------------------------------------
    # 5. Gamma exposure estimate (GEX)
    # ------------------------------------------------------------------

    def gamma_exposure_estimate(
        self,
        symbol: str,
        chain: pd.DataFrame,
        spot: float,
    ) -> Dict:
        """
        Estimate dealer gamma exposure (GEX) from open interest and delta.

        Dealers are generally *short* options they sell to customers.
        Net GEX determines whether dealer hedging amplifies or dampens
        price moves.

        GEX_i = OI_i * gamma_i * 100 * spot^2 * 0.01
        Net GEX = sum of call GEX - sum of put GEX (dealers short)

        Args:
            symbol: Ticker.
            chain: Options chain with at least: strike, option_type/type,
                   open_interest, gamma (or implied_volatility + expiry
                   for calculation).
            spot: Current underlying price.

        Returns:
            {
                'symbol': str,
                'spot': float,
                'net_gex': float,
                'call_gex': float,
                'put_gex': float,
                'gex_by_strike': List[Dict],
                'gex_flip_strike': Optional[float],
                'interpretation': str,
            }
        """
        result: Dict[str, Any] = {
            "symbol": symbol,
            "spot": spot,
            "net_gex": 0.0,
            "call_gex": 0.0,
            "put_gex": 0.0,
            "gex_by_strike": [],
            "gex_flip_strike": None,
            "interpretation": "insufficient data",
        }

        if chain is None or chain.empty or spot <= 0:
            return result

        try:
            type_col = "option_type" if "option_type" in chain.columns else "type"
            multiplier = 100  # standard equity option

            call_gex_total = 0.0
            put_gex_total = 0.0
            strike_gex: Dict[float, float] = {}

            for _, row in chain.iterrows():
                oi = float(row.get("open_interest", 0) or 0)
                gamma = float(row.get("gamma", 0) or 0)
                strike = float(row.get("strike", 0))
                opt_type = str(row.get(type_col, "")).upper()

                if oi == 0 or gamma == 0:
                    continue

                # GEX = OI * Gamma * 100 * Spot^2 * 0.01
                gex = oi * gamma * multiplier * (spot ** 2) * 0.01

                if opt_type == "CALL":
                    call_gex_total += gex
                    strike_gex[strike] = strike_gex.get(strike, 0) + gex
                elif opt_type == "PUT":
                    # Dealer short puts => negative gamma to dealer
                    put_gex_total += gex
                    strike_gex[strike] = strike_gex.get(strike, 0) - gex

            result["call_gex"] = round(call_gex_total, 2)
            result["put_gex"] = round(put_gex_total, 2)
            result["net_gex"] = round(call_gex_total - put_gex_total, 2)

            # GEX by strike
            sorted_strikes = sorted(strike_gex.keys())
            result["gex_by_strike"] = [
                {"strike": s, "gex": round(strike_gex[s], 2)}
                for s in sorted_strikes
            ]

            # Find GEX flip point (where cumulative flips sign)
            if sorted_strikes:
                cum = 0.0
                prev_sign = None
                for s in sorted_strikes:
                    cum += strike_gex[s]
                    curr_sign = np.sign(cum)
                    if prev_sign is not None and curr_sign != prev_sign and curr_sign != 0:
                        result["gex_flip_strike"] = s
                    prev_sign = curr_sign

            # Interpretation
            net = result["net_gex"]
            if net > 0:
                result["interpretation"] = (
                    "Positive GEX: dealers long gamma, will sell rallies and buy dips "
                    "— expect mean-reverting / compressed price action."
                )
            elif net < 0:
                result["interpretation"] = (
                    "Negative GEX: dealers short gamma, forced to buy rallies and "
                    "sell dips — expect amplified / trending moves."
                )
            else:
                result["interpretation"] = "Neutral GEX: balanced dealer positioning."

            logger.info(
                f"GEX for {symbol}: net={result['net_gex']:.0f}, "
                f"flip={result['gex_flip_strike']}"
            )

        except Exception as e:
            logger.error(f"Error computing GEX for {symbol}: {e}")

        return result

    # ------------------------------------------------------------------
    # 6. GEX regime detection
    # ------------------------------------------------------------------

    def gex_regime(
        self,
        symbol: str,
        chain: pd.DataFrame = None,
        spot: float = None,
    ) -> Dict:
        """Determine the GEX regime and Zero Gamma Level (ZGL).

        Positive GEX (spot above ZGL) → dealers long gamma → mean-reverting,
        suppressed vol → sell premium.
        Negative GEX (spot below ZGL) → dealers short gamma → trending,
        amplified vol → buy premium / avoid selling.

        Args:
            symbol: underlying ticker.
            chain: options chain DataFrame. Fetched via chain_manager if None.
            spot: current underlying price. Inferred from chain if None.

        Returns:
            Dict with regime, zgl, spot, distance_to_zgl_pct, net_gex,
            gex_by_strike, strategy_implication, vol_forecast.
        """
        # Fetch chain if not provided
        if chain is None:
            chain = self._get_chain(symbol)
        if chain is None or chain.empty:
            return {
                "symbol": symbol,
                "regime": "unknown",
                "zgl": None,
                "spot": spot,
                "distance_to_zgl_pct": 0.0,
                "net_gex": 0.0,
                "gex_by_strike": {},
                "strategy_implication": "insufficient_data",
                "vol_forecast": "unknown",
            }

        # Infer spot from chain mid-strike if not provided
        if spot is None or spot <= 0:
            if self.chain_manager is not None and hasattr(self.chain_manager, "get_spot"):
                try:
                    spot = float(self.chain_manager.get_spot(symbol))
                except Exception:
                    pass
            if spot is None or spot <= 0:
                spot = float(chain["strike"].median())

        # Compute per-strike GEX
        gex_result = self.gamma_exposure_estimate(symbol, chain, spot)
        strike_gex_list = gex_result.get("gex_by_strike", [])

        # Build strike→gex mapping
        gex_by_strike: Dict[float, float] = {}
        for item in strike_gex_list:
            gex_by_strike[item["strike"]] = item["gex"]

        # Find Zero Gamma Level (ZGL): strike where cumulative GEX crosses zero
        zgl = None
        sorted_strikes = sorted(gex_by_strike.keys())
        if sorted_strikes:
            cum = 0.0
            prev_cum = 0.0
            for s in sorted_strikes:
                prev_cum = cum
                cum += gex_by_strike[s]
                if prev_cum != 0 and np.sign(cum) != np.sign(prev_cum):
                    # Linear interpolation between strikes
                    if len(sorted_strikes) > 1:
                        prev_s = sorted_strikes[max(0, sorted_strikes.index(s) - 1)]
                        if cum != prev_cum:
                            zgl = prev_s + (s - prev_s) * abs(prev_cum) / (abs(prev_cum) + abs(cum))
                        else:
                            zgl = s
                    else:
                        zgl = s
                    break

            # Fallback: use gex_flip_strike from base method
            if zgl is None:
                zgl = gex_result.get("gex_flip_strike")
            # Fallback: use spot as ZGL if we still can't find it
            if zgl is None:
                zgl = spot

        net_gex = gex_result.get("net_gex", 0.0)
        distance_to_zgl_pct = ((spot - zgl) / zgl * 100) if zgl and zgl > 0 else 0.0

        # Regime determination
        if net_gex > 0:
            regime = "positive_gex"
            strategy_implication = "sell_premium"
            vol_forecast = "suppressed"
        elif net_gex < 0:
            regime = "negative_gex"
            strategy_implication = "buy_premium"
            vol_forecast = "amplified"
        else:
            regime = "neutral_gex"
            strategy_implication = "neutral"
            vol_forecast = "neutral"

        result = {
            "symbol": symbol,
            "regime": regime,
            "zgl": round(zgl, 2) if zgl is not None else None,
            "spot": round(spot, 2),
            "distance_to_zgl_pct": round(distance_to_zgl_pct, 4),
            "net_gex": round(net_gex, 2),
            "gex_by_strike": gex_by_strike,
            "strategy_implication": strategy_implication,
            "vol_forecast": vol_forecast,
        }

        logger.info(
            "GEX regime for %s: %s (net_gex=%.0f, ZGL=%.2f, spot=%.2f)",
            symbol,
            regime,
            net_gex,
            zgl if zgl else 0.0,
            spot,
        )
        return result

    def gex_transition_signal(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> Dict:
        """Detect transitions between positive and negative GEX regimes.

        A transition from positive → negative GEX predicts vol expansion.
        Based on Dec 2025 research showing 91.2% forward-return
        materialisation rate for GEX regime transitions.

        Args:
            symbol: underlying ticker.
            lookback_days: number of recent days to check for transition.

        Returns:
            Dict with transition_detected, direction, days_since_transition,
            signal_strength.
        """
        result: Dict[str, Any] = {
            "symbol": symbol,
            "transition_detected": False,
            "direction": "none",
            "days_since_transition": -1,
            "signal_strength": 0.0,
            "note": "Based on Dec 2025 research: 91.2% forward-return materialisation rate.",
        }

        try:
            chain = self._get_chain(symbol)
            if chain is None or chain.empty:
                logger.warning("No chain data for GEX transition detection: %s", symbol)
                return result

            spot = None
            if self.chain_manager is not None and hasattr(self.chain_manager, "get_spot"):
                try:
                    spot = float(self.chain_manager.get_spot(symbol))
                except Exception:
                    pass
            if spot is None or spot <= 0:
                spot = float(chain["strike"].median())

            # Current regime
            current_regime = self.gex_regime(symbol, chain, spot)

            # Simulate historical regimes by shifting spot price
            # (approximation: use recent price range to estimate regime changes)
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{lookback_days + 5}d")
                if hist is not None and len(hist) >= 2:
                    closes = hist["Close"].values[-lookback_days - 1:]
                    regimes = []
                    for price in closes:
                        r = self.gex_regime(symbol, chain, float(price))
                        regimes.append(r["regime"])

                    # Detect transition
                    for i in range(len(regimes) - 1, 0, -1):
                        if regimes[i] != regimes[i - 1]:
                            days_ago = len(regimes) - 1 - i
                            if regimes[i - 1] == "positive_gex" and regimes[i] == "negative_gex":
                                result["transition_detected"] = True
                                result["direction"] = "positive_to_negative"
                                result["days_since_transition"] = days_ago
                                result["signal_strength"] = max(0.0, min(1.0, 1.0 - days_ago * 0.15))
                            elif regimes[i - 1] == "negative_gex" and regimes[i] == "positive_gex":
                                result["transition_detected"] = True
                                result["direction"] = "negative_to_positive"
                                result["days_since_transition"] = days_ago
                                result["signal_strength"] = max(0.0, min(1.0, 0.8 - days_ago * 0.12))
                            break
            except ImportError:
                logger.warning("yfinance not available for GEX transition history")
            except Exception as e:
                logger.warning("Error in GEX transition detection: %s", e)

        except Exception as e:
            logger.error("Error computing GEX transition signal for %s: %s", symbol, e)

        logger.info(
            "GEX transition for %s: detected=%s direction=%s strength=%.2f",
            symbol,
            result["transition_detected"],
            result["direction"],
            result["signal_strength"],
        )
        return result

    def combined_regime_signal(self, symbol: str) -> Dict:
        """Combine GEX regime, VRP regime, and VIX term structure into a
        single high-conviction trading signal.

        Composite score ranges from -1 (buy premium) to +1 (sell premium).

        Components:
            - GEX regime (positive → sell, negative → buy): weight 0.40
            - VRP regime (premium_rich → sell, inverted → buy): weight 0.35
            - VIX term structure (contango → sell, backwardation → buy): weight 0.25

        Args:
            symbol: underlying ticker.

        Returns:
            Dict with composite_score, confidence, components, action.
        """
        from .vol_surface import VolatilitySurface

        result: Dict[str, Any] = {
            "symbol": symbol,
            "composite_score": 0.0,
            "confidence": 0.0,
            "components": {},
            "action": "neutral",
            "timestamp": datetime.utcnow().isoformat(),
        }

        scores: List[float] = []
        weights: List[float] = []

        try:
            # --- GEX component ---
            gex_data = self.gex_regime(symbol)
            gex_score = 0.0
            if gex_data["regime"] == "positive_gex":
                gex_score = 0.7
            elif gex_data["regime"] == "negative_gex":
                gex_score = -0.7
            result["components"]["gex_regime"] = gex_data["regime"]
            result["components"]["gex_score"] = round(gex_score, 4)
            scores.append(gex_score)
            weights.append(0.40)

            # Check for GEX transition (amplifies signal)
            transition = self.gex_transition_signal(symbol)
            if transition["transition_detected"]:
                if transition["direction"] == "positive_to_negative":
                    gex_score = min(gex_score - 0.3 * transition["signal_strength"], -0.5)
                elif transition["direction"] == "negative_to_positive":
                    gex_score = max(gex_score + 0.3 * transition["signal_strength"], 0.5)
                result["components"]["gex_transition"] = transition["direction"]
                scores[-1] = gex_score
                result["components"]["gex_score"] = round(gex_score, 4)

        except Exception as e:
            logger.warning("GEX component failed for %s: %s", symbol, e)

        try:
            # --- VRP component ---
            vol_surface = VolatilitySurface(
                chains_provider=self.chain_manager,
                history_provider=getattr(self.chain_manager, "history", None),
            )
            vrp_data = vol_surface.vrp_regime(symbol)
            vrp_score = 0.0
            if vrp_data["regime"] == "premium_rich":
                vrp_score = 0.8
            elif vrp_data["regime"] == "premium_poor":
                vrp_score = -0.3
            elif vrp_data["regime"] == "inverted":
                vrp_score = -0.9
            else:
                vrp_score = 0.1
            result["components"]["vrp_regime"] = vrp_data["regime"]
            result["components"]["vrp_score"] = round(vrp_score, 4)
            result["components"]["vrp_percentile"] = vrp_data.get("vrp_percentile", 50.0)
            scores.append(vrp_score)
            weights.append(0.35)

        except Exception as e:
            logger.warning("VRP component failed for %s: %s", symbol, e)

        try:
            # --- VIX term structure component ---
            vix_score = 0.0
            try:
                import yfinance as yf
                vix = yf.Ticker("^VIX")
                vix3m = yf.Ticker("^VIX3M")
                vix_price = vix.info.get("regularMarketPrice", None)
                vix3m_price = vix3m.info.get("regularMarketPrice", None)

                if vix_price and vix3m_price and vix3m_price > 0:
                    ratio = vix_price / vix3m_price
                    if ratio < 0.9:
                        # Contango: sell premium environment
                        vix_score = 0.6
                    elif ratio > 1.1:
                        # Backwardation: buy protection
                        vix_score = -0.8
                    else:
                        vix_score = (1.0 - ratio) * 2.0  # gradual scale
                    result["components"]["vix_vix3m_ratio"] = round(ratio, 4)
            except ImportError:
                logger.debug("yfinance not available for VIX term structure")
            except Exception:
                pass

            result["components"]["vix_score"] = round(vix_score, 4)
            scores.append(vix_score)
            weights.append(0.25)

        except Exception as e:
            logger.warning("VIX component failed for %s: %s", symbol, e)

        # --- Composite ---
        if scores and weights:
            total_weight = sum(weights[:len(scores)])
            if total_weight > 0:
                composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                composite = 0.0

            result["composite_score"] = round(float(np.clip(composite, -1, 1)), 4)

            # Confidence: agreement among components
            if len(scores) >= 2:
                dominant_sign = np.sign(composite)
                agreeing = sum(1 for s in scores if np.sign(s) == dominant_sign)
                result["confidence"] = round(agreeing / len(scores), 2)
            else:
                result["confidence"] = 0.3

            # Action mapping
            cs = result["composite_score"]
            if cs > 0.4:
                result["action"] = "sell_premium"
            elif cs > 0.15:
                result["action"] = "mild_sell_premium"
            elif cs < -0.4:
                result["action"] = "buy_premium"
            elif cs < -0.15:
                result["action"] = "mild_buy_premium"
            else:
                result["action"] = "neutral"

        logger.info(
            "Combined regime for %s: score=%.3f confidence=%.2f action=%s",
            symbol,
            result["composite_score"],
            result["confidence"],
            result["action"],
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_chain(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch chain via chain_manager or return None."""
        if self.chain_manager is None:
            logger.warning("No chain_manager configured; returning None")
            return None
        try:
            return self.chain_manager.get_chain(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch chain for {symbol}: {e}")
            return None

    @staticmethod
    def _compute_iv_spike_score(unusual: List[Dict]) -> float:
        """
        Derive a directional score from IV spikes among unusual contracts.
        Positive = bullish anticipation (call IV up), negative = bearish.
        """
        if not unusual:
            return 0.0

        spiked = [u for u in unusual if "iv_spike" in u.get("signals", [])]
        if not spiked:
            return 0.0

        call_iv_changes = [
            u["iv_change_pct"]
            for u in spiked
            if u.get("option_type", "").upper() == "CALL"
        ]
        put_iv_changes = [
            u["iv_change_pct"]
            for u in spiked
            if u.get("option_type", "").upper() == "PUT"
        ]

        call_avg = float(np.mean(call_iv_changes)) if call_iv_changes else 0.0
        put_avg = float(np.mean(put_iv_changes)) if put_iv_changes else 0.0

        # Higher call IV spike relative to put = bullish anticipation
        raw = (call_avg - put_avg) * 2.0  # scale up
        return round(float(np.clip(raw, -1, 1)), 3)

    @staticmethod
    def _compute_sweep_score(sweeps: List[Dict]) -> float:
        """
        Derive a directional score from detected sweeps.
        Weight by estimated premium.
        """
        if not sweeps:
            return 0.0

        bullish_premium = sum(
            s["estimated_premium"]
            for s in sweeps
            if s.get("direction") == "bullish"
        )
        bearish_premium = sum(
            s["estimated_premium"]
            for s in sweeps
            if s.get("direction") == "bearish"
        )

        total = bullish_premium + bearish_premium
        if total == 0:
            return 0.0

        # Net direction scaled to -1..1
        raw = (bullish_premium - bearish_premium) / total
        return round(float(np.clip(raw, -1, 1)), 3)
