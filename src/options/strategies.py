"""Options strategy construction, payoff analysis, and scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptionsStrategyBuilder:
    """
    Construct multi-leg options strategies, compute payoffs, and score
    strategies against a volatility surface for edge detection.
    """

    def __init__(self, chains_provider=None, greeks_provider=None):
        """
        Args:
            chains_provider: object with ``get_chain(symbol, expiration)``
                returning a DataFrame with strike, bid, ask, lastPrice,
                impliedVolatility, delta, gamma, theta, vega, type/option_type.
            greeks_provider: object with aggregate greek calculation helpers.
        """
        self.chains = chains_provider
        self.greeks = greeks_provider

    # ------------------------------------------------------------------
    # Strategy constructors
    # ------------------------------------------------------------------

    def vertical_spread(
        self,
        symbol: str,
        expiration: str,
        long_strike: float,
        short_strike: float,
        option_type: str = "call",
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Bull/bear call or put vertical spread.

        For a bull call spread: buy lower strike call, sell higher strike call.
        For a bear put spread: buy higher strike put, sell lower strike put.
        """
        chain = self._get_validated_chain(symbol, expiration)
        self._validate_strikes(chain, [long_strike, short_strike], option_type)

        long_leg = self._build_leg(chain, long_strike, option_type, "buy", quantity)
        short_leg = self._build_leg(chain, short_strike, option_type, "sell", quantity)
        legs = [long_leg, short_leg]

        net_debit = long_leg["price"] - short_leg["price"]
        width = abs(long_strike - short_strike)

        if option_type == "call":
            if long_strike < short_strike:  # bull call
                max_profit = (width - net_debit) * 100 * quantity
                max_loss = net_debit * 100 * quantity
                breakevens = [long_strike + net_debit]
            else:  # bear call
                max_profit = abs(net_debit) * 100 * quantity
                max_loss = (width - abs(net_debit)) * 100 * quantity
                breakevens = [short_strike + abs(net_debit)]
        else:  # put
            if long_strike > short_strike:  # bear put
                max_profit = (width - net_debit) * 100 * quantity
                max_loss = net_debit * 100 * quantity
                breakevens = [long_strike - net_debit]
            else:  # bull put
                max_profit = abs(net_debit) * 100 * quantity
                max_loss = (width - abs(net_debit)) * 100 * quantity
                breakevens = [short_strike - abs(net_debit)]

        greeks = self._aggregate_greeks(legs)
        pop = self._estimate_pop(chain, breakevens, symbol)

        return self._package_strategy(
            name="vertical_spread",
            symbol=symbol,
            expiration=expiration,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    def iron_condor(
        self,
        symbol: str,
        expiration: str,
        put_long: float,
        put_short: float,
        call_short: float,
        call_long: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Iron condor: sell OTM put spread + sell OTM call spread."""
        chain = self._get_validated_chain(symbol, expiration)
        self._validate_strikes(chain, [put_long, put_short], "put")
        self._validate_strikes(chain, [call_short, call_long], "call")

        legs = [
            self._build_leg(chain, put_long, "put", "buy", quantity),
            self._build_leg(chain, put_short, "put", "sell", quantity),
            self._build_leg(chain, call_short, "call", "sell", quantity),
            self._build_leg(chain, call_long, "call", "buy", quantity),
        ]

        credit = (
            legs[1]["price"] + legs[2]["price"]
            - legs[0]["price"] - legs[3]["price"]
        )
        put_width = abs(put_short - put_long)
        call_width = abs(call_long - call_short)
        max_width = max(put_width, call_width)

        max_profit = credit * 100 * quantity
        max_loss = (max_width - credit) * 100 * quantity
        breakevens = [put_short - credit, call_short + credit]

        greeks = self._aggregate_greeks(legs)
        pop = self._estimate_pop(chain, breakevens, symbol)

        return self._package_strategy(
            name="iron_condor",
            symbol=symbol,
            expiration=expiration,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    def straddle(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Long straddle: buy ATM call + ATM put at same strike."""
        chain = self._get_validated_chain(symbol, expiration)
        self._validate_strikes(chain, [strike], "call")
        self._validate_strikes(chain, [strike], "put")

        legs = [
            self._build_leg(chain, strike, "call", "buy", quantity),
            self._build_leg(chain, strike, "put", "buy", quantity),
        ]

        total_debit = (legs[0]["price"] + legs[1]["price"])
        max_profit = float("inf")
        max_loss = total_debit * 100 * quantity
        breakevens = [strike - total_debit, strike + total_debit]

        greeks = self._aggregate_greeks(legs)
        pop = self._estimate_pop(chain, breakevens, symbol)

        return self._package_strategy(
            name="straddle",
            symbol=symbol,
            expiration=expiration,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    def strangle(
        self,
        symbol: str,
        expiration: str,
        put_strike: float,
        call_strike: float,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Long strangle: buy OTM put + OTM call."""
        chain = self._get_validated_chain(symbol, expiration)
        self._validate_strikes(chain, [put_strike], "put")
        self._validate_strikes(chain, [call_strike], "call")

        legs = [
            self._build_leg(chain, call_strike, "call", "buy", quantity),
            self._build_leg(chain, put_strike, "put", "buy", quantity),
        ]

        total_debit = legs[0]["price"] + legs[1]["price"]
        max_profit = float("inf")
        max_loss = total_debit * 100 * quantity
        breakevens = [put_strike - total_debit, call_strike + total_debit]

        greeks = self._aggregate_greeks(legs)
        pop = self._estimate_pop(chain, breakevens, symbol)

        return self._package_strategy(
            name="strangle",
            symbol=symbol,
            expiration=expiration,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    def butterfly(
        self,
        symbol: str,
        expiration: str,
        lower: float,
        middle: float,
        upper: float,
        option_type: str = "call",
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Long butterfly: buy 1 lower, sell 2 middle, buy 1 upper."""
        chain = self._get_validated_chain(symbol, expiration)
        self._validate_strikes(chain, [lower, middle, upper], option_type)

        legs = [
            self._build_leg(chain, lower, option_type, "buy", quantity),
            self._build_leg(chain, middle, option_type, "sell", 2 * quantity),
            self._build_leg(chain, upper, option_type, "buy", quantity),
        ]

        net_debit = (
            legs[0]["price"] + legs[2]["price"] - 2 * legs[1]["price"]
        )
        wing_width = middle - lower  # assumes symmetric wings

        max_profit = (wing_width - net_debit) * 100 * quantity
        max_loss = net_debit * 100 * quantity
        breakevens = [lower + net_debit, upper - net_debit]

        greeks = self._aggregate_greeks(legs)
        pop = self._estimate_pop(chain, breakevens, symbol)

        return self._package_strategy(
            name="butterfly",
            symbol=symbol,
            expiration=expiration,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    def calendar_spread(
        self,
        symbol: str,
        near_exp: str,
        far_exp: str,
        strike: float,
        option_type: str = "call",
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Calendar (time) spread: sell near, buy far at same strike."""
        near_chain = self._get_validated_chain(symbol, near_exp)
        far_chain = self._get_validated_chain(symbol, far_exp)
        self._validate_strikes(near_chain, [strike], option_type)
        self._validate_strikes(far_chain, [strike], option_type)

        short_leg = self._build_leg(near_chain, strike, option_type, "sell", quantity)
        short_leg["expiration"] = near_exp
        long_leg = self._build_leg(far_chain, strike, option_type, "buy", quantity)
        long_leg["expiration"] = far_exp
        legs = [long_leg, short_leg]

        net_debit = long_leg["price"] - short_leg["price"]

        # Calendar P&L is path-dependent; approximate bounds
        max_loss = net_debit * 100 * quantity
        # Max profit is hard to compute without simulation; estimate
        max_profit = max_loss * 1.5  # rough heuristic

        breakevens = [strike - net_debit, strike + net_debit]  # approximate

        greeks = self._aggregate_greeks(legs)
        pop = 0.5  # calendars are complex; default estimate

        return self._package_strategy(
            name="calendar_spread",
            symbol=symbol,
            expiration=f"{near_exp}/{far_exp}",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=breakevens,
            margin_required=max_loss,
            greeks=greeks,
            probability_of_profit=pop,
        )

    # ------------------------------------------------------------------
    # Payoff & analysis
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_payoff(strategy: Dict[str, Any], price_range: np.ndarray) -> np.ndarray:
        """Calculate strategy P&L at expiration across a range of prices.

        Args:
            strategy: strategy dict returned by a constructor method.
            price_range: 1-D array of underlying prices.

        Returns:
            1-D array of P&L values (in dollars, per-unit).
        """
        payoff = np.zeros_like(price_range, dtype=float)

        for leg in strategy["legs"]:
            strike = leg["strike"]
            price = leg["price"]
            qty = leg["quantity"]
            opt_type = leg["option_type"]
            direction = 1 if leg["side"] == "buy" else -1

            if opt_type == "call":
                intrinsic = np.maximum(price_range - strike, 0.0)
            else:
                intrinsic = np.maximum(strike - price_range, 0.0)

            leg_pnl = direction * (intrinsic - price) * qty * 100
            payoff += leg_pnl

        return payoff

    def optimal_spread(
        self,
        symbol: str,
        expiration: str,
        direction: str = "bullish",
        max_risk: float = 500.0,
        target_return: float = 1.0,
    ) -> Dict[str, Any]:
        """Auto-select best vertical spread based on chain data.

        Args:
            symbol: ticker.
            expiration: expiration date string.
            direction: 'bullish' or 'bearish'.
            max_risk: maximum dollar risk per spread.
            target_return: minimum reward/risk ratio target.

        Returns:
            Best strategy dict, or empty dict if nothing qualifies.
        """
        chain = self._get_validated_chain(symbol, expiration)

        if direction == "bullish":
            option_type = "call"
            chain_filtered = self._filter_chain(chain, option_type)
            strikes = sorted(chain_filtered["strike"].unique())
        else:
            option_type = "put"
            chain_filtered = self._filter_chain(chain, option_type)
            strikes = sorted(chain_filtered["strike"].unique(), reverse=True)

        best: Optional[Dict] = None
        best_score = -np.inf

        for i, ls in enumerate(strikes[:-1]):
            for ss in strikes[i + 1 : min(i + 6, len(strikes))]:
                try:
                    if direction == "bullish":
                        strat = self.vertical_spread(
                            symbol, expiration, ls, ss, option_type
                        )
                    else:
                        strat = self.vertical_spread(
                            symbol, expiration, ls, ss, option_type
                        )

                    ml = abs(strat["max_loss"])
                    if ml <= 0 or ml > max_risk:
                        continue

                    mp = strat["max_profit"]
                    if mp == float("inf"):
                        mp = max_risk * 5

                    rr = mp / ml
                    if rr < target_return:
                        continue

                    pop = strat.get("probability_of_profit", 0.5)
                    ev = pop * mp - (1 - pop) * ml
                    score = ev / ml if ml > 0 else 0

                    if score > best_score:
                        best_score = score
                        best = strat

                except (ValueError, KeyError):
                    continue

        if best is None:
            logger.warning(
                "No qualifying %s spread found for %s %s (max_risk=%.0f, target_rr=%.1f)",
                direction, symbol, expiration, max_risk, target_return,
            )
            return {}

        logger.info(
            "Optimal %s spread: %s max_profit=%.0f max_loss=%.0f score=%.2f",
            direction,
            best.get("name", ""),
            best["max_profit"],
            best["max_loss"],
            best_score,
        )
        return best

    def score_strategy(
        self, strategy: Dict[str, Any], vol_surface: Dict[str, Any]
    ) -> float:
        """Score a strategy based on mispricing edge from the vol surface.

        Compares each leg's market IV to the smooth surface IV.
        Positive score = strategy has edge (selling overpriced / buying underpriced).

        Args:
            strategy: strategy dict from a constructor.
            vol_surface: dict from VolatilitySurface.build_surface().

        Returns:
            Edge score (higher = more favorable).
        """
        from scipy.interpolate import RectBivariateSpline

        m_grid = vol_surface["moneyness"]
        t_grid = vol_surface["tte"]
        iv_grid = vol_surface["iv_grid"]
        spot = vol_surface["spot"]

        try:
            spline = RectBivariateSpline(m_grid, t_grid, iv_grid, kx=3, ky=3)
        except Exception:
            logger.warning("Cannot build spline for scoring")
            return 0.0

        total_edge = 0.0
        for leg in strategy["legs"]:
            strike = leg["strike"]
            moneyness = strike / spot
            tte = leg.get("tte", 30 / 365)

            moneyness = np.clip(moneyness, m_grid[0], m_grid[-1])
            tte = np.clip(tte, t_grid[0], t_grid[-1])

            fair_iv = float(spline(moneyness, tte)[0, 0])
            market_iv = leg.get("iv", fair_iv)

            iv_diff = market_iv - fair_iv
            direction = 1 if leg["side"] == "buy" else -1
            vega = abs(leg.get("vega", 0.01))

            # Edge: positive when we're selling overpriced or buying underpriced
            leg_edge = -direction * iv_diff * vega * leg["quantity"] * 100
            total_edge += leg_edge

        return float(total_edge)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_validated_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Fetch and validate an options chain."""
        if self.chains is None:
            raise RuntimeError("No chains provider configured")
        chain = self.chains.get_chain(symbol, expiration)
        if chain is None or chain.empty:
            raise ValueError(
                f"No chain data for {symbol} expiration {expiration}"
            )
        return chain

    @staticmethod
    def _validate_strikes(
        chain: pd.DataFrame, strikes: List[float], option_type: str
    ) -> None:
        """Verify that requested strikes exist in the chain."""
        type_col = "type" if "type" in chain.columns else "option_type"
        if type_col in chain.columns:
            available = chain.loc[
                chain[type_col].str.lower() == option_type.lower(), "strike"
            ].unique()
        else:
            available = chain["strike"].unique()

        for s in strikes:
            if s not in available:
                # Allow nearest strike within 0.5%
                nearest = available[np.argmin(np.abs(available - s))]
                if abs(nearest - s) / s > 0.005:
                    raise ValueError(
                        f"Strike {s} not found in chain for {option_type}. "
                        f"Available: {sorted(available)[:10]}..."
                    )

    @staticmethod
    def _filter_chain(chain: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """Filter chain to a single option type."""
        type_col = "type" if "type" in chain.columns else "option_type"
        if type_col in chain.columns:
            return chain[chain[type_col].str.lower() == option_type.lower()]
        return chain

    @staticmethod
    def _build_leg(
        chain: pd.DataFrame,
        strike: float,
        option_type: str,
        side: str,
        quantity: int,
    ) -> Dict[str, Any]:
        """Build a single leg dict from chain data."""
        type_col = "type" if "type" in chain.columns else "option_type"

        if type_col in chain.columns:
            mask = (chain["strike"] == strike) & (
                chain[type_col].str.lower() == option_type.lower()
            )
        else:
            mask = chain["strike"] == strike

        rows = chain.loc[mask]
        if rows.empty:
            # Find nearest strike
            all_strikes = chain["strike"].unique()
            nearest = all_strikes[np.argmin(np.abs(all_strikes - strike))]
            if type_col in chain.columns:
                mask = (chain["strike"] == nearest) & (
                    chain[type_col].str.lower() == option_type.lower()
                )
            else:
                mask = chain["strike"] == nearest
            rows = chain.loc[mask]
            strike = nearest

        row = rows.iloc[0]

        # Price: use mid if available, else lastPrice
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        if bid > 0 and ask > 0:
            price = (bid + ask) / 2
        else:
            price = float(row.get("lastPrice", row.get("last", row.get("mid", 0))) or 0)

        # Greeks
        iv_col = None
        for c in ("impliedVolatility", "implied_volatility", "iv", "IV"):
            if c in row.index and pd.notna(row[c]):
                iv_col = c
                break

        return {
            "strike": float(strike),
            "option_type": option_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "bid": bid,
            "ask": ask,
            "iv": float(row[iv_col]) if iv_col else 0.0,
            "delta": float(row.get("delta", 0) or 0),
            "gamma": float(row.get("gamma", 0) or 0),
            "theta": float(row.get("theta", 0) or 0),
            "vega": float(row.get("vega", 0) or 0),
        }

    @staticmethod
    def _aggregate_greeks(legs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Sum greeks across all legs with direction adjustments."""
        agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in legs:
            sign = 1 if leg["side"] == "buy" else -1
            qty = leg["quantity"]
            for g in agg:
                agg[g] += sign * qty * leg.get(g, 0)
        return agg

    @staticmethod
    def _estimate_pop(
        chain: pd.DataFrame,
        breakevens: List[float],
        symbol: str,
    ) -> float:
        """Rough probability of profit estimate using delta as proxy.

        For proper PoP, we'd need the full distribution. This uses the
        ATM IV to build a lognormal estimate.
        """
        if not breakevens or len(breakevens) < 1:
            return 0.5

        # Get approximate spot from chain
        strikes = sorted(chain["strike"].unique())
        spot = strikes[len(strikes) // 2]  # rough ATM

        # Get approximate IV
        iv_col = None
        for c in ("impliedVolatility", "implied_volatility", "iv", "IV"):
            if c in chain.columns:
                iv_col = c
                break

        if iv_col is None:
            return 0.5

        # ATM IV
        atm_idx = np.argmin(np.abs(np.array(strikes) - spot))
        atm_rows = chain[chain["strike"] == strikes[atm_idx]]
        if atm_rows.empty or atm_rows[iv_col].isna().all():
            return 0.5

        iv = float(atm_rows[iv_col].dropna().iloc[0])
        if iv <= 0:
            return 0.5

        # Assume ~30 DTE for simplicity; lognormal PoP
        tte = 30 / 365
        from scipy.stats import norm

        if len(breakevens) == 1:
            be = breakevens[0]
            d = (np.log(spot / be) + 0.5 * iv**2 * tte) / (iv * np.sqrt(tte))
            return float(norm.cdf(d))
        elif len(breakevens) == 2:
            be_low, be_high = sorted(breakevens)
            d_low = (np.log(spot / be_low) + 0.5 * iv**2 * tte) / (
                iv * np.sqrt(tte)
            )
            d_high = (np.log(spot / be_high) + 0.5 * iv**2 * tte) / (
                iv * np.sqrt(tte)
            )
            # Prob price is between breakevens (for credit strategies)
            # or outside (for debit strategies)
            prob_between = norm.cdf(d_low) - norm.cdf(d_high)
            return float(max(prob_between, 1 - prob_between))

        return 0.5

    @staticmethod
    def _package_strategy(**kwargs) -> Dict[str, Any]:
        """Package strategy components into a standardised dict."""
        return {
            "name": kwargs["name"],
            "symbol": kwargs["symbol"],
            "expiration": kwargs["expiration"],
            "legs": kwargs["legs"],
            "max_profit": kwargs["max_profit"],
            "max_loss": kwargs["max_loss"],
            "breakevens": kwargs["breakevens"],
            "margin_required": kwargs["margin_required"],
            "greeks": kwargs["greeks"],
            "probability_of_profit": kwargs["probability_of_profit"],
        }
