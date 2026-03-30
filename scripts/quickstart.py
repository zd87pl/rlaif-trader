#!/usr/bin/env python3
"""
RLAIF Trading - Quickstart Demo
================================

A simple end-to-end demo of the options analysis pipeline.
Uses yfinance (free, no API key needed) for market data.
No broker connection required -- analysis only.

Usage:
    python scripts/quickstart.py
    python scripts/quickstart.py --symbol AAPL
"""

import argparse
import sys
import os
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def banner(text: str) -> None:
    """Print a section banner."""
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def sub_banner(text: str) -> None:
    """Print a sub-section header."""
    print(f"\n--- {text} ---")


def main():
    parser = argparse.ArgumentParser(description="RLAIF Trading Quickstart Demo")
    parser.add_argument(
        "--symbol", default="SPY", help="Ticker symbol to analyze (default: SPY)"
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()

    banner("RLAIF Trading Pipeline - Quickstart")
    print(f"  Symbol:    {symbol}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode:      Analysis only (no broker)")

    # ------------------------------------------------------------------
    # 1. Check dependencies
    # ------------------------------------------------------------------
    sub_banner("Checking dependencies")

    try:
        import yfinance as yf  # noqa: F401
        print("[OK] yfinance")
    except ImportError:
        print("[FAIL] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    try:
        import numpy as np  # noqa: F401
        print("[OK] numpy")
    except ImportError:
        print("[FAIL] numpy not installed. Run: pip install numpy")
        sys.exit(1)

    try:
        import pandas as pd  # noqa: F401
        print("[OK] pandas")
    except ImportError:
        print("[FAIL] pandas not installed. Run: pip install pandas")
        sys.exit(1)

    try:
        from scipy import interpolate  # noqa: F401
        print("[OK] scipy")
    except ImportError:
        print("[WARN] scipy not installed -- volatility surface will be limited")

    # ------------------------------------------------------------------
    # 2. Import RLAIF Trading components
    # ------------------------------------------------------------------
    sub_banner("Loading RLAIF Trading components")

    try:
        from src.options import (
            OptionsChainManager,
            GreeksCalculator,
            VolatilitySurface,
            OptionsStrategyBuilder,
            OptionsFlowAnalyzer,
        )
        print("[OK] Options modules loaded")
    except ImportError as e:
        print(f"[FAIL] Could not import options modules: {e}")
        print("       Make sure you're running from the project root directory.")
        sys.exit(1)

    try:
        from src.models import MLXModelManager
        if MLXModelManager is not None:
            print("[OK] MLX model manager available (Apple Silicon detected)")
        else:
            print("[INFO] MLX not available (requires Apple Silicon) -- using CPU")
    except ImportError:
        print("[INFO] MLX not available -- this is fine for analysis")

    # ------------------------------------------------------------------
    # 3. Options Chain
    # ------------------------------------------------------------------
    banner(f"Step 1: Fetching Options Chain for {symbol}")

    chain_mgr = OptionsChainManager(backend="yfinance")

    print(f"Fetching available expirations for {symbol}...")
    try:
        expirations = chain_mgr.get_expirations(symbol)
    except Exception as e:
        print(f"[ERROR] Failed to get expirations: {e}")
        print("        Check your network connection and ticker symbol.")
        sys.exit(1)

    if not expirations:
        print(f"[ERROR] No options expirations found for {symbol}")
        sys.exit(1)

    print(f"Found {len(expirations)} expiration dates")
    print(f"Nearest expirations: {', '.join(expirations[:5])}")

    # Use the nearest expiration for the demo
    target_exp = expirations[0]
    print(f"\nFetching chain for {target_exp}...")

    try:
        chain = chain_mgr.fetch_chain(symbol, expiration=target_exp)
    except Exception as e:
        print(f"[ERROR] Failed to fetch chain: {e}")
        sys.exit(1)

    n_calls = len(chain[chain["option_type"] == "call"]) if "option_type" in chain.columns else "?"
    n_puts = len(chain[chain["option_type"] == "put"]) if "option_type" in chain.columns else "?"
    print(f"Chain loaded: {len(chain)} contracts ({n_calls} calls, {n_puts} puts)")

    if not chain.empty:
        print(f"\nSample (first 5 rows):")
        display_cols = [c for c in ["strike", "lastPrice", "bid", "ask", "volume",
                                     "openInterest", "impliedVolatility", "option_type"]
                        if c in chain.columns]
        print(chain[display_cols].head().to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Greeks Calculation
    # ------------------------------------------------------------------
    banner("Step 2: Computing Greeks")

    greeks_calc = GreeksCalculator()
    print(f"Risk-free rate: {greeks_calc.rate:.4f}")

    # Compute Greeks for a sample ATM call
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
        if spot is None:
            hist = ticker.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        spot = None

    if spot and not chain.empty:
        print(f"Current {symbol} price: ${spot:.2f}")

        # Find nearest ATM strike
        calls = chain[chain["option_type"] == "call"] if "option_type" in chain.columns else chain
        if not calls.empty and "strike" in calls.columns:
            atm_idx = (calls["strike"] - spot).abs().idxmin()
            atm_row = calls.loc[atm_idx]
            atm_strike = float(atm_row["strike"])
            atm_price = float(atm_row.get("lastPrice", atm_row.get("ask", 0)))
            atm_iv = float(atm_row.get("impliedVolatility", 0.25))

            # Parse expiration to get T in years
            from datetime import datetime as dt
            exp_date = dt.strptime(target_exp, "%Y-%m-%d")
            T = max((exp_date - dt.now()).days / 365.0, 1 / 365.0)

            print(f"\nATM Call @ ${atm_strike:.0f} (exp {target_exp}):")
            print(f"  Price: ${atm_price:.2f}")
            print(f"  IV:    {atm_iv:.1%}")

            try:
                greeks = greeks_calc.compute_greeks(
                    S=spot, K=atm_strike, T=T, sigma=atm_iv,
                    option_type="call"
                )
                print(f"\n  Greeks:")
                for name, val in greeks.items():
                    if isinstance(val, float):
                        print(f"    {name:>8s}: {val:>10.6f}")
                    else:
                        print(f"    {name:>8s}: {val}")
            except Exception as e:
                print(f"  [WARN] Could not compute Greeks: {e}")
    else:
        print("[WARN] Could not determine spot price -- skipping Greeks demo")

    # ------------------------------------------------------------------
    # 5. Volatility Surface
    # ------------------------------------------------------------------
    banner("Step 3: Building Volatility Surface")

    vol_surface = VolatilitySurface(
        chains_provider=chain_mgr,
        greeks_provider=greeks_calc,
    )

    try:
        surface = vol_surface.build_surface(symbol)
        if surface:
            print(f"Surface built successfully for {symbol}")
            if "moneyness_range" in surface:
                print(f"  Moneyness range: {surface['moneyness_range']}")
            if "expirations_used" in surface:
                print(f"  Expirations used: {surface['expirations_used']}")
            if "n_points" in surface:
                print(f"  Data points: {surface['n_points']}")

            # Show IV at key moneyness levels if available
            if "atm_iv" in surface:
                print(f"\n  ATM IV:   {surface['atm_iv']:.1%}")
            if "skew" in surface:
                print(f"  Skew:     {surface['skew']}")
        else:
            print("[INFO] Surface returned empty -- may need more expiration data")
    except Exception as e:
        print(f"[WARN] Could not build full surface: {e}")
        print("       This can happen with limited options data.")

    # ------------------------------------------------------------------
    # 6. Flow Analysis
    # ------------------------------------------------------------------
    banner("Step 4: Analyzing Options Flow")

    flow_analyzer = OptionsFlowAnalyzer(
        chain_manager=chain_mgr,
        greeks_calculator=greeks_calc,
    )

    try:
        unusual = flow_analyzer.detect_unusual_activity(symbol)
        if unusual:
            print(f"Found {len(unusual)} unusual activity signals:")
            for i, signal in enumerate(unusual[:5], 1):
                strike = signal.get("strike", "?")
                opt_type = signal.get("option_type", "?")
                reason = signal.get("reason", signal.get("signal", "unusual volume"))
                vol = signal.get("volume", "?")
                oi = signal.get("open_interest", signal.get("openInterest", "?"))
                print(f"  {i}. {opt_type.upper():>4s} ${strike} | vol={vol} oi={oi} | {reason}")
        else:
            print(f"No unusual activity detected for {symbol} right now.")
            print("(This is normal -- unusual activity is by definition uncommon)")
    except Exception as e:
        print(f"[WARN] Flow analysis error: {e}")
        print("       This feature works best with active market data.")

    # ------------------------------------------------------------------
    # 7. Strategy Builder - Iron Condor
    # ------------------------------------------------------------------
    banner("Step 5: Building Iron Condor Strategy")

    strategy_builder = OptionsStrategyBuilder(
        chains_provider=chain_mgr,
        greeks_provider=greeks_calc,
    )

    if spot and not chain.empty:
        # Pick strikes for an iron condor roughly 5% OTM on each side
        put_long_strike = round(spot * 0.90)
        put_short_strike = round(spot * 0.95)
        call_short_strike = round(spot * 1.05)
        call_long_strike = round(spot * 1.10)

        # Snap to nearest available strikes
        available_strikes = sorted(chain["strike"].unique()) if "strike" in chain.columns else []

        def snap_strike(target, strikes):
            if not strikes:
                return target
            return min(strikes, key=lambda s: abs(s - target))

        if available_strikes:
            put_long_strike = snap_strike(put_long_strike, available_strikes)
            put_short_strike = snap_strike(put_short_strike, available_strikes)
            call_short_strike = snap_strike(call_short_strike, available_strikes)
            call_long_strike = snap_strike(call_long_strike, available_strikes)

        print(f"Spot: ${spot:.2f}")
        print(f"Expiration: {target_exp}")
        print(f"Legs:")
        print(f"  Buy  put  @ ${put_long_strike}")
        print(f"  Sell put  @ ${put_short_strike}")
        print(f"  Sell call @ ${call_short_strike}")
        print(f"  Buy  call @ ${call_long_strike}")

        try:
            ic = strategy_builder.iron_condor(
                symbol=symbol,
                expiration=target_exp,
                put_long=put_long_strike,
                put_short=put_short_strike,
                call_short=call_short_strike,
                call_long=call_long_strike,
            )
            print(f"\nIron Condor Analysis:")
            if "max_profit" in ic:
                print(f"  Max Profit:  ${ic['max_profit']:>10.2f}")
            if "max_loss" in ic:
                print(f"  Max Loss:    ${ic['max_loss']:>10.2f}")
            if "breakevens" in ic:
                bees = ic["breakevens"]
                print(f"  Breakevens:  ${bees[0]:.2f} / ${bees[1]:.2f}")
            if "probability_of_profit" in ic:
                print(f"  Est. PoP:    {ic['probability_of_profit']:.1%}")
            if "greeks" in ic:
                print(f"  Net Greeks:")
                for name, val in ic["greeks"].items():
                    if isinstance(val, (int, float)):
                        print(f"    {name:>8s}: {val:>10.4f}")
        except Exception as e:
            print(f"[WARN] Could not build iron condor: {e}")
            print("       Strikes may not be available in the chain for this expiration.")
    else:
        print("[SKIP] No spot price available -- cannot construct strategy")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("Quickstart Complete!")
    print(f"""
  What we did:
    1. Fetched live options chain for {symbol} via yfinance (free)
    2. Computed Black-Scholes Greeks for ATM options
    3. Built an implied volatility surface
    4. Scanned for unusual options flow activity
    5. Constructed an iron condor strategy with P&L analysis

  Next steps:
    - Add your API keys to .env for broker connectivity
    - Run the multi-agent analysis: python scripts/example_multi_agent.py
    - Try RLAIF training: python scripts/example_rlaif_training.py
    - See full pipeline: python scripts/example_pipeline.py
""")


if __name__ == "__main__":
    main()
