#!/usr/bin/env python3
"""
RLAIF Trading Pipeline — Main Entry Point

Usage:
    # Dry run (signals only, no orders, no risk):
    python run_trading.py --mode dry_run --symbols AAPL MSFT NVDA

    # Alpaca paper trading:
    python run_trading.py --mode alpaca_paper --symbols AAPL MSFT

    # IBKR paper trading (requires TWS/Gateway on port 7497):
    python run_trading.py --mode ibkr_paper --symbols AAPL MSFT

    # IBKR live trading (port 7496) — REAL MONEY:
    python run_trading.py --mode ibkr_live --symbols AAPL

    # Run continuously every 60 minutes:
    python run_trading.py --mode alpaca_paper --symbols AAPL MSFT --loop --interval 60

    # Single cycle then exit:
    python run_trading.py --mode dry_run --symbols AAPL
"""

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from src.pipeline import TradingPipeline


def main():
    parser = argparse.ArgumentParser(description="RLAIF Trading Pipeline")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "alpaca_paper", "ibkr_paper", "ibkr_live"],
        default="dry_run",
        help="Trading mode (default: dry_run)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA"],
        help="Stock symbols to trade",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously on interval",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Minutes between cycles (with --loop)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.10,
        help="Max position size as %% of portfolio (default: 10%%)",
    )
    parser.add_argument(
        "--max-exposure-pct",
        type=float,
        default=0.60,
        help="Max total exposure as %% of portfolio (default: 60%%)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Min confidence to trade (default: 0.55)",
    )

    args = parser.parse_args()

    # Safety check for live trading
    if args.mode == "ibkr_live":
        print("\n" + "=" * 60)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY AT RISK")
        print("=" * 60)
        confirm = input("Type 'I UNDERSTAND' to proceed: ")
        if confirm != "I UNDERSTAND":
            print("Aborted.")
            sys.exit(0)

    risk_config = {
        "max_position_pct": args.max_position_pct,
        "max_total_exposure_pct": args.max_exposure_pct,
        "min_confidence": args.min_confidence,
    }

    pipeline = TradingPipeline(
        symbols=args.symbols,
        mode=args.mode,
        risk_config=risk_config,
    )

    if args.loop:
        pipeline.run_loop(interval_minutes=args.interval)
    else:
        results = pipeline.run_once()
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for r in results:
            symbol = r["symbol"]
            action = r.get("action", "?")
            score = r.get("score", 0)
            conf = r.get("confidence", 0)
            price = r.get("price", 0)
            exec_status = r.get("execution", {}).get("status", "n/a")

            print(f"  {symbol:6s}  {action:6s}  score={score:+.2f}  conf={conf:.0%}  "
                  f"price=${price:.2f}  exec={exec_status}")
        print("=" * 60)

        # Print portfolio status
        status = pipeline.status()
        print(f"\nPortfolio: {json.dumps(status.get('portfolio_summary', {}), indent=2)}")

        if "account" in status:
            acct = status["account"]
            print(f"\nAccount: equity=${acct['equity']:,.2f}, "
                  f"buying_power=${acct['buying_power']:,.2f}")


if __name__ == "__main__":
    main()
