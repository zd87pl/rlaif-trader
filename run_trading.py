#!/usr/bin/env python3
"""
RLAIF Trading Pipeline — Main Entry Point

Usage:
    # Dry run (signals only, no orders):
    python run_trading.py --mode dry_run --symbols AAPL MSFT NVDA

    # Alpaca paper trading with full ensemble:
    python run_trading.py --mode alpaca_paper --symbols AAPL MSFT

    # IBKR paper trading:
    python run_trading.py --mode ibkr_paper --symbols AAPL MSFT

    # Technical strategies only (no Claude API costs):
    python run_trading.py --mode dry_run --symbols AAPL --no-agents

    # Conviction-weighted ensemble (most conservative):
    python run_trading.py --mode alpaca_paper --symbols AAPL --ensemble conviction_weighted

    # Run continuously every 60 minutes:
    python run_trading.py --mode alpaca_paper --symbols AAPL MSFT --loop --interval 60

    # IBKR live trading — REAL MONEY:
    python run_trading.py --mode ibkr_live --symbols AAPL --max-position-pct 0.05
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
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable Claude agent strategy (saves API costs)",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Disable news data fetching",
    )
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Disable fundamental data fetching",
    )
    parser.add_argument(
        "--ensemble",
        choices=["conviction_weighted", "weighted_average", "majority_vote"],
        default="conviction_weighted",
        help="Ensemble mode (default: conviction_weighted)",
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
        enable_news=not args.no_news,
        enable_fundamentals=not args.no_fundamentals,
        enable_agents=not args.no_agents,
        ensemble_mode=args.ensemble,
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

            # Show individual strategy signals
            if "strategy_signals" in r:
                for ss in r["strategy_signals"]:
                    print(f"    └─ {ss['strategy']:18s}  {ss['action']:6s}  "
                          f"score={ss['score']:+.2f}  conf={ss['confidence']:.0%}")

        print("=" * 60)

        status = pipeline.status()
        print(f"\nStrategies: {status.get('strategies', [])}")
        print(f"Ensemble: {status.get('ensemble_mode', '')}")
        print(f"Portfolio: {json.dumps(status.get('portfolio_summary', {}), indent=2)}")

        if "account" in status:
            acct = status["account"]
            print(f"\nAccount: equity=${acct['equity']:,.2f}, "
                  f"buying_power=${acct['buying_power']:,.2f}")


if __name__ == "__main__":
    main()
