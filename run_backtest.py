#!/usr/bin/env python3
"""
Backtest the trading strategy against historical data.

Usage:
    # Technical-only backtest (free, no Claude API calls):
    python run_backtest.py --symbols AAPL MSFT --start 2024-01-01 --end 2025-01-01

    # Full agent backtest (uses Claude API — costs ~$0.05-0.20 per signal):
    python run_backtest.py --symbols AAPL --start 2024-06-01 --end 2025-01-01 --use-agents

    # Custom parameters:
    python run_backtest.py --symbols AAPL NVDA TSLA \\
        --start 2024-01-01 --end 2025-01-01 \\
        --capital 50000 --interval 10 --cost 0.001
"""

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from src.backtester import Backtester


def main():
    parser = argparse.ArgumentParser(description="RLAIF Backtester")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL"],
        help="Stock symbols to backtest",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-01-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Initial capital (default: $100,000)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Days between signals (default: 5)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Transaction cost as fraction (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--use-agents",
        action="store_true",
        help="Use Claude multi-agent system (costs API credits)",
    )

    args = parser.parse_args()

    print(f"\nBacktest: {args.symbols}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Agents: {'ON (Claude API)' if args.use_agents else 'OFF (technical only)'}")
    print(f"Signal interval: every {args.interval} days")
    print(f"Transaction cost: {args.cost:.2%}")
    print()

    bt = Backtester(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        transaction_cost_pct=args.cost,
        signal_interval_days=args.interval,
        use_agents=args.use_agents,
    )

    result = bt.run()
    metrics = result.compute_metrics(args.capital)

    # Save results
    with open("backtest_results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "symbols": args.symbols,
                    "start": args.start,
                    "end": args.end,
                    "capital": args.capital,
                    "use_agents": args.use_agents,
                },
                "metrics": metrics,
                "trades": result.trades,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to backtest_results.json")
    print(f"Trades: {len(result.trades)}")


if __name__ == "__main__":
    main()
