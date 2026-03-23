#!/usr/bin/env python3
"""
Backtest the trading strategy against historical data.

Usage:
    # Default: momentum + mean reversion ensemble (free, no API costs):
    python run_backtest.py --symbols AAPL MSFT --start 2024-01-01 --end 2025-01-01

    # Test individual strategies:
    python run_backtest.py --symbols AAPL --start 2024-01-01 --end 2025-01-01 --strategies momentum
    python run_backtest.py --symbols AAPL --start 2024-01-01 --end 2025-01-01 --strategies mean_reversion

    # Full ensemble with Claude agents ($$$):
    python run_backtest.py --symbols AAPL --start 2024-06-01 --end 2025-01-01 --use-agents

    # Different ensemble modes:
    python run_backtest.py --symbols AAPL MSFT NVDA --start 2024-01-01 --end 2025-01-01 \\
        --ensemble conviction_weighted
    python run_backtest.py --symbols AAPL MSFT NVDA --start 2024-01-01 --end 2025-01-01 \\
        --ensemble majority_vote

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
        "--symbols", nargs="+", default=["AAPL"], help="Stock symbols",
    )
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--interval", type=int, default=5, help="Days between signals")
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost fraction")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["momentum", "mean_reversion"],
        help="Strategies to test (momentum, mean_reversion, agent)",
    )
    parser.add_argument(
        "--ensemble",
        choices=["conviction_weighted", "weighted_average", "majority_vote"],
        default="conviction_weighted",
        help="Ensemble mode",
    )
    parser.add_argument(
        "--use-agents",
        action="store_true",
        help="Include Claude agent strategy (costs API credits)",
    )

    args = parser.parse_args()

    print(f"\nBacktest: {args.symbols}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Strategies: {args.strategies}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Agents: {'ON (Claude API)' if args.use_agents else 'OFF'}")
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
        strategies=args.strategies,
        ensemble_mode=args.ensemble,
        use_agents=args.use_agents,
    )

    result = bt.run()
    metrics = result.compute_metrics(args.capital)

    with open("backtest_results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "symbols": args.symbols,
                    "start": args.start,
                    "end": args.end,
                    "capital": args.capital,
                    "strategies": args.strategies,
                    "ensemble": args.ensemble,
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
