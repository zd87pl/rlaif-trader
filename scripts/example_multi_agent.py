#!/usr/bin/env python3
"""
Example: Complete Multi-Agent Analysis Pipeline

Demonstrates the full multi-agent system:
1. Data collection (market, fundamentals, sentiment, technical)
2. RAG system with financial documents
3. Specialist agent analyses (4 agents)
4. Structured debate (2 rounds)
5. Final trading decision from ManagerAgent
6. Explainable reasoning chain

This is the CORE INNOVATION from the research - multi-agent debate
leads to 20-40% improvement over single-agent approaches.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.agents import (
    ClaudeClient,
    ManagerAgent,
    RAGSystem,
)
from src.data import AlpacaDataClient, DataPreprocessor
from src.features import TechnicalFeatureEngine, SentimentAnalyzer
from src.utils import setup_logging, set_seed

# Load environment
load_dotenv()

# Setup logging
logger = setup_logging(log_level="INFO")


def main():
    """Run complete multi-agent analysis"""
    logger.info("=" * 80)
    logger.info("MULTI-AGENT ANALYSIS PIPELINE")
    logger.info("=" * 80)

    # Set seed
    set_seed(42)

    # Configuration
    SYMBOL = "AAPL"
    DAYS = 90

    # =========================================================================
    # Step 1: Initialize Systems
    # =========================================================================
    logger.info("\n[Step 1] Initializing Systems")
    logger.info("-" * 80)

    # Initialize Claude client (shared across all agents)
    claude_client = ClaudeClient(model="claude-3-5-sonnet-20241022")

    # Initialize RAG system
    rag_system = RAGSystem(chunk_size=512, chunk_overlap=50)

    # Initialize Manager Agent (coordinates all specialist agents)
    manager = ManagerAgent(claude_client=claude_client, debate_rounds=2)

    logger.info("✓ Systems initialized")

    # =========================================================================
    # Step 2: Collect Financial Documents for RAG
    # =========================================================================
    logger.info("\n[Step 2] Building RAG Knowledge Base")
    logger.info("-" * 80)

    # Sample financial documents (in production, scrape from SEC EDGAR, news APIs, etc.)
    financial_docs = [
        "Apple Inc. Q4 2024 earnings: Revenue $94.9B, up 12% YoY. iPhone revenue $51.3B. Services revenue $20.0B, a new record.",
        "Apple's gross margin improved to 46.2%, reflecting strong Services growth and favorable product mix.",
        "Balance sheet remains strong with $170B in cash and marketable securities. Total debt $107B.",
        "Management highlighted strong customer loyalty with active installed base exceeding 2 billion devices.",
        "Services segment continues to drive margin expansion, now representing 21% of total revenue.",
        "iPhone 15 Pro sales exceeded expectations, particularly in US and Europe. China demand mixed.",
        "Apple Vision Pro launch scheduled for early 2025 targeting $3,500 price point.",
        "R&D spending increased 15% YoY as company invests in AI and spatial computing.",
        "CFO noted strong free cash flow generation of $25B in Q4, supporting capital returns.",
        "Apple upgraded by Goldman Sachs to Buy on Services growth and AI potential.",
    ]

    doc_metadata = [
        {"symbol": SYMBOL, "doc_type": "earnings", "date": "2024-11-01", "title": "Q4 2024 Earnings"},
        {"symbol": SYMBOL, "doc_type": "earnings", "date": "2024-11-01", "title": "Q4 2024 Earnings"},
        {"symbol": SYMBOL, "doc_type": "10-K", "date": "2024-10-30", "title": "FY2024 10-K"},
        {"symbol": SYMBOL, "doc_type": "earnings_call", "date": "2024-11-01", "title": "Q4 2024 Call"},
        {"symbol": SYMBOL, "doc_type": "earnings_call", "date": "2024-11-01", "title": "Q4 2024 Call"},
        {"symbol": SYMBOL, "doc_type": "news", "date": "2024-11-02", "title": "iPhone 15 Sales"},
        {"symbol": SYMBOL, "doc_type": "news", "date": "2024-10-20", "title": "Vision Pro Launch"},
        {"symbol": SYMBOL, "doc_type": "10-K", "date": "2024-10-30", "title": "FY2024 10-K"},
        {"symbol": SYMBOL, "doc_type": "earnings_call", "date": "2024-11-01", "title": "Q4 2024 Call"},
        {"symbol": SYMBOL, "doc_type": "analyst_report", "date": "2024-11-03", "title": "Goldman Upgrade"},
    ]

    # Add documents to RAG
    rag_system.add_documents(financial_docs, doc_metadata)

    rag_stats = rag_system.get_stats()
    logger.info(f"✓ RAG system loaded: {rag_stats['total_documents']} documents")

    # =========================================================================
    # Step 3: Collect Market Data
    # =========================================================================
    logger.info("\n[Step 3] Collecting Market Data")
    logger.info("-" * 80)

    try:
        # Download historical data
        client = AlpacaDataClient()
        df_raw = client.download_latest(symbols=SYMBOL, days=DAYS, timeframe="1Day")

        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df_raw, symbol=SYMBOL)

        logger.info(f"✓ Downloaded {len(df)} days of data for {SYMBOL}")
    except Exception as e:
        logger.warning(f"Could not download data: {e}. Using mock data.")

        # Mock data for demonstration
        import numpy as np
        import pandas as pd

        dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq="1D")
        df = pd.DataFrame(
            {
                "open": 170 + np.random.randn(90).cumsum(),
                "high": 172 + np.random.randn(90).cumsum(),
                "low": 168 + np.random.randn(90).cumsum(),
                "close": 170 + np.random.randn(90).cumsum(),
                "volume": np.random.randint(50_000_000, 100_000_000, 90),
            },
            index=dates,
        )
        logger.info("✓ Using mock data for demonstration")

    # =========================================================================
    # Step 4: Generate Features
    # =========================================================================
    logger.info("\n[Step 4] Generating Features")
    logger.info("-" * 80)

    # Technical indicators
    tech_engine = TechnicalFeatureEngine()
    df_tech = tech_engine.compute_all(df)

    current_price = df_tech["close"].iloc[-1]
    logger.info(f"Current Price: ${current_price:.2f}")

    # Sample sentiment analysis
    sample_news = [
        "Apple reports strong Q4 earnings, beating analyst expectations on iPhone and Services.",
        "Goldman Sachs upgrades Apple to Buy citing AI potential and Services growth.",
        "Concerns about iPhone demand in China as local competitors gain market share.",
    ]

    sentiment_analyzer = SentimentAnalyzer()
    sentiments = sentiment_analyzer.analyze(sample_news)
    agg_sentiment = sentiment_analyzer.aggregate_sentiments(sentiments)

    logger.info(f"Sentiment: {agg_sentiment['sentiment']} (score: {agg_sentiment['score']:.2f})")
    logger.info("✓ Features generated")

    # =========================================================================
    # Step 5: Prepare Data for Multi-Agent Analysis
    # =========================================================================
    logger.info("\n[Step 5] Preparing Multi-Agent Data")
    logger.info("-" * 80)

    # Compile data for each specialist agent
    multi_agent_data = {
        "fundamentals": {
            # Sample fundamental data
            "roe": 28.5,
            "roa": 15.2,
            "profit_margin": 22.5,
            "current_ratio": 1.8,
            "debt_to_equity": 0.65,
            "pe_ratio": 28.3,
            "pb_ratio": 8.5,
            "ps_ratio": 7.2,
            "revenue_growth_yoy": 12.0,
            "earnings_growth_yoy": 15.8,
            "eps_growth_yoy": 16.2,
            "current_price": current_price,
            "market_cap": "2.8T",
        },
        "sentiment": {
            "news_sentiment": agg_sentiment["score"],
            "news_count": len(sample_news),
            "sentiment_confidence": agg_sentiment["confidence"],
            "analyst_ratings": "12 Buy, 3 Hold, 1 Sell",
            "recent_headlines": sample_news,
        },
        "technical": {
            "current_price": current_price,
            "rsi": df_tech["rsi"].iloc[-1] if "rsi" in df_tech else 65.0,
            "macd": df_tech["macd"].iloc[-1] if "macd" in df_tech else 2.5,
            "macd_signal": df_tech["macd_signal"].iloc[-1] if "macd_signal" in df_tech else 2.0,
            "bb_upper": df_tech["bb_upper"].iloc[-1] if "bb_upper" in df_tech else current_price * 1.05,
            "bb_lower": df_tech["bb_lower"].iloc[-1] if "bb_lower" in df_tech else current_price * 0.95,
            "atr": df_tech["atr"].iloc[-1] if "atr" in df_tech else 3.5,
            "volume_ratio": df_tech["volume_ratio"].iloc[-1] if "volume_ratio" in df_tech else 1.1,
            "trend": "uptrend",
        },
        "risk": {
            "volatility": 0.25,
            "volatility_20d": df_tech["hvol_20"].iloc[-1] if "hvol_20" in df_tech else 0.22,
            "max_drawdown": 0.18,
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.4,
            "beta": 1.1,
            "var_95": -0.05,
        },
    }

    logger.info("✓ Multi-agent data prepared")

    # =========================================================================
    # Step 6: Retrieve RAG Context
    # =========================================================================
    logger.info("\n[Step 6] Retrieving Relevant Context from RAG")
    logger.info("-" * 80)

    # Query RAG for relevant context
    rag_query = f"What are {SYMBOL}'s recent financial performance, growth trends, and key risks?"
    rag_results = rag_system.retrieve(rag_query, top_k=5, symbol=SYMBOL)

    rag_context = "\n".join([r["text"] for r in rag_results])
    logger.info(f"✓ Retrieved {len(rag_results)} relevant documents")

    # =========================================================================
    # Step 7: Run Multi-Agent Analysis with Debate
    # =========================================================================
    logger.info("\n[Step 7] Running Multi-Agent Analysis")
    logger.info("=" * 80)

    logger.info("\nThis will:")
    logger.info("  1. Run 4 specialist analysts (Fundamental, Sentiment, Technical, Risk)")
    logger.info("  2. Facilitate 2 rounds of structured debate")
    logger.info("  3. Synthesize final trading decision")
    logger.info("")

    try:
        # Run multi-agent analysis
        final_decision = manager.analyze(
            symbol=SYMBOL,
            data=multi_agent_data,
            context=rag_context,
        )

        # =====================================================================
        # Step 8: Display Results
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info(f"MULTI-AGENT DECISION FOR {SYMBOL}")
        logger.info("=" * 80)

        print(f"\n{'='*80}")
        print(f"FINAL TRADING DECISION: {SYMBOL}")
        print(f"{'='*80}\n")

        print(f"Overall Score: {final_decision.score:.2f} (Range: -1=Strong Sell to 1=Strong Buy)")
        print(f"Confidence: {final_decision.confidence:.0%}")

        # Decode recommendation
        if final_decision.score > 0.5:
            recommendation = "STRONG BUY"
        elif final_decision.score > 0.2:
            recommendation = "BUY"
        elif final_decision.score > -0.2:
            recommendation = "HOLD"
        elif final_decision.score > -0.5:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"

        print(f"Recommendation: {recommendation}")

        print(f"\n{'='*80}")
        print("SPECIALIST ANALYST SCORES:")
        print(f"{'='*80}")

        specialist_scores = final_decision.data.get("specialist_scores", {})
        for analyst, score in specialist_scores.items():
            print(f"{analyst.upper():20s}: {score:+.2f}")

        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS:")
        print(f"{'='*80}\n")

        print(final_decision.analysis[:1000])
        if len(final_decision.analysis) > 1000:
            print("\n... (truncated)")

        print(f"\n{'='*80}")
        print("REASONING STEPS:")
        print(f"{'='*80}\n")

        for i, step in enumerate(final_decision.reasoning[:5], 1):
            print(f"{i}. {step}")

        # Usage stats
        print(f"\n{'='*80}")
        print("CLAUDE API USAGE:")
        print(f"{'='*80}")
        usage_stats = claude_client.get_usage_stats()
        print(f"Total Tokens: {usage_stats['total_tokens']:,}")
        print(f"Total Cost: ${usage_stats['total_cost']:.4f}")

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}")
        logger.error("This may be due to missing API keys or network issues.")
        logger.info("\nTo run this example:")
        logger.info("  1. Set ANTHROPIC_API_KEY in .env")
        logger.info("  2. Ensure Claude API access")
        logger.info("  3. Run: python scripts/example_multi_agent.py")


if __name__ == "__main__":
    main()
