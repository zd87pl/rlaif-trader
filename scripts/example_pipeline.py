#!/usr/bin/env python3
"""
Example: Complete RLAIF Trading Pipeline

This script demonstrates the full pipeline:
1. Data ingestion (Alpaca API)
2. Preprocessing and feature engineering
3. Foundation model predictions (TimesFM/TTM)
4. Feature combination for RL
5. Basic analysis and visualization

For full RLAIF loop with Claude agents, see scripts/run_rlaif.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data import AlpacaDataClient, DataPreprocessor
from src.features import TechnicalFeatureEngine, SentimentAnalyzer, FundamentalAnalyzer
from src.models import TimesFMPredictor, TTMPredictor
from src.utils import setup_logging, set_seed

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logging(log_level="INFO")


def main():
    """Run example pipeline"""
    logger.info("=" * 80)
    logger.info("RLAIF Trading Pipeline - Example Run")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    set_seed(42)

    # Configuration
    SYMBOL = "AAPL"
    DAYS = 365
    TIMEFRAME = "1Day"

    # =========================================================================
    # Step 1: Data Ingestion
    # =========================================================================
    logger.info("\n[Step 1] Data Ingestion")
    logger.info("-" * 80)

    client = AlpacaDataClient()

    df_raw = client.download_latest(
        symbols=SYMBOL,
        days=DAYS,
        timeframe=TIMEFRAME,
        use_cache=True,
    )

    logger.info(f"Downloaded {len(df_raw)} bars for {SYMBOL}")
    logger.info(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

    # =========================================================================
    # Step 2: Data Preprocessing
    # =========================================================================
    logger.info("\n[Step 2] Data Preprocessing")
    logger.info("-" * 80)

    preprocessor = DataPreprocessor(
        fill_method="forward",
        outlier_threshold=5.0,
    )

    df_processed = preprocessor.preprocess(df_raw, symbol=SYMBOL)

    logger.info(f"Processed data shape: {df_processed.shape}")
    logger.info(f"Columns: {df_processed.columns.tolist()}")

    # =========================================================================
    # Step 3: Feature Engineering
    # =========================================================================
    logger.info("\n[Step 3] Feature Engineering")
    logger.info("-" * 80)

    # Technical indicators
    logger.info("Computing technical indicators...")
    tech_engine = TechnicalFeatureEngine()
    df_with_tech = tech_engine.compute_all(df_processed)

    logger.info(f"Added {len(tech_engine.get_feature_names())} technical indicators")

    # Display sample features
    sample_features = [
        "close",
        "rsi",
        "macd",
        "bb_upper",
        "bb_lower",
        "atr",
        "volume_ratio",
    ]
    logger.info("\nSample features (last 5 rows):")
    logger.info("\n" + str(df_with_tech[sample_features].tail()))

    # Sentiment analysis (placeholder - would need actual news data)
    logger.info("\nSentiment analysis (example)...")
    sentiment_analyzer = SentimentAnalyzer()

    # Example news texts
    example_news = [
        f"{SYMBOL} reports strong quarterly earnings, beating expectations.",
        f"Analysts upgrade {SYMBOL} to buy rating on growth prospects.",
        f"{SYMBOL} faces regulatory scrutiny over new product launch.",
    ]

    sentiments = sentiment_analyzer.analyze(example_news)
    agg_sentiment = sentiment_analyzer.aggregate_sentiments(sentiments)

    logger.info(f"Aggregated sentiment: {agg_sentiment}")

    # =========================================================================
    # Step 4: Foundation Model Predictions
    # =========================================================================
    logger.info("\n[Step 4] Foundation Model Predictions")
    logger.info("-" * 80)

    # Prepare time series (just close prices for simplicity)
    close_prices = df_with_tech["close"].values

    # TimesFM predictions
    logger.info("Loading TimesFM model...")
    try:
        timesfm_model = TimesFMPredictor(
            context_length=min(64, len(close_prices) // 2),
            horizon=30,
        )

        logger.info("Generating TimesFM predictions...")
        timesfm_preds, timesfm_uncertainty = timesfm_model.predict(
            close_prices[-128:],  # Use last 128 points
            horizon=30,
            return_uncertainty=True,
        )

        logger.info(f"TimesFM predictions shape: {timesfm_preds.shape}")
        logger.info(f"Mean prediction: ${timesfm_preds.mean():.2f}")
        logger.info(f"Mean uncertainty: ${timesfm_uncertainty.mean():.2f}")

    except Exception as e:
        logger.warning(f"TimesFM prediction failed: {e}")
        logger.info("Skipping TimesFM (may need to install: pip install timesfm[torch])")
        timesfm_preds = None

    # TTM predictions
    logger.info("\nLoading TTM model...")
    try:
        ttm_model = TTMPredictor(
            context_length=min(64, len(close_prices) // 2),
            patch_size=16,
        )

        logger.info("Generating TTM predictions...")
        ttm_preds, ttm_uncertainty = ttm_model.predict(
            close_prices[-128:],
            horizon=30,
            return_uncertainty=True,
        )

        logger.info(f"TTM predictions shape: {ttm_preds.shape}")
        logger.info(f"Mean prediction: ${ttm_preds.mean():.2f}")
        logger.info(f"Mean uncertainty: ${ttm_uncertainty.mean():.2f}")

    except Exception as e:
        logger.warning(f"TTM prediction failed: {e}")
        logger.info("Skipping TTM (may need correct model configuration)")
        ttm_preds = None

    # =========================================================================
    # Step 5: Feature Combination for RL
    # =========================================================================
    logger.info("\n[Step 5] Feature Combination")
    logger.info("-" * 80)

    # Combine all features for RL state space
    rl_features = []

    # Price features (normalized)
    price_norm = (df_with_tech["close"] - df_with_tech["close"].mean()) / df_with_tech[
        "close"
    ].std()
    rl_features.append(price_norm)

    # Technical indicators (sample)
    for feat in ["rsi", "macd", "atr", "volume_ratio"]:
        if feat in df_with_tech.columns:
            feat_norm = (df_with_tech[feat] - df_with_tech[feat].mean()) / (
                df_with_tech[feat].std() + 1e-9
            )
            rl_features.append(feat_norm)

    # Combine
    rl_state = pd.concat(rl_features, axis=1)
    rl_state = rl_state.fillna(0)

    logger.info(f"RL state space shape: {rl_state.shape}")
    logger.info(f"RL features: {rl_state.columns.tolist()}")

    # =========================================================================
    # Step 6: Summary Statistics
    # =========================================================================
    logger.info("\n[Step 6] Summary Statistics")
    logger.info("-" * 80)

    # Price statistics
    returns = df_with_tech["returns"].dropna()

    logger.info(f"\nPrice Statistics for {SYMBOL}:")
    logger.info(f"  Current Price: ${df_with_tech['close'].iloc[-1]:.2f}")
    logger.info(f"  Price Change: {(df_with_tech['close'].iloc[-1] / df_with_tech['close'].iloc[0] - 1) * 100:.2f}%")
    logger.info(f"  Mean Return: {returns.mean() * 100:.4f}%")
    logger.info(f"  Volatility (daily): {returns.std() * 100:.4f}%")
    logger.info(f"  Volatility (annual): {returns.std() * np.sqrt(252) * 100:.2f}%")
    logger.info(f"  Sharpe Ratio (est): {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")

    # Technical indicators
    logger.info(f"\nTechnical Indicators:")
    logger.info(f"  RSI: {df_with_tech['rsi'].iloc[-1]:.2f}")
    logger.info(f"  MACD: {df_with_tech['macd'].iloc[-1]:.4f}")
    logger.info(f"  ATR: ${df_with_tech['atr'].iloc[-1]:.2f}")

    # Sentiment
    logger.info(f"\nSentiment (example):")
    logger.info(f"  Score: {agg_sentiment['score']:.2f}")
    logger.info(f"  Label: {agg_sentiment['sentiment']}")

    # =========================================================================
    # Conclusion
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Completed Successfully!")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("  1. Fine-tune foundation models on financial data")
    logger.info("  2. Implement multi-agent Claude LLM system")
    logger.info("  3. Train RL agents with augmented state space")
    logger.info("  4. Build RLAIF feedback loop")
    logger.info("  5. Run backtesting with walk-forward validation")
    logger.info("\nSee scripts/run_rlaif.py for full RLAIF pipeline")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
