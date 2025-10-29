"""
RunPod Serverless Handler for RLAIF Trading Pipeline

This handler wraps the FastAPI application for RunPod's serverless architecture.
Supports both HTTP and job-based invocations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import os
import time
from typing import Any, Dict

import numpy as np
import runpod

from src.features import TechnicalFeatureEngine, SentimentAnalyzer
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

# ===============================================================================
# Model Initialization (runs once on cold start)
# ===============================================================================

logger.info("Initializing models...")

# Initialize models
try:
    sentiment_analyzer = SentimentAnalyzer(
        model_name="yiyanghkust/finbert-tone",
        batch_size=32,
    )
    logger.info("FinBERT loaded successfully")
except Exception as e:
    logger.warning(f"Could not load FinBERT: {e}")
    sentiment_analyzer = None

try:
    technical_engine = TechnicalFeatureEngine()
    logger.info("Technical engine loaded successfully")
except Exception as e:
    logger.warning(f"Could not load technical engine: {e}")
    technical_engine = None

logger.info("Model initialization complete")


# ===============================================================================
# Handler Functions
# ===============================================================================


def predict_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle prediction requests

    Expected input:
    {
        "input": {
            "action": "predict",
            "symbol": "AAPL",
            "time_series": [100.0, 101.0, ...],
            "horizon": 30,
            "return_uncertainty": true
        }
    }
    """
    try:
        input_data = job["input"]
        symbol = input_data.get("symbol", "UNKNOWN")
        time_series = np.array(input_data["time_series"])
        horizon = input_data.get("horizon", 30)
        return_uncertainty = input_data.get("return_uncertainty", True)

        logger.info(f"Prediction request for {symbol}, horizon={horizon}")

        # Simple MA-based prediction (placeholder for foundation model)
        window = min(30, len(time_series))
        ma = np.mean(time_series[-window:])
        trend = (time_series[-1] - time_series[-window]) / window

        predictions = [ma + trend * i for i in range(1, horizon + 1)]

        result = {
            "symbol": symbol,
            "predictions": predictions,
            "model_used": "moving_average_placeholder",
            "timestamp": time.time(),
        }

        if return_uncertainty:
            uncertainty = [abs(trend) * i * 0.1 for i in range(1, horizon + 1)]
            result["uncertainty"] = uncertainty

        return {"status": "success", "output": result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"status": "error", "error": str(e)}


def sentiment_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle sentiment analysis requests

    Expected input:
    {
        "input": {
            "action": "sentiment",
            "texts": ["text1", "text2", ...],
            "aggregate": true
        }
    }
    """
    try:
        if sentiment_analyzer is None:
            return {
                "status": "error",
                "error": "Sentiment analyzer not loaded",
            }

        input_data = job["input"]
        texts = input_data["texts"]
        aggregate = input_data.get("aggregate", True)

        logger.info(f"Sentiment analysis for {len(texts)} texts")

        # Analyze
        results = sentiment_analyzer.analyze(texts)

        # Aggregate if requested
        aggregated = None
        if aggregate:
            aggregated = sentiment_analyzer.aggregate_sentiments(results)

        return {
            "status": "success",
            "output": {
                "sentiments": results,
                "aggregated": aggregated,
                "timestamp": time.time(),
            },
        }

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"status": "error", "error": str(e)}


def indicators_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle technical indicators computation

    Expected input:
    {
        "input": {
            "action": "indicators",
            "ohlcv": {
                "open": [...],
                "high": [...],
                "low": [...],
                "close": [...],
                "volume": [...]
            }
        }
    }
    """
    try:
        if technical_engine is None:
            return {
                "status": "error",
                "error": "Technical engine not loaded",
            }

        input_data = job["input"]
        ohlcv = input_data["ohlcv"]

        logger.info("Computing technical indicators")

        # Convert to DataFrame
        import pandas as pd

        df = pd.DataFrame(ohlcv)

        # Compute indicators
        df_with_indicators = technical_engine.compute_all(df)

        # Convert back to dict
        indicators = {}
        for col in df_with_indicators.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                values = df_with_indicators[col].fillna(0).tolist()
                indicators[col] = values

        return {
            "status": "success",
            "output": {
                "indicators": indicators,
                "timestamp": time.time(),
            },
        }

    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        return {"status": "error", "error": str(e)}


def health_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Handle health check requests"""
    import torch

    return {
        "status": "success",
        "output": {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": {
                "sentiment_analyzer": sentiment_analyzer is not None,
                "technical_engine": technical_engine is not None,
            },
            "timestamp": time.time(),
        },
    }


# ===============================================================================
# Main Handler (routes to specific handlers)
# ===============================================================================


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler

    Routes requests to appropriate sub-handlers based on 'action' field.

    Supported actions:
    - predict: Stock price prediction
    - sentiment: Sentiment analysis
    - indicators: Technical indicators
    - health: Health check

    Input format:
    {
        "input": {
            "action": "predict|sentiment|indicators|health",
            ... (action-specific parameters)
        }
    }
    """
    try:
        # Extract action
        input_data = job.get("input", {})
        action = input_data.get("action", "health")

        logger.info(f"Received request: action={action}")

        # Route to appropriate handler
        if action == "predict":
            return predict_handler(job)
        elif action == "sentiment":
            return sentiment_handler(job)
        elif action == "indicators":
            return indicators_handler(job)
        elif action == "health":
            return health_handler(job)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "supported_actions": ["predict", "sentiment", "indicators", "health"],
            }

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# ===============================================================================
# RunPod Startup
# ===============================================================================

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")

    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})

    logger.info("RunPod handler started successfully")
