"""
RunPod Serverless Handler for RLAIF Trading Pipeline

This handler wraps the FastAPI application for RunPod's serverless architecture.
Supports both HTTP and job-based invocations.

Improvements:
- Proper error handling and graceful degradation
- Input validation and size limits
- GPU memory management
- Request tracing with unique IDs
- Better health checks
"""

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, List

import numpy as np
import runpod
import torch

from src.features import TechnicalFeatureEngine, SentimentAnalyzer
from src.utils import setup_logging

# ===============================================================================
# Configuration Constants
# ===============================================================================

# Input validation limits
MAX_TIME_SERIES_LENGTH = 10000
MIN_TIME_SERIES_LENGTH = 30
MAX_TEXTS = 100
MAX_TEXT_LENGTH = 10000
MAX_HORIZON = 365
MIN_HORIZON = 1
MAX_OHLCV_LENGTH = 10000

# Request timeout (seconds)
REQUEST_TIMEOUT = 300  # 5 minutes, matches RunPod default

# ===============================================================================
# Setup Logging
# ===============================================================================

logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

# ===============================================================================
# Model Initialization (runs once on cold start)
# ===============================================================================

sentiment_analyzer: Optional[SentimentAnalyzer] = None
technical_engine: Optional[TechnicalFeatureEngine] = None
models_initialized = False


def initialize_models():
    """
    Initialize models with proper error handling and graceful degradation.
    
    This function can be called multiple times safely - it will only
    initialize once.
    """
    global sentiment_analyzer, technical_engine, models_initialized

    if models_initialized:
        return

    logger.info("Initializing models...")

    # Initialize sentiment analyzer
    try:
        sentiment_analyzer = SentimentAnalyzer(
            model_name="yiyanghkust/finbert-tone",
            batch_size=32,
        )
        logger.info("? FinBERT loaded successfully")
    except Exception as e:
        logger.error(f"? Could not load FinBERT: {e}", exc_info=True)
        sentiment_analyzer = None
        # Don't raise - allow graceful degradation

    # Initialize technical engine
    try:
        technical_engine = TechnicalFeatureEngine()
        logger.info("? Technical engine loaded successfully")
    except Exception as e:
        logger.error(f"? Could not load technical engine: {e}", exc_info=True)
        technical_engine = None
        # Don't raise - allow graceful degradation

    models_initialized = True
    logger.info(f"Model initialization complete. Sentiment: {sentiment_analyzer is not None}, Technical: {technical_engine is not None}")


def clear_gpu_cache():
    """Clear GPU cache to prevent memory leaks"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize models on module load
try:
    initialize_models()
except Exception as e:
    logger.critical(f"Critical error during model initialization: {e}", exc_info=True)
    # Continue anyway - handlers will fail gracefully


# ===============================================================================
# Input Validation Functions
# ===============================================================================


def validate_prediction_input(input_data: Dict[str, Any]) -> tuple[str, np.ndarray, int, bool]:
    """
    Validate and extract prediction input parameters.
    
    Returns:
        tuple: (symbol, time_series, horizon, return_uncertainty)
    
    Raises:
        ValueError: If validation fails
    """
    # Validate symbol
    symbol = input_data.get("symbol", "UNKNOWN")
    if not isinstance(symbol, str) or len(symbol) == 0:
        raise ValueError("Invalid symbol: must be a non-empty string")

    # Validate time_series
    if "time_series" not in input_data:
        raise ValueError("Missing required field: time_series")

    time_series = input_data["time_series"]
    if not isinstance(time_series, list):
        raise ValueError("time_series must be a list")

    if len(time_series) < MIN_TIME_SERIES_LENGTH:
        raise ValueError(f"time_series too short (min {MIN_TIME_SERIES_LENGTH}, got {len(time_series)})")

    if len(time_series) > MAX_TIME_SERIES_LENGTH:
        raise ValueError(f"time_series too long (max {MAX_TIME_SERIES_LENGTH}, got {len(time_series)})")

    try:
        time_series_array = np.array(time_series, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"time_series contains invalid values: {e}")

    if np.any(np.isnan(time_series_array)) or np.any(np.isinf(time_series_array)):
        raise ValueError("time_series contains NaN or Inf values")

    # Validate horizon
    horizon = input_data.get("horizon", 30)
    if not isinstance(horizon, int):
        raise ValueError("horizon must be an integer")
    if horizon < MIN_HORIZON or horizon > MAX_HORIZON:
        raise ValueError(f"horizon out of range ({MIN_HORIZON}-{MAX_HORIZON}, got {horizon})")

    # Validate return_uncertainty
    return_uncertainty = input_data.get("return_uncertainty", True)
    if not isinstance(return_uncertainty, bool):
        return_uncertainty = bool(return_uncertainty)

    return symbol, time_series_array, horizon, return_uncertainty


def validate_sentiment_input(input_data: Dict[str, Any]) -> tuple[list[str], bool]:
    """
    Validate and extract sentiment analysis input parameters.
    
    Returns:
        tuple: (texts, aggregate)
    
    Raises:
        ValueError: If validation fails
    """
    if "texts" not in input_data:
        raise ValueError("Missing required field: texts")

    texts = input_data["texts"]
    if not isinstance(texts, list):
        raise ValueError("texts must be a list")

    if len(texts) == 0:
        raise ValueError("texts list cannot be empty")

    if len(texts) > MAX_TEXTS:
        raise ValueError(f"Too many texts (max {MAX_TEXTS}, got {len(texts)})")

    # Validate each text
    validated_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"text[{i}] must be a string")
        if len(text) == 0:
            raise ValueError(f"text[{i}] cannot be empty")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"text[{i}] too long (max {MAX_TEXT_LENGTH} chars)")
        validated_texts.append(text)

    aggregate = input_data.get("aggregate", True)
    if not isinstance(aggregate, bool):
        aggregate = bool(aggregate)

    return validated_texts, aggregate


def validate_indicators_input(input_data: Dict[str, Any]) -> Dict[str, list[float]]:
    """
    Validate and extract technical indicators input parameters.
    
    Returns:
        dict: OHLCV data
    
    Raises:
        ValueError: If validation fails
    """
    if "ohlcv" not in input_data:
        raise ValueError("Missing required field: ohlcv")

    ohlcv = input_data["ohlcv"]
    if not isinstance(ohlcv, dict):
        raise ValueError("ohlcv must be a dictionary")

    required_keys = ["open", "high", "low", "close", "volume"]
    for key in required_keys:
        if key not in ohlcv:
            raise ValueError(f"Missing required OHLCV key: {key}")

    # Validate lengths and convert to lists
    validated_ohlcv = {}
    lengths = []
    for key in required_keys:
        values = ohlcv[key]
        if not isinstance(values, list):
            raise ValueError(f"ohlcv['{key}'] must be a list")
        if len(values) == 0:
            raise ValueError(f"ohlcv['{key}'] cannot be empty")
        if len(values) > MAX_OHLCV_LENGTH:
            raise ValueError(f"ohlcv['{key}'] too long (max {MAX_OHLCV_LENGTH})")
        
        # Validate numeric values
        try:
            validated_values = [float(v) for v in values]
            if any(np.isnan(v) or np.isinf(v) for v in validated_values):
                raise ValueError(f"ohlcv['{key}'] contains NaN or Inf values")
            validated_ohlcv[key] = validated_values
            lengths.append(len(validated_values))
        except (ValueError, TypeError) as e:
            raise ValueError(f"ohlcv['{key}'] contains invalid values: {e}")

    # All arrays must have the same length
    if len(set(lengths)) != 1:
        raise ValueError(f"OHLCV arrays have different lengths: {lengths}")

    return validated_ohlcv


# ===============================================================================
# Handler Functions
# ===============================================================================


def predict_handler(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Handle prediction requests with input validation and error handling.
    
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
        clear_gpu_cache()  # Clear cache before processing

        input_data = job.get("input", {})
        symbol, time_series, horizon, return_uncertainty = validate_prediction_input(input_data)

        logger.info(f"[{request_id}] Prediction request: symbol={symbol}, horizon={horizon}, series_length={len(time_series)}")

        # Simple MA-based prediction (placeholder for foundation model)
        window = min(30, len(time_series))
        ma = np.mean(time_series[-window:])
        trend = (time_series[-1] - time_series[-window]) / window

        predictions = [float(ma + trend * i) for i in range(1, horizon + 1)]

        result = {
            "symbol": symbol,
            "predictions": predictions,
            "model_used": "moving_average_placeholder",
            "timestamp": time.time(),
        }

        if return_uncertainty:
            uncertainty = [float(abs(trend) * i * 0.1) for i in range(1, horizon + 1)]
            result["uncertainty"] = uncertainty

        return {"status": "success", "output": result, "request_id": request_id}

    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {e}")
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "validation"}
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "runtime"}
    finally:
        clear_gpu_cache()  # Clear cache after processing


def sentiment_handler(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Handle sentiment analysis requests with input validation.
    
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
        clear_gpu_cache()  # Clear cache before processing

        if sentiment_analyzer is None:
            return {
                "status": "error",
                "error": "Sentiment analyzer not available (failed to load during initialization)",
                "request_id": request_id,
                "error_type": "service_unavailable",
            }

        input_data = job.get("input", {})
        texts, aggregate = validate_sentiment_input(input_data)

        logger.info(f"[{request_id}] Sentiment analysis: {len(texts)} texts, aggregate={aggregate}")

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
            "request_id": request_id,
        }

    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {e}")
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "validation"}
    except Exception as e:
        logger.error(f"[{request_id}] Sentiment analysis error: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "runtime"}
    finally:
        clear_gpu_cache()  # Clear cache after processing


def indicators_handler(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Handle technical indicators computation with input validation.
    
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
        clear_gpu_cache()  # Clear cache before processing

        if technical_engine is None:
            return {
                "status": "error",
                "error": "Technical engine not available (failed to load during initialization)",
                "request_id": request_id,
                "error_type": "service_unavailable",
            }

        input_data = job.get("input", {})
        ohlcv = validate_indicators_input(input_data)

        logger.info(f"[{request_id}] Computing technical indicators: {len(ohlcv['close'])} data points")

        # Convert to DataFrame
        import pandas as pd

        df = pd.DataFrame(ohlcv)

        # Compute indicators
        df_with_indicators = technical_engine.compute_all(df)

        # Convert back to dict
        indicators = {}
        for col in df_with_indicators.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                # Convert to list, handling NaN
                values = df_with_indicators[col].fillna(0).tolist()
                indicators[col] = [float(v) for v in values]

        return {
            "status": "success",
            "output": {
                "indicators": indicators,
                "timestamp": time.time(),
            },
            "request_id": request_id,
        }

    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {e}")
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "validation"}
    except Exception as e:
        logger.error(f"[{request_id}] Technical indicators error: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "runtime"}
    finally:
        clear_gpu_cache()  # Clear cache after processing


def health_handler(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Enhanced health check endpoint.
    
    Verifies:
    - System status
    - GPU availability
    - Model loading status
    - Memory status
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "gpu": {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
            "models": {
                "sentiment_analyzer": sentiment_analyzer is not None,
                "technical_engine": technical_engine is not None,
                "initialized": models_initialized,
            },
            "request_id": request_id,
        }

        # Add GPU device name if available
        if torch.cuda.is_available():
            try:
                health_status["gpu"]["device_name"] = torch.cuda.get_device_name(0)
                health_status["gpu"]["memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
                health_status["gpu"]["memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024**2
            except Exception as e:
                logger.warning(f"[{request_id}] Could not get GPU details: {e}")
                health_status["gpu"]["error"] = str(e)

        # Test model functionality if loaded
        if sentiment_analyzer is not None:
            try:
                # Quick test inference
                test_result = sentiment_analyzer.analyze(["test"])
                health_status["models"]["sentiment_test"] = "passed"
            except Exception as e:
                health_status["models"]["sentiment_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"

        if technical_engine is not None:
            try:
                # Quick test computation
                import pandas as pd
                test_df = pd.DataFrame({
                    "open": [100, 101],
                    "high": [102, 103],
                    "low": [99, 100],
                    "close": [101, 102],
                    "volume": [1000, 1100],
                })
                test_result = technical_engine.compute_all(test_df)
                health_status["models"]["technical_test"] = "passed"
            except Exception as e:
                health_status["models"]["technical_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"

        # Overall health determination
        if not health_status["models"]["sentiment_analyzer"] and not health_status["models"]["technical_engine"]:
            health_status["status"] = "unhealthy"

        return {"status": "success", "output": health_status, "request_id": request_id}

    except Exception as e:
        logger.error(f"[{request_id}] Health check error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "request_id": request_id,
            "error_type": "health_check_failed",
        }


# ===============================================================================
# Unified Analysis Handler
# ===============================================================================


def _fetch_news_for_symbol(symbol: str) -> List[str]:
    """Fetch news for a symbol (placeholder - implement with Finnhub/Polygon)"""
    # TODO: Implement news fetching with Finnhub/Polygon API
    # For now, return empty list
    return []


def _generate_summary(result: Dict[str, Any]) -> str:
    """Generate human-readable summary"""
    symbol = result.get("symbol", "Unknown")
    summary_parts = [f"Analysis for {symbol}:"]
    
    if "prediction" in result and "error" not in result["prediction"]:
        pred = result["prediction"]
        if "predictions" in pred and len(pred["predictions"]) > 0:
            last_price = pred["predictions"][-1] if pred["predictions"] else None
            if last_price:
                summary_parts.append(f"Predicted price: ${last_price:.2f}")
    
    if "recommendation" in result and "error" not in result["recommendation"]:
        rec = result["recommendation"]
        if "signal" in rec:
            confidence = rec.get("confidence", 0) * 100
            summary_parts.append(f"Recommendation: {rec['signal']} (confidence: {confidence:.0f}%)")
    
    return " ".join(summary_parts)


def analyze_handler(job: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Unified analysis endpoint - analyzes a ticker completely.
    
    Expected input:
    {
        "input": {
            "action": "analyze",
            "symbol": "AAPL",
            "horizon": 30,
            "period": "1y",  # Optional, default 1y
            "include_sentiment": true,  # Optional
            "include_indicators": true,  # Optional
            "include_prediction": true  # Optional
        }
    }
    """
    try:
        clear_gpu_cache()
        
        input_data = job.get("input", {})
        symbol = input_data.get("symbol", "").upper().strip()
        horizon = input_data.get("horizon", 30)
        period = input_data.get("period", "1y")
        include_sentiment = input_data.get("include_sentiment", True)
        include_indicators = input_data.get("include_indicators", True)
        include_prediction = input_data.get("include_prediction", True)
        
        if not symbol:
            raise ValueError("symbol is required")
        
        logger.info(f"[{request_id}] Analyzing {symbol} (horizon={horizon}, period={period})")
        
        # Fetch data automatically
        try:
            import pandas as pd
            
            # Try Alpaca first, fallback to yfinance
            df = None
            try:
                from src.data.ingestion.market_data import AlpacaDataClient
                from dotenv import load_dotenv
                
                load_dotenv()
                data_client = AlpacaDataClient()
                
                # Convert period to days
                days_map = {"1y": 365, "6mo": 180, "3mo": 90, "1mo": 30}
                days = days_map.get(period, 365)
                
                df = data_client.download_latest(
                    symbols=symbol,
                    days=days,
                    timeframe="1Day"
                )
                
                # Convert multi-index to single index if needed
                if isinstance(df.index, pd.MultiIndex):
                    df = df.loc[symbol].reset_index()
                    df = df.set_index("timestamp")
                else:
                    df = df.reset_index()
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                        
            except Exception as e:
                logger.warning(f"[{request_id}] Alpaca failed, using yfinance: {e}")
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period)
                    df.index.name = "timestamp"
                    # Rename columns to lowercase
                    df.columns = df.columns.str.lower()
                except Exception as yf_error:
                    logger.error(f"[{request_id}] yfinance also failed: {yf_error}")
                    raise ValueError(f"Could not fetch data for {symbol}: {str(yf_error)}")
                    
        except Exception as e:
            logger.error(f"[{request_id}] Data fetching failed: {e}")
            raise ValueError(f"Could not fetch data for {symbol}: {e}")
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        result = {
            "symbol": symbol,
            "timestamp": time.time(),
            "data_period": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "days": len(df)
            }
        }
        
        # Prediction
        if include_prediction:
            try:
                # Get close prices
                close_col = "close" if "close" in df.columns else df.columns[0]
                prices = df[close_col].values.tolist()
                
                if len(prices) < MIN_TIME_SERIES_LENGTH:
                    logger.warning(f"[{request_id}] Not enough data for prediction: {len(prices)} < {MIN_TIME_SERIES_LENGTH}")
                    result["prediction"] = {"error": f"Not enough data: {len(prices)} < {MIN_TIME_SERIES_LENGTH}"}
                else:
                    pred_result = predict_handler({
                        "input": {
                            "symbol": symbol,
                            "time_series": prices,
                            "horizon": horizon,
                            "return_uncertainty": True
                        }
                    }, request_id)
                    
                    if pred_result.get("status") == "success":
                        result["prediction"] = pred_result["output"]
                    else:
                        result["prediction"] = {"error": pred_result.get("error", "Unknown error")}
            except Exception as e:
                logger.warning(f"[{request_id}] Prediction failed: {e}")
                result["prediction"] = {"error": str(e)}
        
        # Indicators
        if include_indicators:
            try:
                # Ensure we have OHLCV columns
                ohlcv_cols = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
                ohlcv_data = {}
                
                for key, col_name in ohlcv_cols.items():
                    if col_name in df.columns:
                        ohlcv_data[key] = df[col_name].values.tolist()
                    else:
                        # Fallback: use close price for missing columns
                        logger.warning(f"[{request_id}] Missing {col_name} column, using close price")
                        close_col = "close" if "close" in df.columns else df.columns[0]
                        ohlcv_data[key] = df[close_col].values.tolist()
                
                if len(ohlcv_data["close"]) == 0:
                    raise ValueError("No OHLCV data available")
                
                indicators_result = indicators_handler({
                    "input": {
                        "action": "indicators",
                        "ohlcv": ohlcv_data
                    }
                }, request_id)
                
                if indicators_result.get("status") == "success":
                    result["indicators"] = indicators_result["output"]
                else:
                    result["indicators"] = {"error": indicators_result.get("error", "Unknown error")}
            except Exception as e:
                logger.warning(f"[{request_id}] Indicators failed: {e}")
                result["indicators"] = {"error": str(e)}
        
        # Sentiment (requires news fetching - optional)
        if include_sentiment:
            try:
                # Try to fetch news
                news_texts = _fetch_news_for_symbol(symbol)
                if news_texts:
                    sentiment_result = sentiment_handler({
                        "input": {
                            "action": "sentiment",
                            "texts": news_texts,
                            "aggregate": True
                        }
                    }, request_id)
                    
                    if sentiment_result.get("status") == "success":
                        result["sentiment"] = sentiment_result["output"]
                    else:
                        result["sentiment"] = {"error": sentiment_result.get("error", "Unknown error")}
                else:
                    result["sentiment"] = {"note": "No news data available (news fetching not implemented)"}
            except Exception as e:
                logger.warning(f"[{request_id}] Sentiment failed: {e}")
                result["sentiment"] = {"error": str(e)}
        
        # Generate recommendation
        try:
            from src.analysis.recommendation import generate_recommendation
            recommendation = generate_recommendation(result)
            result["recommendation"] = recommendation
        except Exception as e:
            logger.warning(f"[{request_id}] Recommendation generation failed: {e}")
            result["recommendation"] = {"error": str(e)}
        
        # Generate summary
        result["summary"] = _generate_summary(result)
        
        return {
            "status": "success",
            "output": result,
            "request_id": request_id
        }
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {e}")
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "validation"}
    except Exception as e:
        logger.error(f"[{request_id}] Analysis error: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "request_id": request_id, "error_type": "runtime"}
    finally:
        clear_gpu_cache()


# ===============================================================================
# Main Handler (routes to specific handlers)
# ===============================================================================


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler with request tracing and error handling.
    
    Routes requests to appropriate sub-handlers based on 'action' field.
    
    Supported actions:
    - analyze: Unified ticker analysis (NEW - recommended)
    - predict: Stock price prediction
    - sentiment: Sentiment analysis
    - indicators: Technical indicators
    - health: Health check
    
    Input format:
    {
        "input": {
            "action": "analyze|predict|sentiment|indicators|health",
            ... (action-specific parameters)
        }
    }
    """
    # Generate unique request ID for tracing
    request_id = str(uuid.uuid4())

    try:
        # Ensure models are initialized
        if not models_initialized:
            logger.info(f"[{request_id}] Models not initialized, attempting initialization...")
            initialize_models()

        # Extract action
        input_data = job.get("input", {})
        action = input_data.get("action", "health")

        logger.info(f"[{request_id}] Received request: action={action}")

        # Route to appropriate handler
        if action == "analyze":
            return analyze_handler(job, request_id)
        elif action == "predict":
            return predict_handler(job, request_id)
        elif action == "sentiment":
            return sentiment_handler(job, request_id)
        elif action == "indicators":
            return indicators_handler(job, request_id)
        elif action == "health":
            return health_handler(job, request_id)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "supported_actions": ["analyze", "predict", "sentiment", "indicators", "health"],
                "request_id": request_id,
                "error_type": "invalid_action",
            }

    except KeyboardInterrupt:
        logger.warning(f"[{request_id}] Request interrupted")
        raise  # Re-raise to allow RunPod to handle
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled handler error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "Internal server error",
            "detail": str(e),
            "request_id": request_id,
            "error_type": "internal_error",
        }
    finally:
        # Always clear GPU cache after request
        clear_gpu_cache()


# ===============================================================================
# RunPod Startup
# ===============================================================================

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    logger.info(f"GPU available: {torch.cuda.is_available()}")

    # Ensure models are initialized before starting
    try:
        initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize models: {e}", exc_info=True)
        logger.critical("Starting handler anyway - requests may fail gracefully")

    # Start RunPod serverless worker
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

    logger.info("RunPod handler started successfully")
