"""
FastAPI application for RLAIF Trading Pipeline

Provides inference endpoints for:
- Market prediction
- Sentiment analysis
- Technical indicator computation
- Feature extraction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.features import TechnicalFeatureEngine, SentimentAnalyzer
from src.utils import setup_logging, get_settings

# Setup logging
logger = setup_logging(log_level="INFO", log_format="json")

# Initialize FastAPI app
app = FastAPI(
    title="RLAIF Trading API",
    description="AI-powered stock prediction with RLAIF",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================================
# Request/Response Models
# ===============================================================================


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    gpu_available: bool
    models_loaded: bool


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    time_series: List[float] = Field(
        ...,
        description="Historical price data",
        min_items=30,
        max_items=10000,
    )
    horizon: int = Field(
        default=30,
        description="Prediction horizon (days)",
        ge=1,
        le=365,
    )
    return_uncertainty: bool = Field(
        default=True,
        description="Whether to return uncertainty estimates",
    )


class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    uncertainty: Optional[List[float]] = None
    confidence: float
    timestamp: float
    model_used: str


class SentimentRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of text strings to analyze",
        min_items=1,
        max_items=100,
    )
    aggregate: bool = Field(
        default=True,
        description="Whether to aggregate results",
    )


class SentimentResponse(BaseModel):
    sentiments: List[Dict]
    aggregated: Optional[Dict] = None
    timestamp: float


class TechnicalIndicatorsRequest(BaseModel):
    ohlcv: Dict[str, List[float]] = Field(
        ...,
        description="OHLCV data with keys: open, high, low, close, volume",
    )
    indicators: Optional[List[str]] = Field(
        default=None,
        description="Specific indicators to compute (None = all)",
    )


class TechnicalIndicatorsResponse(BaseModel):
    indicators: Dict[str, List[float]]
    timestamp: float


# ===============================================================================
# Global State - Model Loading
# ===============================================================================


class ModelManager:
    """Manage model loading and caching"""

    def __init__(self):
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.technical_engine: Optional[TechnicalFeatureEngine] = None
        self.models_loaded = False

        logger.info("Initializing ModelManager")

    def load_models(self):
        """Load models on startup"""
        if self.models_loaded:
            return

        try:
            logger.info("Loading sentiment analyzer (FinBERT)...")
            self.sentiment_analyzer = SentimentAnalyzer(
                model_name="yiyanghkust/finbert-tone",
                batch_size=32,
            )

            logger.info("Loading technical indicator engine...")
            self.technical_engine = TechnicalFeatureEngine()

            self.models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def get_sentiment_analyzer(self) -> SentimentAnalyzer:
        """Get sentiment analyzer (lazy load if needed)"""
        if self.sentiment_analyzer is None:
            self.load_models()
        return self.sentiment_analyzer

    def get_technical_engine(self) -> TechnicalFeatureEngine:
        """Get technical indicator engine (lazy load if needed)"""
        if self.technical_engine is None:
            self.load_models()
        return self.technical_engine


# Initialize model manager
model_manager = ModelManager()


# ===============================================================================
# Startup/Shutdown Events
# ===============================================================================


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting RLAIF Trading API...")

    # Load models in background (don't block startup)
    try:
        model_manager.load_models()
    except Exception as e:
        logger.warning(f"Could not load all models on startup: {e}")

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down RLAIF Trading API...")


# ===============================================================================
# Middleware
# ===============================================================================


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ===============================================================================
# Health & Status Endpoints
# ===============================================================================


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "service": "RLAIF Trading API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        gpu_available=torch.cuda.is_available(),
        models_loaded=model_manager.models_loaded,
    )


@app.get("/status")
async def status():
    """Detailed status endpoint"""
    return {
        "status": "running",
        "timestamp": time.time(),
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            ),
        },
        "models": {
            "loaded": model_manager.models_loaded,
            "sentiment_analyzer": model_manager.sentiment_analyzer is not None,
            "technical_engine": model_manager.technical_engine is not None,
        },
    }


# ===============================================================================
# Prediction Endpoints
# ===============================================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate stock price predictions

    This endpoint uses foundation models (TimesFM/TTM) to forecast future prices.
    """
    try:
        logger.info(f"Prediction request for {request.symbol}")

        # Convert to numpy array
        time_series = np.array(request.time_series)

        # For now, return a simple moving average prediction
        # In production, this would use TimesFM or TTM
        logger.warning("Using placeholder prediction (foundation model not loaded)")

        # Simple MA-based prediction
        window = min(30, len(time_series))
        ma = np.mean(time_series[-window:])
        trend = (time_series[-1] - time_series[-window]) / window

        predictions = [ma + trend * i for i in range(1, request.horizon + 1)]
        uncertainty = [abs(trend) * i * 0.1 for i in range(1, request.horizon + 1)] if request.return_uncertainty else None

        # Calculate confidence (placeholder)
        volatility = np.std(time_series[-window:])
        confidence = 1.0 / (1.0 + volatility / ma)

        return PredictionResponse(
            symbol=request.symbol,
            predictions=predictions,
            uncertainty=uncertainty,
            confidence=float(confidence),
            timestamp=time.time(),
            model_used="moving_average_placeholder",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ===============================================================================
# Sentiment Analysis Endpoints
# ===============================================================================


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of financial texts using FinBERT

    Supports batch processing of up to 100 texts.
    """
    try:
        logger.info(f"Sentiment analysis for {len(request.texts)} texts")

        # Get sentiment analyzer
        analyzer = model_manager.get_sentiment_analyzer()

        # Analyze
        results = analyzer.analyze(request.texts)

        # Aggregate if requested
        aggregated = None
        if request.aggregate:
            aggregated = analyzer.aggregate_sentiments(results)

        return SentimentResponse(
            sentiments=results,
            aggregated=aggregated,
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


# ===============================================================================
# Technical Indicators Endpoints
# ===============================================================================


@app.post("/indicators", response_model=TechnicalIndicatorsResponse)
async def compute_indicators(request: TechnicalIndicatorsRequest):
    """
    Compute technical indicators from OHLCV data

    Returns 60+ technical indicators including RSI, MACD, Bollinger Bands, etc.
    """
    try:
        logger.info("Computing technical indicators")

        # Get technical engine
        engine = model_manager.get_technical_engine()

        # Convert to DataFrame
        import pandas as pd

        df = pd.DataFrame(request.ohlcv)

        # Compute indicators
        df_with_indicators = engine.compute_all(df)

        # Convert back to dict
        indicators = {}
        for col in df_with_indicators.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                # Convert to list, handling NaN
                values = df_with_indicators[col].fillna(0).tolist()
                indicators[col] = values

        return TechnicalIndicatorsResponse(
            indicators=indicators,
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Technical indicators error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Technical indicators computation failed: {str(e)}",
        )


# ===============================================================================
# Error Handlers
# ===============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": time.time(),
        },
    )


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
