# Implementation Plan: Trader-Friendly API

## Overview

This document outlines the specific code changes needed to make the system trader-friendly.

---

## Implementation Priority

### 🔥 Priority 1: Unified Analysis Endpoint

**Goal**: Single endpoint that takes a ticker symbol and returns complete analysis.

**Files to Modify**:
1. `deployment/runpod/handler.py` - Add `analyze_handler()`
2. `src/data/ingestion/market_data.py` - Ensure usable in handler
3. `src/analysis/recommendation.py` - NEW FILE - Recommendation logic

---

## Code Implementation

### 1. Unified Analyze Handler

**File**: `deployment/runpod/handler.py`

**Add this handler**:

```python
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
        symbol = input_data.get("symbol", "").upper()
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
            from src.data.ingestion.market_data import AlpacaDataClient
            from dotenv import load_dotenv
            import os
            
            load_dotenv()
            
            # Try Alpaca first, fallback to yfinance
            try:
                data_client = AlpacaDataClient()
                df = data_client.download_latest(
                    symbols=symbol,
                    days=365 if period == "1y" else 30,
                    timeframe="1Day"
                )
                
                # Convert multi-index to single index if needed
                if isinstance(df.index, pd.MultiIndex):
                    df = df.loc[symbol].reset_index()
                    df = df.set_index("timestamp")
                else:
                    df = df.reset_index()
                    df = df.set_index("timestamp")
                    
            except Exception as e:
                logger.warning(f"[{request_id}] Alpaca failed, using yfinance: {e}")
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                df.index.name = "timestamp"
                
        except Exception as e:
            logger.error(f"[{request_id}] Data fetching failed: {e}")
            raise ValueError(f"Could not fetch data for {symbol}: {e}")
        
        if df.empty:
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
                prices = df['close'].values.tolist()
                if len(prices) < MIN_TIME_SERIES_LENGTH:
                    raise ValueError(f"Not enough data: {len(prices)} < {MIN_TIME_SERIES_LENGTH}")
                
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
            except Exception as e:
                logger.warning(f"[{request_id}] Prediction failed: {e}")
                result["prediction"] = {"error": str(e)}
        
        # Indicators
        if include_indicators:
            try:
                ohlcv = {
                    "open": df['open'].values.tolist(),
                    "high": df['high'].values.tolist(),
                    "low": df['low'].values.tolist(),
                    "close": df['close'].values.tolist(),
                    "volume": df['volume'].values.tolist()
                }
                
                indicators_result = indicators_handler({
                    "input": {
                        "action": "indicators",
                        "ohlcv": ohlcv
                    }
                }, request_id)
                
                if indicators_result.get("status") == "success":
                    result["indicators"] = indicators_result["output"]
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
                    result["sentiment"] = {"note": "No news data available"}
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


def _fetch_news_for_symbol(symbol: str) -> List[str]:
    """Fetch news for a symbol (placeholder - implement with Finnhub/Polygon)"""
    # TODO: Implement news fetching
    # For now, return empty list
    return []


def _generate_summary(result: Dict[str, Any]) -> str:
    """Generate human-readable summary"""
    symbol = result.get("symbol", "Unknown")
    summary_parts = [f"Analysis for {symbol}:"]
    
    if "prediction" in result and "error" not in result["prediction"]:
        pred = result["prediction"]
        if "predictions" in pred:
            last_price = pred["predictions"][-1] if pred["predictions"] else "N/A"
            summary_parts.append(f"Predicted price: ${last_price:.2f}")
    
    if "recommendation" in result and "error" not in result["recommendation"]:
        rec = result["recommendation"]
        if "signal" in rec:
            summary_parts.append(f"Recommendation: {rec['signal']} (confidence: {rec.get('confidence', 0):.0%})")
    
    return " ".join(summary_parts)
```

**Update main handler**:

```python
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing code ...
    
    # Route to appropriate handler
    if action == "predict":
        return predict_handler(job, request_id)
    elif action == "sentiment":
        return sentiment_handler(job, request_id)
    elif action == "indicators":
        return indicators_handler(job, request_id)
    elif action == "analyze":  # NEW
        return analyze_handler(job, request_id)
    elif action == "health":
        return health_handler(job, request_id)
    else:
        # ... existing error handling ...
```

---

### 2. Recommendation Engine

**File**: `src/analysis/recommendation.py` (NEW FILE)

```python
"""
Trading recommendation engine.

Combines predictions, indicators, and sentiment to generate
BUY/SELL/HOLD signals with confidence scores.
"""

from typing import Dict, Any, Optional
import numpy as np


def generate_recommendation(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate trading recommendation from analysis results.
    
    Args:
        analysis_result: Result from analyze_handler containing:
            - prediction: Price predictions
            - indicators: Technical indicators
            - sentiment: Sentiment analysis
    
    Returns:
        {
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": 0.0-1.0,
            "reasoning": "Explanation...",
            "target_price": float,
            "stop_loss": float,
            "risk_score": 0.0-1.0
        }
    """
    signals = []
    weights = []
    reasoning_parts = []
    
    # 1. Prediction-based signal
    if "prediction" in analysis_result and "error" not in analysis_result["prediction"]:
        pred = analysis_result["prediction"]
        if "predictions" in pred and len(pred["predictions"]) > 0:
            predictions = pred["predictions"]
            current_price = predictions[0] if len(predictions) > 0 else None
            
            if current_price and len(predictions) > 1:
                # Compare first prediction to last
                price_change_pct = (predictions[-1] - current_price) / current_price * 100
                
                if price_change_pct > 5:
                    signals.append("BUY")
                    weights.append(0.4)
                    reasoning_parts.append(f"Predicted {price_change_pct:.1f}% price increase")
                elif price_change_pct < -5:
                    signals.append("SELL")
                    weights.append(0.4)
                    reasoning_parts.append(f"Predicted {price_change_pct:.1f}% price decrease")
                else:
                    signals.append("HOLD")
                    weights.append(0.2)
    
    # 2. Technical indicators signal
    if "indicators" in analysis_result and "error" not in analysis_result["indicators"]:
        indicators = analysis_result["indicators"].get("indicators", {})
        
        rsi_values = indicators.get("rsi", [])
        macd_values = indicators.get("macd", [])
        
        if rsi_values:
            rsi = rsi_values[-1]
            if rsi < 30:
                signals.append("BUY")
                weights.append(0.3)
                reasoning_parts.append("RSI indicates oversold (RSI < 30)")
            elif rsi > 70:
                signals.append("SELL")
                weights.append(0.3)
                reasoning_parts.append("RSI indicates overbought (RSI > 70)")
            else:
                signals.append("HOLD")
                weights.append(0.1)
        
        if macd_values:
            macd = macd_values[-1]
            if macd > 0:
                signals.append("BUY")
                weights.append(0.2)
                reasoning_parts.append("MACD bullish")
            else:
                signals.append("SELL")
                weights.append(0.2)
                reasoning_parts.append("MACD bearish")
    
    # 3. Sentiment signal
    if "sentiment" in analysis_result and "error" not in analysis_result["sentiment"]:
        sentiment = analysis_result["sentiment"]
        if "aggregated" in sentiment:
            agg = sentiment["aggregated"]
            score = agg.get("score", 0)
            
            if score > 0.3:
                signals.append("BUY")
                weights.append(0.2)
                reasoning_parts.append("Positive sentiment")
            elif score < -0.3:
                signals.append("SELL")
                weights.append(0.2)
                reasoning_parts.append("Negative sentiment")
            else:
                signals.append("HOLD")
                weights.append(0.1)
    
    # Aggregate signals
    if not signals:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": "Insufficient data for recommendation",
            "risk_score": 0.5
        }
    
    # Weighted voting
    buy_score = sum(w for s, w in zip(signals, weights) if s == "BUY")
    sell_score = sum(w for s, w in zip(signals, weights) if s == "SELL")
    hold_score = sum(w for s, w in zip(signals, weights) if s == "HOLD")
    
    total_score = buy_score + sell_score + hold_score
    
    if buy_score > sell_score and buy_score > hold_score:
        signal = "BUY"
        confidence = buy_score / total_score if total_score > 0 else 0.5
    elif sell_score > buy_score and sell_score > hold_score:
        signal = "SELL"
        confidence = sell_score / total_score if total_score > 0 else 0.5
    else:
        signal = "HOLD"
        confidence = hold_score / total_score if total_score > 0 else 0.5
    
    # Calculate target price and stop loss
    target_price = None
    stop_loss = None
    
    if "prediction" in analysis_result and "error" not in analysis_result["prediction"]:
        pred = analysis_result["prediction"]
        if "predictions" in pred and len(pred["predictions"]) > 0:
            current_price = pred["predictions"][0]
            
            if signal == "BUY":
                target_price = pred["predictions"][-1] * 1.05  # 5% above prediction
                stop_loss = current_price * 0.95  # 5% below current
            elif signal == "SELL":
                target_price = pred["predictions"][-1] * 0.95  # 5% below prediction
                stop_loss = current_price * 1.05  # 5% above current
    
    # Risk score (higher = riskier)
    risk_score = 1.0 - confidence
    
    return {
        "signal": signal,
        "confidence": float(confidence),
        "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "No specific reasoning available",
        "target_price": float(target_price) if target_price else None,
        "stop_loss": float(stop_loss) if stop_loss else None,
        "risk_score": float(risk_score),
        "signals_breakdown": {
            "buy_score": float(buy_score),
            "sell_score": float(sell_score),
            "hold_score": float(hold_score)
        }
    }
```

**Create directory**: `src/analysis/__init__.py`

```python
"""Analysis modules"""

from .recommendation import generate_recommendation

__all__ = ["generate_recommendation"]
```

---

### 3. Enhanced Python Client

**File**: `src/clients/python_client.py` (NEW FILE)

```python
"""
Python client for RLAIF Trading API

Provides easy-to-use interface for traders.
"""

import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Structured analysis result"""
    symbol: str
    prediction: Optional[Dict] = None
    indicators: Optional[Dict] = None
    sentiment: Optional[Dict] = None
    recommendation: Optional[Dict] = None
    summary: Optional[str] = None
    
    @property
    def signal(self) -> str:
        """Get trading signal"""
        if self.recommendation:
            return self.recommendation.get("signal", "HOLD")
        return "HOLD"
    
    @property
    def confidence(self) -> float:
        """Get confidence score"""
        if self.recommendation:
            return self.recommendation.get("confidence", 0.0)
        return 0.0


class TradingAnalyzer:
    """Client for RLAIF Trading API"""
    
    def __init__(self, endpoint_url: str, api_key: str):
        """
        Initialize client.
        
        Args:
            endpoint_url: RunPod endpoint URL or FastAPI URL
            api_key: API key (for RunPod) or None for FastAPI
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request"""
        if self.api_key:
            # RunPod format
            response = requests.post(
                self.endpoint_url,
                json={"input": payload},
                headers=self.headers,
                timeout=300
            )
        else:
            # FastAPI format
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self.headers,
                timeout=300
            )
        
        response.raise_for_status()
        result = response.json()
        
        if self.api_key:
            # RunPod format
            if result.get("status") == "error":
                raise Exception(result.get("error", "Unknown error"))
            return result.get("output", result)
        else:
            # FastAPI format
            return result
    
    def analyze_ticker(
        self,
        symbol: str,
        horizon: int = 30,
        period: str = "1y",
        include_sentiment: bool = True,
        include_indicators: bool = True,
        include_prediction: bool = True
    ) -> AnalysisResult:
        """
        Analyze a ticker symbol completely.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            horizon: Prediction horizon in days
            period: Data period ("1y", "6mo", "3mo", "1mo")
            include_sentiment: Include sentiment analysis
            include_indicators: Include technical indicators
            include_prediction: Include price prediction
        
        Returns:
            AnalysisResult object
        """
        payload = {
            "action": "analyze",
            "symbol": symbol.upper(),
            "horizon": horizon,
            "period": period,
            "include_sentiment": include_sentiment,
            "include_indicators": include_indicators,
            "include_prediction": include_prediction
        }
        
        result = self._make_request(payload)
        
        return AnalysisResult(
            symbol=result.get("symbol", symbol),
            prediction=result.get("prediction"),
            indicators=result.get("indicators"),
            sentiment=result.get("sentiment"),
            recommendation=result.get("recommendation"),
            summary=result.get("summary")
        )
    
    def predict(self, symbol: str, time_series: List[float], horizon: int = 30) -> Dict:
        """Get price prediction (legacy method)"""
        payload = {
            "action": "predict",
            "symbol": symbol,
            "time_series": time_series,
            "horizon": horizon,
            "return_uncertainty": True
        }
        return self._make_request(payload)
    
    def get_indicators(self, ohlcv: Dict[str, List[float]]) -> Dict:
        """Get technical indicators (legacy method)"""
        payload = {
            "action": "indicators",
            "ohlcv": ohlcv
        }
        return self._make_request(payload)
    
    def analyze_sentiment(self, texts: List[str]) -> Dict:
        """Analyze sentiment (legacy method)"""
        payload = {
            "action": "sentiment",
            "texts": texts,
            "aggregate": True
        }
        return self._make_request(payload)
```

---

## Testing

### Test the New Endpoint

```python
# Test unified analyze endpoint
payload = {
    "input": {
        "action": "analyze",
        "symbol": "AAPL",
        "horizon": 30
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    json=payload,
    headers={"Authorization": "Bearer YOUR_KEY"}
)

print(response.json())
```

### Test Python Client

```python
from src.clients.python_client import TradingAnalyzer

# Initialize client
analyzer = TradingAnalyzer(
    endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    api_key="YOUR_KEY"
)

# Analyze ticker
result = analyzer.analyze_ticker("AAPL", horizon=30)

print(f"Signal: {result.signal}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Recommendation: {result.recommendation}")
print(f"Summary: {result.summary}")
```

---

## Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "yfinance>=0.2.0",  # For fallback data fetching
]
```

---

## Summary

**Files to Create**:
1. `src/analysis/recommendation.py` - Recommendation engine
2. `src/analysis/__init__.py` - Analysis module init
3. `src/clients/python_client.py` - Python SDK

**Files to Modify**:
1. `deployment/runpod/handler.py` - Add `analyze_handler()`
2. `pyproject.toml` - Add `yfinance` dependency

**Estimated Effort**: 1-2 days for core implementation

**Impact**: Transforms system from "functional" to "trader-friendly"
