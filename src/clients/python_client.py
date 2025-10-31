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
    
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
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
