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
                    reasoning_parts.append(f"Predicted {price_change_pct:.1f}% price change (within ±5%)")
    
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
                reasoning_parts.append(f"RSI neutral ({rsi:.1f})")
        
        if macd_values:
            macd = macd_values[-1]
            if macd > 0:
                signals.append("BUY")
                weights.append(0.2)
                reasoning_parts.append("MACD bullish")
            elif macd < 0:
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
                reasoning_parts.append("Neutral sentiment")
    
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
