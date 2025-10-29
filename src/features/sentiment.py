"""Sentiment analysis using FinBERT"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT

    Features:
    - Sentiment scoring for financial news and text
    - Batch processing for efficiency
    - Confidence scoring
    - Aggregation of multiple texts
    - News novelty detection
    """

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name (default: FinBERT)
            device: Device to use (cuda/cpu/mps, auto-detected if None)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        logger.info(f"Loading FinBERT model: {model_name} on {device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping (FinBERT outputs)
        self.labels = ["positive", "negative", "neutral"]

        logger.info("FinBERT model loaded successfully")

    def analyze(
        self,
        texts: Union[str, List[str]],
        return_confidence: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """
        Analyze sentiment of text(s)

        Args:
            texts: Single text or list of texts
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary or list of dictionaries with sentiment results
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self._process_batch(batch, return_confidence)
            results.extend(batch_results)

        return results[0] if single_input else results

    def _process_batch(
        self,
        texts: List[str],
        return_confidence: bool,
    ) -> List[Dict]:
        """Process a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Convert to numpy
        probs_np = probs.cpu().numpy()

        # Create results
        results = []
        for prob in probs_np:
            sentiment_idx = int(np.argmax(prob))
            sentiment = self.labels[sentiment_idx]
            confidence = float(prob[sentiment_idx])

            result = {
                "sentiment": sentiment,
                "score": self._sentiment_to_score(sentiment),
                "confidence": confidence,
            }

            if return_confidence:
                result["probabilities"] = {
                    label: float(prob[i]) for i, label in enumerate(self.labels)
                }

            results.append(result)

        return results

    @staticmethod
    def _sentiment_to_score(sentiment: str) -> float:
        """Convert sentiment label to numerical score"""
        mapping = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0,
        }
        return mapping.get(sentiment, 0.0)

    def aggregate_sentiments(
        self,
        sentiments: List[Dict],
        method: str = "weighted_mean",
        confidence_threshold: float = 0.6,
    ) -> Dict:
        """
        Aggregate multiple sentiment results

        Args:
            sentiments: List of sentiment dictionaries
            method: Aggregation method ("mean", "weighted_mean", "max")
            confidence_threshold: Minimum confidence to include

        Returns:
            Aggregated sentiment dictionary
        """
        # Filter by confidence
        filtered = [s for s in sentiments if s["confidence"] >= confidence_threshold]

        if not filtered:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "count": 0,
            }

        scores = [s["score"] for s in filtered]
        confidences = [s["confidence"] for s in filtered]

        if method == "mean":
            agg_score = np.mean(scores)
            agg_confidence = np.mean(confidences)
        elif method == "weighted_mean":
            weights = np.array(confidences)
            agg_score = np.average(scores, weights=weights)
            agg_confidence = np.mean(confidences)
        elif method == "max":
            # Use sentiment with highest confidence
            max_idx = np.argmax(confidences)
            agg_score = scores[max_idx]
            agg_confidence = confidences[max_idx]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Determine aggregated sentiment
        if agg_score > 0.3:
            agg_sentiment = "positive"
        elif agg_score < -0.3:
            agg_sentiment = "negative"
        else:
            agg_sentiment = "neutral"

        return {
            "sentiment": agg_sentiment,
            "score": float(agg_score),
            "confidence": float(agg_confidence),
            "count": len(filtered),
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        date_column: Optional[str] = None,
        symbol_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of texts

        Args:
            df: DataFrame with text data
            text_column: Column containing text
            date_column: Column with dates (for time-based aggregation)
            symbol_column: Column with symbols (for symbol-based aggregation)

        Returns:
            DataFrame with sentiment columns added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts")

        texts = df[text_column].tolist()
        results = self.analyze(texts)

        # Add results to DataFrame
        df = df.copy()
        df["sentiment"] = [r["sentiment"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]

        return df

    def compute_news_novelty(
        self,
        current_texts: List[str],
        historical_texts: List[str],
        lookback_window: int = 7,
    ) -> float:
        """
        Compute news novelty score (how different from recent history)

        Args:
            current_texts: Current news texts
            historical_texts: Historical news texts
            lookback_window: Days to look back

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if not current_texts or not historical_texts:
            return 0.5

        # Analyze sentiments
        current_sentiments = self.analyze(current_texts)
        historical_sentiments = self.analyze(historical_texts[-lookback_window * 10 :])

        # Aggregate
        current_agg = self.aggregate_sentiments(current_sentiments)
        historical_agg = self.aggregate_sentiments(historical_sentiments)

        # Calculate difference (novelty)
        score_diff = abs(current_agg["score"] - historical_agg["score"])

        # Normalize to 0-1
        novelty = min(score_diff / 2.0, 1.0)

        return float(novelty)


class NewsDataProcessor:
    """
    Process financial news data for sentiment analysis

    Features:
    - News aggregation by time/symbol
    - Deduplication
    - Time-weighted scoring
    - Integration with SentimentAnalyzer
    """

    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize news processor

        Args:
            sentiment_analyzer: SentimentAnalyzer instance (creates new if None)
        """
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()

    def process_news_feed(
        self,
        news_df: pd.DataFrame,
        symbol: str,
        aggregation_window: str = "1D",
        text_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process news feed and compute aggregated sentiment

        Args:
            news_df: DataFrame with news data (must have 'date' and text columns)
            symbol: Stock symbol
            aggregation_window: Time window for aggregation ("1D", "1H", etc.)
            text_columns: Columns to combine for text (default: ["headline", "summary"])

        Returns:
            DataFrame with aggregated sentiment by time period
        """
        if text_columns is None:
            text_columns = ["headline", "summary"]

        # Combine text columns
        news_df = news_df.copy()
        news_df["combined_text"] = news_df[text_columns].fillna("").agg(" ".join, axis=1)

        # Filter for symbol
        if "symbol" in news_df.columns:
            news_df = news_df[news_df["symbol"] == symbol]

        # Analyze sentiment
        news_df = self.sentiment_analyzer.analyze_dataframe(
            news_df, text_column="combined_text", date_column="date"
        )

        # Aggregate by time window
        news_df["date"] = pd.to_datetime(news_df["date"])
        news_df = news_df.set_index("date")

        agg_df = (
            news_df.groupby(pd.Grouper(freq=aggregation_window))
            .agg(
                {
                    "sentiment_score": ["mean", "std", "count"],
                    "sentiment_confidence": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        agg_df.columns = [
            "date",
            "sentiment_mean",
            "sentiment_std",
            "news_count",
            "sentiment_confidence",
        ]

        # Fill missing values
        agg_df["sentiment_std"] = agg_df["sentiment_std"].fillna(0)
        agg_df["news_count"] = agg_df["news_count"].fillna(0)

        return agg_df


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Example financial texts
    texts = [
        "The company reported strong earnings, beating analyst expectations by 15%.",
        "Regulatory concerns and declining market share weigh on the stock.",
        "The quarterly results were in line with expectations.",
    ]

    # Analyze
    results = analyzer.analyze(texts)

    for text, result in zip(texts, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (score: {result['score']:.2f})")
        print(f"Confidence: {result['confidence']:.2f}")

    # Aggregate
    agg = analyzer.aggregate_sentiments(results)
    print(f"\nAggregated: {agg}")
