"""
News data clients — Finnhub, Polygon, and Alpaca news APIs.

Each client returns a standardized list of news items:
    [{"headline": str, "summary": str, "source": str, "datetime": str, "url": str, "symbol": str}]
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from ...utils.logging import get_logger

logger = get_logger(__name__)


class FinnhubNewsClient:
    """
    Finnhub news API client.

    Free tier: 60 calls/min, company news + general market news.
    Get key at: https://finnhub.io/register
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY required. Get one free at https://finnhub.io/register")
        self._last_call = 0.0

    def get_company_news(
        self,
        symbol: str,
        days_back: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get company-specific news."""
        end = datetime.now()
        start = end - timedelta(days=days_back)

        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/company-news",
            params={
                "symbol": symbol,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "token": self.api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()

        return [
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "finnhub"),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "url": item.get("url", ""),
                "symbol": symbol,
                "category": item.get("category", ""),
            }
            for item in resp.json()
        ]

    def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        """Get general market news."""
        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/news",
            params={"category": category, "token": self.api_key},
            timeout=10,
        )
        resp.raise_for_status()

        return [
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", "finnhub"),
                "datetime": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                "url": item.get("url", ""),
                "symbol": "",
                "category": item.get("category", ""),
            }
            for item in resp.json()
        ]

    def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Finnhub's own social sentiment scores."""
        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/stock/social-sentiment",
            params={"symbol": symbol, "token": self.api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        reddit = data.get("reddit", [])
        twitter = data.get("twitter", [])

        reddit_score = 0.0
        if reddit:
            reddit_score = sum(r.get("score", 0) for r in reddit) / len(reddit)

        twitter_score = 0.0
        if twitter:
            twitter_score = sum(t.get("score", 0) for t in twitter) / len(twitter)

        return {
            "reddit_sentiment": reddit_score,
            "twitter_sentiment": twitter_score,
            "reddit_mentions": sum(r.get("mention", 0) for r in reddit),
            "twitter_mentions": sum(t.get("mention", 0) for t in twitter),
        }

    def get_recommendation_trends(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendation trends."""
        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/stock/recommendation",
            params={"symbol": symbol, "token": self.api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return {"buy": 0, "hold": 0, "sell": 0, "strong_buy": 0, "strong_sell": 0}

        latest = data[0]
        return {
            "buy": latest.get("buy", 0),
            "hold": latest.get("hold", 0),
            "sell": latest.get("sell", 0),
            "strong_buy": latest.get("strongBuy", 0),
            "strong_sell": latest.get("strongSell", 0),
            "period": latest.get("period", ""),
        }

    def _rate_limit(self) -> None:
        """Respect 60 calls/min limit."""
        elapsed = time.time() - self._last_call
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        self._last_call = time.time()


class PolygonNewsClient:
    """
    Polygon.io news API client.

    Free tier: 5 calls/min, limited history.
    Starter ($29/mo): unlimited, full history.
    Get key at: https://polygon.io/
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY required. Get one at https://polygon.io/")
        self._last_call = 0.0

    def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        days_back: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get news articles, optionally filtered by symbol."""
        self._rate_limit()

        params = {
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": self.api_key,
        }
        if symbol:
            params["ticker"] = symbol

        published_after = (datetime.now() - timedelta(days=days_back)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        params["published_utc.gte"] = published_after

        resp = requests.get(
            f"{self.BASE_URL}/v2/reference/news",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            tickers = item.get("tickers", [])
            results.append(
                {
                    "headline": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "source": item.get("publisher", {}).get("name", "polygon"),
                    "datetime": item.get("published_utc", ""),
                    "url": item.get("article_url", ""),
                    "symbol": tickers[0] if tickers else (symbol or ""),
                    "tickers": tickers,
                }
            )

        return results

    def _rate_limit(self) -> None:
        """Respect rate limits (free=5/min, paid=unlimited)."""
        elapsed = time.time() - self._last_call
        if elapsed < 12.5:
            time.sleep(12.5 - elapsed)
        self._last_call = time.time()


class AlpacaNewsClient:
    """
    Alpaca news API — free with any Alpaca account (even paper trading).
    No separate API key needed if you already have Alpaca credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials required for news API.")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def get_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        days_back: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get news from Alpaca (free, included with paper account)."""
        params = {
            "limit": limit,
            "sort": "desc",
        }
        if symbol:
            params["symbols"] = symbol

        start = (datetime.now() - timedelta(days=days_back)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        params["start"] = start

        resp = requests.get(
            "https://data.alpaca.markets/v1beta1/news",
            params=params,
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("news", []):
            symbols = item.get("symbols", [])
            results.append(
                {
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", "alpaca"),
                    "datetime": item.get("created_at", ""),
                    "url": item.get("url", ""),
                    "symbol": symbols[0] if symbols else (symbol or ""),
                }
            )

        return results


class NewsAggregator:
    """
    Aggregates news from all available sources.
    Tries each client, uses whatever's available.
    """

    def __init__(self):
        self.clients = []

        # Always try Alpaca first (free with existing account)
        try:
            self.clients.append(("alpaca", AlpacaNewsClient()))
            logger.info("Alpaca news client initialized")
        except (ValueError, Exception) as e:
            logger.debug(f"Alpaca news not available: {e}")

        # Try Finnhub
        try:
            self.clients.append(("finnhub", FinnhubNewsClient()))
            logger.info("Finnhub news client initialized")
        except (ValueError, Exception) as e:
            logger.debug(f"Finnhub not available: {e}")

        # Try Polygon
        try:
            self.clients.append(("polygon", PolygonNewsClient()))
            logger.info("Polygon news client initialized")
        except (ValueError, Exception) as e:
            logger.debug(f"Polygon not available: {e}")

        if not self.clients:
            logger.warning(
                "No news clients available. Set ALPACA_API_KEY, "
                "FINNHUB_API_KEY, or POLYGON_API_KEY."
            )

    def get_news(
        self,
        symbol: str,
        days_back: int = 7,
        max_articles: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get news from all available sources, deduplicated."""
        all_news = []
        seen_headlines = set()

        for name, client in self.clients:
            try:
                if name == "alpaca":
                    articles = client.get_news(symbol=symbol, days_back=days_back, limit=max_articles)
                elif name == "finnhub":
                    articles = client.get_company_news(symbol=symbol, days_back=days_back)
                elif name == "polygon":
                    articles = client.get_news(symbol=symbol, days_back=days_back, limit=max_articles)
                else:
                    articles = []

                for article in articles:
                    headline = article.get("headline", "").strip()
                    if headline and headline not in seen_headlines:
                        seen_headlines.add(headline)
                        article["_source_api"] = name
                        all_news.append(article)

                logger.info(f"{name}: fetched {len(articles)} articles for {symbol}")

            except Exception as e:
                logger.warning(f"{name} news fetch failed for {symbol}: {e}")

        # Sort by datetime descending
        all_news.sort(key=lambda x: x.get("datetime", ""), reverse=True)

        return all_news[:max_articles]

    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment (Finnhub only)."""
        for name, client in self.clients:
            if name == "finnhub":
                try:
                    return client.get_sentiment(symbol)
                except Exception as e:
                    logger.warning(f"Finnhub social sentiment failed: {e}")
        return {"reddit_sentiment": 0.0, "twitter_sentiment": 0.0}

    def get_analyst_ratings(self, symbol: str) -> str:
        """Get analyst ratings as a formatted string."""
        for name, client in self.clients:
            if name == "finnhub":
                try:
                    recs = client.get_recommendation_trends(symbol)
                    sb = recs.get("strong_buy", 0)
                    b = recs.get("buy", 0)
                    h = recs.get("hold", 0)
                    s = recs.get("sell", 0)
                    ss = recs.get("strong_sell", 0)
                    return f"{sb} Strong Buy, {b} Buy, {h} Hold, {s} Sell, {ss} Strong Sell"
                except Exception as e:
                    logger.warning(f"Finnhub recommendations failed: {e}")
        return "Not available"
