"""
Fundamental data feeds — yfinance (free) + SEC EDGAR (free, no key).

yfinance: Income statement, balance sheet, cash flow, key ratios, analyst estimates.
SEC EDGAR: 10-K, 10-Q filings via the free EDGAR API.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ...utils.logging import get_logger

logger = get_logger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class YFinanceFundamentals:
    """
    Free fundamental data via yfinance.
    No API key required. Works immediately.

    Returns standardized dict matching what FundamentalAnalyst expects.
    """

    def __init__(self):
        if not HAS_YFINANCE:
            raise ImportError("yfinance required. Install with: pip install yfinance")

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete fundamental data for a symbol.

        Returns dict with:
        - fundamentals: Key ratios and metrics
        - financials: Raw financial statements as DataFrames
        - market_data: Price and market cap
        - analyst_estimates: Analyst targets and recommendations
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        fundamentals = {
            # Profitability
            "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
            "roa": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else 0,
            "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0,
            "gross_margin": info.get("grossMargins", 0) * 100 if info.get("grossMargins") else 0,
            "operating_margin": info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else 0,
            # Valuation
            "pe_ratio": info.get("trailingPE", 0) or 0,
            "forward_pe": info.get("forwardPE", 0) or 0,
            "pb_ratio": info.get("priceToBook", 0) or 0,
            "ps_ratio": info.get("priceToSalesTrailing12Months", 0) or 0,
            "peg_ratio": info.get("pegRatio", 0) or 0,
            "ev_ebitda": info.get("enterpriseToEbitda", 0) or 0,
            "ev_revenue": info.get("enterpriseToRevenue", 0) or 0,
            # Growth
            "revenue_growth_yoy": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0,
            "earnings_growth_yoy": info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else 0,
            # Leverage
            "debt_to_equity": info.get("debtToEquity", 0) or 0,
            "current_ratio": info.get("currentRatio", 0) or 0,
            "quick_ratio": info.get("quickRatio", 0) or 0,
            # Dividends
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "payout_ratio": info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0,
            # Size
            "market_cap": info.get("marketCap", 0) or 0,
            "enterprise_value": info.get("enterpriseValue", 0) or 0,
        }

        market_data = {
            "current_price": info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0,
            "market_cap": self._format_market_cap(info.get("marketCap", 0)),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0) or 0,
            "52_week_low": info.get("fiftyTwoWeekLow", 0) or 0,
            "50_day_avg": info.get("fiftyDayAverage", 0) or 0,
            "200_day_avg": info.get("twoHundredDayAverage", 0) or 0,
            "beta": info.get("beta", 1.0) or 1.0,
            "avg_volume": info.get("averageVolume", 0) or 0,
        }

        analyst_estimates = {
            "target_high": info.get("targetHighPrice", 0) or 0,
            "target_low": info.get("targetLowPrice", 0) or 0,
            "target_mean": info.get("targetMeanPrice", 0) or 0,
            "target_median": info.get("targetMedianPrice", 0) or 0,
            "recommendation": info.get("recommendationKey", "none"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0) or 0,
        }

        # Industry context
        industry_averages = {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "pe_ratio": info.get("trailingPE", 0) or 0,  # Will be same, but signals context
            "note": "Industry averages require sector ETF comparison for accuracy",
        }

        return {
            "fundamentals": fundamentals,
            "market_data": market_data,
            "analyst_estimates": analyst_estimates,
            "industry_averages": industry_averages,
        }

    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get raw financial statements as DataFrames."""
        ticker = yf.Ticker(symbol)

        statements = {}
        try:
            income = ticker.financials
            if income is not None and not income.empty:
                statements["income"] = income.T
        except Exception as e:
            logger.warning(f"Income statement fetch failed for {symbol}: {e}")

        try:
            balance = ticker.balance_sheet
            if balance is not None and not balance.empty:
                statements["balance"] = balance.T
        except Exception as e:
            logger.warning(f"Balance sheet fetch failed for {symbol}: {e}")

        try:
            cashflow = ticker.cashflow
            if cashflow is not None and not cashflow.empty:
                statements["cashflow"] = cashflow.T
        except Exception as e:
            logger.warning(f"Cash flow fetch failed for {symbol}: {e}")

        return statements

    def get_earnings_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical earnings surprises."""
        ticker = yf.Ticker(symbol)
        try:
            earnings = ticker.earnings_history
            if earnings is not None and not earnings.empty:
                return earnings.to_dict("records")
        except Exception:
            pass
        return []

    @staticmethod
    def _format_market_cap(cap: int) -> str:
        if not cap:
            return "N/A"
        if cap >= 1e12:
            return f"${cap / 1e12:.1f}T"
        if cap >= 1e9:
            return f"${cap / 1e9:.1f}B"
        if cap >= 1e6:
            return f"${cap / 1e6:.1f}M"
        return f"${cap:,.0f}"


class SECEdgarClient:
    """
    SEC EDGAR API client — completely free, no API key.

    Rate limit: 10 requests/second (be respectful).
    https://www.sec.gov/edgar/sec-api-documentation
    """

    BASE_URL = "https://efts.sec.gov/LATEST"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"

    def __init__(self, user_agent: str = "RLAIFTrader/1.0 (trading@example.com)"):
        self.headers = {"User-Agent": user_agent}
        self._last_call = 0.0

    def get_company_filings(
        self,
        symbol: str,
        filing_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent SEC filings for a company.

        Args:
            symbol: Stock ticker
            filing_types: Filter by filing type (e.g., ["10-K", "10-Q", "8-K"])
            limit: Max filings to return
        """
        filing_types = filing_types or ["10-K", "10-Q", "8-K"]

        # First, get CIK from ticker
        cik = self._ticker_to_cik(symbol)
        if not cik:
            logger.warning(f"Could not find CIK for {symbol}")
            return []

        # Get submissions
        self._rate_limit()
        resp = requests.get(
            f"{self.SUBMISSIONS_URL}/CIK{cik}.json",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i, form in enumerate(forms):
            if form in filing_types:
                filings.append(
                    {
                        "form_type": form,
                        "filing_date": dates[i] if i < len(dates) else "",
                        "accession_number": accessions[i] if i < len(accessions) else "",
                        "description": descriptions[i] if i < len(descriptions) else "",
                        "symbol": symbol,
                        "cik": cik,
                    }
                )
                if len(filings) >= limit:
                    break

        logger.info(f"Found {len(filings)} SEC filings for {symbol}")
        return filings

    def get_company_facts(self, symbol: str) -> Dict[str, Any]:
        """
        Get structured financial facts from XBRL filings.
        Returns key financial metrics directly from SEC data.
        """
        cik = self._ticker_to_cik(symbol)
        if not cik:
            return {}

        self._rate_limit()
        resp = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=self.headers,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"SEC company facts not available for {symbol}")
            return {}

        data = resp.json()
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        # Extract key metrics
        result = {}
        key_metrics = {
            "Revenues": "revenue",
            "NetIncomeLoss": "net_income",
            "EarningsPerShareBasic": "eps",
            "Assets": "total_assets",
            "StockholdersEquity": "total_equity",
            "LongTermDebt": "long_term_debt",
            "OperatingIncomeLoss": "operating_income",
        }

        for sec_key, our_key in key_metrics.items():
            if sec_key in us_gaap:
                units = us_gaap[sec_key].get("units", {})
                # Get USD values (most common)
                usd_values = units.get("USD", units.get("USD/shares", []))
                if usd_values:
                    # Get most recent annual (10-K) filing
                    annual = [
                        v for v in usd_values if v.get("form") == "10-K"
                    ]
                    if annual:
                        latest = sorted(annual, key=lambda x: x.get("end", ""))[-1]
                        result[our_key] = latest.get("val", 0)
                        result[f"{our_key}_period"] = latest.get("end", "")

        return result

    def _ticker_to_cik(self, symbol: str) -> Optional[str]:
        """Convert ticker to CIK (Central Index Key)."""
        self._rate_limit()
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == symbol.upper():
                return str(entry["cik_str"]).zfill(10)
        return None

    def _rate_limit(self) -> None:
        """Respect 10 req/sec SEC rate limit."""
        elapsed = time.time() - self._last_call
        if elapsed < 0.15:
            time.sleep(0.15 - elapsed)
        self._last_call = time.time()


class FundamentalDataAggregator:
    """
    Aggregates fundamental data from all available sources.
    yfinance first (fast, comprehensive), SEC EDGAR for depth.
    """

    def __init__(self):
        self.yfinance = None
        self.sec = None

        try:
            self.yfinance = YFinanceFundamentals()
            logger.info("yfinance fundamentals initialized")
        except ImportError as e:
            logger.warning(f"yfinance not available: {e}")

        try:
            self.sec = SECEdgarClient()
            logger.info("SEC EDGAR client initialized")
        except Exception as e:
            logger.warning(f"SEC EDGAR not available: {e}")

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data.
        Merges yfinance ratios with SEC filing info.
        """
        result = {
            "fundamentals": {},
            "market_data": {},
            "analyst_estimates": {},
            "industry_averages": {},
            "sec_filings": [],
            "sec_facts": {},
        }

        # yfinance (primary — fastest, most complete for ratios)
        if self.yfinance:
            try:
                yf_data = self.yfinance.get_fundamentals(symbol)
                result["fundamentals"] = yf_data.get("fundamentals", {})
                result["market_data"] = yf_data.get("market_data", {})
                result["analyst_estimates"] = yf_data.get("analyst_estimates", {})
                result["industry_averages"] = yf_data.get("industry_averages", {})
                logger.info(f"yfinance: loaded {len(result['fundamentals'])} metrics for {symbol}")
            except Exception as e:
                logger.warning(f"yfinance fundamentals failed for {symbol}: {e}")

        # SEC EDGAR (supplementary — authoritative filing data)
        if self.sec:
            try:
                result["sec_filings"] = self.sec.get_company_filings(
                    symbol, filing_types=["10-K", "10-Q", "8-K"], limit=5
                )
            except Exception as e:
                logger.warning(f"SEC filings fetch failed for {symbol}: {e}")

            try:
                result["sec_facts"] = self.sec.get_company_facts(symbol)
            except Exception as e:
                logger.warning(f"SEC facts fetch failed for {symbol}: {e}")

        return result

    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements as DataFrames."""
        if self.yfinance:
            try:
                return self.yfinance.get_financial_statements(symbol)
            except Exception as e:
                logger.warning(f"Financial statements failed for {symbol}: {e}")
        return {}
